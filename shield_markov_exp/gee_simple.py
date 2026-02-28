"""
SHIELD — gee_simple.py
Simplified GEE Data Exporter (Tkinter GUI)

Extracts: Rainfall (GPM IMERG) + Elevation (SRTM) + Soil Texture (SoilGrids)
Use this for a quick basic dataset when you DON'T need river/water-body features.
Best combined with flood_train_xgboost (Phase 1 training only).

Based on: FloodData2.py + GEEDataExtractor.py (merged best of both)
"""

import threading
import tkinter as tk
from tkinter import messagebox, ttk
import logging

log = logging.getLogger(__name__)


def _run_gee_export(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float,
    start_date: str, end_date: str,
    output_name: str,
    status_var: tk.StringVar,
    progress_var: tk.DoubleVar,
    root: tk.Tk,
):
    """
    Background worker: authenticate, build request, export to Google Drive.
    All GEE calls are in this thread — never blocks the Tkinter event loop.
    """
    try:
        import ee
        from shield.config import GEE_PROJECT, GEE_SERVICE_ACCOUNT, GEE_KEY_FILE

        def _set(msg: str, pct: float = None):
            status_var.set(msg)
            if pct is not None:
                progress_var.set(pct)
            root.update_idletasks()

        _set("Authenticating with Earth Engine…", 5)
        # Service accounts have NO Google Drive quota, so toDrive() exports always fail
        # with "Service accounts do not have storage quota". We must use the personal
        # OAuth credentials saved by ee.Authenticate() instead.
        # If credentials are missing, prompt the user to run ee.Authenticate() once.
        try:
            ee.Initialize(project=GEE_PROJECT)
        except Exception as auth_err:
            raise RuntimeError(
                "Earth Engine OAuth credentials not found.\n\n"
                "Please run this once in a Python shell:\n"
                "  import ee\n"
                "  ee.Authenticate()\n\n"
                "Then retry the export."
            ) from auth_err

        _set("Defining region of interest…", 15)
        region = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)

        # ── Rainfall: GPM IMERG Daily ─────────────────────────────────────────
        _set("Loading rainfall dataset (GPM IMERG)…", 25)
        # Select 'precipitation' band first to make the collection homogeneous.
        # V07 mixes images with 5 and 9 bands; selecting one band normalises it.
        # Band was renamed from 'precipitationCal' (V06) to 'precipitation' (V07).
        gpm = ee.ImageCollection("NASA/GPM_L3/IMERG_V07") \
               .filterDate(start_date, end_date) \
               .filterBounds(region) \
               .select("precipitation")

        def make_daily(img):
            date = img.date().format("YYYY-MM-dd")
            daily = gpm.filterDate(img.date(), img.date().advance(1, "day")).sum()
            rain  = daily.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=1000, maxPixels=1e9
            ).get("precipitation")
            return ee.Feature(None, {"date": date, "rainfall_mm": rain})

        date_range  = ee.DateRange(start_date, end_date)
        date_list   = ee.List.sequence(
            0,
            ee.Date(end_date).difference(ee.Date(start_date), "day").subtract(1)
        ).map(lambda d: ee.Date(start_date).advance(d, "day"))
        dummy_imgs  = ee.ImageCollection.fromImages(
            date_list.map(lambda d: ee.Image.constant(0).set("system:time_start", ee.Date(d).millis()))
        )
        daily_rain_fc = dummy_imgs.map(make_daily)

        # ── Elevation: SRTM ───────────────────────────────────────────────────
        _set("Loading elevation (SRTM)…", 45)
        elevation = ee.Image("USGS/SRTMGL1_003").select("elevation")
        elev_val  = elevation.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=90, maxPixels=1e9
        ).getInfo().get("elevation", 50)

        # ── Soil Texture: SoilGrids ───────────────────────────────────────────
        _set("Loading soil texture (SoilGrids)…", 65)
        # SoilGrids wv1500 (wilting-point water content, 0-5 cm)
        # Current GEE asset path as of 2024; falls back to a mid-range value if unavailable.
        try:
            soil_raw = ee.Image("projects/soilgrids-isric/soilgrids/v2.0/wv1500_mean").select("b0")
            soil_val_raw = soil_raw.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=250, maxPixels=1e9
            ).getInfo().get("b0", 4000)
        except Exception:
            soil_val_raw = 4000  # fallback: mid-range wilting point
        # Map SoilGrids wilting-volume to approximate USDA class (1–12)
        soil_val_raw = soil_val_raw if soil_val_raw is not None else 4000
        soil_texture = min(12, max(1, round(soil_val_raw / 1000)))

        # ── Combine & Add Static Columns ──────────────────────────────────────
        _set("Combining features…", 80)

        def add_statics(feat):
            return feat.set({
                "elevation":    elev_val,
                "soil_texture": soil_texture,
                # Dummy water columns so the CSV is compatible with features.py
                "water_occurrence":   0,
                "water_seasonality":  0,
                "distance_to_water":  1000,
            })

        final_fc = daily_rain_fc.map(add_statics)

        # ── Export to Drive ───────────────────────────────────────────────────
        _set("Starting export to Google Drive…", 88)
        task = ee.batch.Export.table.toDrive(
            collection=final_fc,
            description=output_name,
            fileNamePrefix=output_name,
            fileFormat="CSV",
        )
        task.start()

        # Monitor without blocking
        import time
        while task.status()["state"] in ("READY", "RUNNING"):
            info = task.status()
            desc = info.get("description", "")
            pct  = 88 + int(0.12 * info.get("progress", 0) * 100)
            _set(f"Export running… ({desc})", pct)
            time.sleep(8)

        final_state = task.status()["state"]
        if final_state == "COMPLETED":
            _set(f"✅ Exported '{output_name}.csv' to Google Drive!", 100)
            messagebox.showinfo(
                "Export Complete",
                f"'{output_name}.csv' has been saved to your Google Drive.\n"
                "Download it and use it in the Train tab."
            )
        else:
            err = task.status().get("error_message", "Unknown error")
            _set(f"❌ Export failed: {err}", 0)
            messagebox.showerror("Export Failed", err)

    except Exception as e:
        status_var.set(f"❌ Error: {e}")
        log.exception("GEE Simple export error")
        messagebox.showerror("Error", str(e))


class GEESimpleFrame(tk.Frame):
    """
    Tab frame for the Simple GEE exporter.
    Drop this into any Tkinter Notebook with .add(GEESimpleFrame(nb), text="Export (Simple)").
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        lbl_w = 24

        tk.Label(self, text="Simple GEE Exporter",
                 font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=2, pady=(12, 6))
        tk.Label(self, text="Exports: Rainfall · Elevation · Soil Texture",
                 font=("Segoe UI", 9), fg="gray").grid(row=1, column=0, columnspan=2)
        tk.Label(self, text="(No river features — use Export (Full) for complete dataset)",
                 font=("Segoe UI", 8), fg="gray").grid(row=2, column=0, columnspan=2, pady=(0, 10))

        fields = [
            ("Min Longitude:",  "88.5"),
            ("Min Latitude:",   "26.0"),
            ("Max Longitude:",  "91.5"),
            ("Max Latitude:",   "27.0"),
            ("Start Date (YYYY-MM-DD):", "2023-04-01"),
            ("End Date (YYYY-MM-DD):",   "2023-07-31"),
            ("Output File Name:",        "barpeta_simple"),
        ]
        self._vars = {}
        for r, (label, default) in enumerate(fields, start=3):
            tk.Label(self, text=label, width=lbl_w, anchor="w").grid(row=r, column=0, **pad)
            var = tk.StringVar(value=default)
            tk.Entry(self, textvariable=var, width=30).grid(row=r, column=1, **pad)
            key = label.split(":")[0].lower().replace(" ", "_").replace("(", "").replace(")", "")
            self._vars[key] = var

        r = 3 + len(fields)
        tk.Button(self, text="▶  Start Export", command=self._start,
                  bg="#2196F3", fg="white", padx=12).grid(row=r, column=0, columnspan=2, pady=10)

        self._status_var   = tk.StringVar(value="Ready.")
        self._progress_var = tk.DoubleVar()
        tk.Label(self, textvariable=self._status_var, wraplength=420,
                 font=("Segoe UI", 9)).grid(row=r+1, column=0, columnspan=2, **pad)
        ttk.Progressbar(self, variable=self._progress_var,
                        maximum=100, length=420).grid(row=r+2, column=0, columnspan=2, **pad)

    def _start(self):
        v = self._vars
        try:
            min_lon = float(v["min_longitude"].get())
            min_lat = float(v["min_latitude"].get())
            max_lon = float(v["max_longitude"].get())
            max_lat = float(v["max_latitude"].get())
        except ValueError:
            messagebox.showerror("Invalid input", "Coordinates must be numbers.")
            return

        start = v["start_date_yyyy-mm-dd"].get().strip()
        end   = v["end_date_yyyy-mm-dd"].get().strip()
        name  = v["output_file_name"].get().strip() or "shield_export_simple"

        root = self.winfo_toplevel()
        thread = threading.Thread(
            target=_run_gee_export,
            args=(min_lon, min_lat, max_lon, max_lat,
                  start, end, name,
                  self._status_var, self._progress_var, root),
            daemon=True,
        )
        thread.start()
