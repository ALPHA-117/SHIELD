"""
SHIELD — gee_full.py
Full GEE Data Exporter with River / Water-Body Features (Tkinter GUI)

Extracts:
  - Rainfall (GPM IMERG daily)
  - Elevation (SRTM 90m)
  - Soil Texture (SoilGrids)
  - Water Occurrence & Seasonality (JRC Global Surface Water)
  - Distance to Water (computed via JRC)

Use this to build the complete 18-feature training dataset for the SHIELD model.
Based on: GEE2.py (best production version), improved with better error handling.
"""

import threading
import tkinter as tk
from tkinter import messagebox, ttk
import logging

log = logging.getLogger(__name__)


def _run_full_export(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float,
    start_date: str, end_date: str,
    output_name: str,
    status_var: tk.StringVar,
    progress_var: tk.DoubleVar,
    root: tk.Tk,
):
    """
    Background worker: extract ALL SHIELD features and export to Google Drive.
    """
    try:
        import ee
        import math
        from shield.config import GEE_PROJECT, GEE_SERVICE_ACCOUNT, GEE_KEY_FILE

        def _set(msg: str, pct: float = None):
            status_var.set(msg)
            if pct is not None:
                progress_var.set(pct)
            root.update_idletasks()

        _set("Authenticating with Earth Engine…", 5)
        credentials = ee.ServiceAccountCredentials(GEE_SERVICE_ACCOUNT, GEE_KEY_FILE)
        ee.Initialize(credentials=credentials, project=GEE_PROJECT)

        _set("Defining region & loading static layers…", 12)
        region = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)

        # ── Elevation ─────────────────────────────────────────────────────────
        elevation = ee.Image("USGS/SRTMGL1_003").select("elevation")
        elev_val  = elevation.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=90, maxPixels=1e9
        ).getInfo().get("elevation", 50)

        # ── Soil Texture (SoilGrids → approx USDA class 1-12) ─────────────────
        soil_raw     = ee.Image("projects/soilgrids-isric/properties/wv1500").select("mean")
        soil_raw_val = soil_raw.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=250, maxPixels=1e9
        ).getInfo().get("mean", 4000)
        soil_texture = min(12, max(1, round((soil_raw_val or 4000) / 1000)))

        # ── JRC Water: Occurrence & Seasonality ───────────────────────────────
        _set("Loading JRC water layers…", 25)
        jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        water_occ = jrc.select("occurrence").reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=30, maxPixels=1e9
        ).getInfo().get("occurrence", 0) or 0

        water_seas = jrc.select("seasonality").reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=30, maxPixels=1e9
        ).getInfo().get("seasonality", 0) or 0

        # ── Distance to Water ─────────────────────────────────────────────────
        _set("Computing distance to water…", 38)
        water_mask    = jrc.select("occurrence").gt(50)   # pixels with >50% water occurrence
        dist_to_water_img = water_mask.fastDistanceTransform(
            neighborhood=1024, units="pixels"
        ).sqrt().multiply(30)  # convert pixels to metres (30m/pixel)

        dist_val = dist_to_water_img.reduceRegion(
            reducer=ee.Reducer.min(), geometry=region, scale=30, maxPixels=1e9
        ).getInfo().get("occurrence", 1000) or 1000      # default 1km if no water nearby

        _set("Static layers loaded. Building daily rainfall time series…", 48)

        # ── Rainfall: GPM IMERG (30-min → daily sum) ─────────────────────────
        gpm = ee.ImageCollection("NASA/GPM_L3/IMERG_V06") \
               .filterDate(start_date, end_date) \
               .filterBounds(region) \
               .select("precipitationCal")

        # Generate list of date strings between start and end
        n_days = (
            ee.Date(end_date).difference(ee.Date(start_date), "day")
        ).getInfo()

        def make_daily_feature(offset):
            day_start = ee.Date(start_date).advance(offset, "day")
            day_end   = day_start.advance(1, "day")
            date_str  = day_start.format("YYYY-MM-dd")
            daily_sum = gpm.filterDate(day_start, day_end).sum()
            rain_val  = daily_sum.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=1000, maxPixels=1e9
            ).get("precipitationCal")
            return ee.Feature(None, {
                "date":             date_str,
                "rainfall_mm":      rain_val,
                "elevation":        elev_val,
                "soil_texture":     soil_texture,
                "water_occurrence": water_occ,
                "water_seasonality": water_seas,
                "distance_to_water": dist_val,
            })

        _set("Mapping daily features over date range…", 60)
        offsets    = ee.List.sequence(0, n_days - 1)
        feature_fc = ee.FeatureCollection(offsets.map(make_daily_feature))

        # ── Export to Google Drive ────────────────────────────────────────────
        _set("Starting export to Google Drive (this may take a few minutes)…", 78)
        task = ee.batch.Export.table.toDrive(
            collection=feature_fc,
            description=output_name,
            fileNamePrefix=output_name,
            fileFormat="CSV",
            selectors=[
                "date", "rainfall_mm", "elevation", "soil_texture",
                "water_occurrence", "water_seasonality", "distance_to_water",
            ],
        )
        task.start()

        # Poll without blocking Tkinter
        import time
        while task.status()["state"] in ("READY", "RUNNING"):
            status_info = task.status()
            pct  = 78 + int(0.22 * status_info.get("progress", 0) * 100)
            desc = status_info.get("description", "")
            _set(f"Export in progress… ({desc})", pct)
            time.sleep(8)

        final = task.status()["state"]
        if final == "COMPLETED":
            _set(f"✅ '{output_name}.csv' exported to Google Drive!", 100)
            messagebox.showinfo(
                "Export Complete",
                f"Full dataset '{output_name}.csv' saved to your Google Drive.\n\n"
                "Columns: date, rainfall_mm, elevation, soil_texture,\n"
                "         water_occurrence, water_seasonality, distance_to_water\n\n"
                "Download it and use it in the Train tab."
            )
        else:
            err = task.status().get("error_message", "Unknown error")
            _set(f"❌ Export failed: {err}", 0)
            messagebox.showerror("Export Failed", f"GEE task failed:\n{err}")

    except ImportError:
        msg = (
            "Earth Engine Python library not installed.\n"
            "Run: pip install earthengine-api"
        )
        status_var.set("❌ " + msg)
        messagebox.showerror("Missing Library", msg)
    except Exception as e:
        status_var.set(f"❌ Error: {e}")
        log.exception("GEE Full export error")
        messagebox.showerror("Error", str(e))


class GEEFullFrame(tk.Frame):
    """
    Tab frame for the Full GEE exporter.
    Drop into any Tkinter Notebook: nb.add(GEEFullFrame(nb), text="Export (Full)").
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        lbl_w = 28

        tk.Label(self, text="Full GEE Exporter — All SHIELD Features",
                 font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=2, pady=(12, 2))
        tk.Label(self,
                 text="Exports: Rainfall · Elevation · Soil Texture · Water Occurrence · "
                      "Water Seasonality · Distance to Water",
                 font=("Segoe UI", 9), fg="#1565C0", wraplength=500).grid(
            row=1, column=0, columnspan=2, pady=(0, 10))

        fields = [
            ("Min Longitude (West):",         "89.5"),
            ("Min Latitude (South):",         "26.0"),
            ("Max Longitude (East):",         "91.5"),
            ("Max Latitude (North):",         "27.0"),
            ("Start Date (YYYY-MM-DD):",      "2022-01-01"),
            ("End Date (YYYY-MM-DD):",        "2023-12-31"),
            ("Output File Name:",             "barpeta_full_shield"),
        ]
        self._vars = {}
        self._row_offset = 2
        for idx, (label, default) in enumerate(fields):
            r = self._row_offset + idx
            tk.Label(self, text=label, width=lbl_w, anchor="w").grid(row=r, column=0, **pad)
            var = tk.StringVar(value=default)
            tk.Entry(self, textvariable=var, width=32).grid(row=r, column=1, **pad)
            safe_key = (label.split(":")[0].strip()
                        .lower().replace(" ", "_")
                        .replace("(", "").replace(")", "")
                        .replace("-", "_"))
            self._vars[safe_key] = var

        btn_row = self._row_offset + len(fields)
        tk.Button(self, text="▶  Start Full Export", command=self._start,
                  bg="#1B5E20", fg="white", padx=12, pady=4).grid(
            row=btn_row, column=0, columnspan=2, pady=12)

        self._status_var   = tk.StringVar(value="Ready.")
        self._progress_var = tk.DoubleVar()
        tk.Label(self, textvariable=self._status_var, wraplength=460,
                 font=("Segoe UI", 9)).grid(row=btn_row+1, column=0, columnspan=2, **pad)
        ttk.Progressbar(self, variable=self._progress_var,
                        maximum=100, length=460).grid(row=btn_row+2, column=0, columnspan=2, **pad)

        tk.Label(self,
                 text="ℹ️  Export may take several minutes for long date ranges. "
                      "The tab remains responsive while uploading.",
                 font=("Segoe UI", 8), fg="gray", wraplength=480).grid(
            row=btn_row+3, column=0, columnspan=2, pady=(4, 0))

    def _start(self):
        v = self._vars
        try:
            min_lon = float(v["min_longitude_west"].get())
            min_lat = float(v["min_latitude_south"].get())
            max_lon = float(v["max_longitude_east"].get())
            max_lat = float(v["max_latitude_north"].get())
        except (ValueError, KeyError):
            messagebox.showerror("Invalid Input", "Coordinates must be valid numbers.")
            return

        start = v.get("start_date_yyyy_mm_dd", v.get("start_date", tk.StringVar())).get().strip()
        end   = v.get("end_date_yyyy_mm_dd",   v.get("end_date",   tk.StringVar())).get().strip()
        name  = v.get("output_file_name",       tk.StringVar(value="shield_full")).get().strip()
        name  = name or "shield_full_export"

        root = self.winfo_toplevel()
        threading.Thread(
            target=_run_full_export,
            args=(min_lon, min_lat, max_lon, max_lat,
                  start, end, name,
                  self._status_var, self._progress_var, root),
            daemon=True,
        ).start()
