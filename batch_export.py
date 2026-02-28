"""
SHIELD — batch_export.py
========================
Headless batch exporter: reads a CSV or Excel file and submits one
Google Earth Engine "toDrive" export task per row.

Each row must have these columns (see batch_template.csv for an example):
  min_lon   – western boundary (float)
  min_lat   – southern boundary (float)
  max_lon   – eastern boundary (float)
  max_lat   – northern boundary (float)
  start_date – YYYY-MM-DD
  end_date   – YYYY-MM-DD
  output_name – name for the exported CSV file on Google Drive

Usage
-----
  python batch_export.py                         # uses batch_template.csv by default
  python batch_export.py my_regions.csv
  python batch_export.py my_regions.xlsx
  python batch_export.py my_regions.csv --no-wait   # submit all tasks, don't wait

Dependencies
------------
  pip install earthengine-api pandas openpyxl
"""

import sys
import time
import logging
import argparse
from pathlib import Path

import pandas as pd
import ee

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_export")

# ── Config ────────────────────────────────────────────────────────────────────
from shield.config import GEE_PROJECT


# ─────────────────────────────────────────────────────────────────────────────
# Core export logic (mirrors gee_simple._run_gee_export but headless)
# ─────────────────────────────────────────────────────────────────────────────

def _build_and_start_task(
    min_lon: float, min_lat: float,
    max_lon: float, max_lat: float,
    start_date: str, end_date: str,
    output_name: str,
) -> "ee.batch.Task":
    """Build and START a single GEE export task. Returns the task object."""

    region = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)

    # ── Rainfall: GPM IMERG V07 Daily ─────────────────────────────────────────
    gpm = (
        ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .select("precipitation")          # normalise band count across the collection
    )

    def make_daily(img):
        date  = img.date().format("YYYY-MM-dd")
        daily = gpm.filterDate(img.date(), img.date().advance(1, "day")).sum()
        rain  = daily.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=1000, maxPixels=1e9
        ).get("precipitation")
        return ee.Feature(None, {"date": date, "rainfall_mm": rain})

    date_list = ee.List.sequence(
        0,
        ee.Date(end_date).difference(ee.Date(start_date), "day").subtract(1)
    ).map(lambda d: ee.Date(start_date).advance(d, "day"))

    dummy_imgs = ee.ImageCollection.fromImages(
        date_list.map(
            lambda d: ee.Image.constant(0).set("system:time_start", ee.Date(d).millis())
        )
    )
    daily_rain_fc = dummy_imgs.map(make_daily)

    # ── Elevation: SRTM ───────────────────────────────────────────────────────
    elev_val = (
        ee.Image("USGS/SRTMGL1_003")
        .select("elevation")
        .reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=90, maxPixels=1e9)
        .getInfo()
        .get("elevation", 50)
    )

    # ── Soil Texture: SoilGrids ───────────────────────────────────────────────
    try:
        soil_raw     = ee.Image("projects/soilgrids-isric/soilgrids/v2.0/wv1500_mean").select("b0")
        soil_val_raw = (
            soil_raw
            .reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=250, maxPixels=1e9)
            .getInfo()
            .get("b0", 4000)
        )
    except Exception:
        soil_val_raw = 4000
    soil_val_raw = soil_val_raw if soil_val_raw is not None else 4000
    soil_texture = min(12, max(1, round(soil_val_raw / 1000)))

    # ── Combine & Dummy Water Columns ─────────────────────────────────────────
    def add_statics(feat):
        return feat.set({
            "elevation":          elev_val,
            "soil_texture":       soil_texture,
            "water_occurrence":   0,
            "water_seasonality":  0,
            "distance_to_water":  1000,
        })

    final_fc = daily_rain_fc.map(add_statics)

    # ── Submit Export ─────────────────────────────────────────────────────────
    task = ee.batch.Export.table.toDrive(
        collection=final_fc,
        description=output_name,
        fileNamePrefix=output_name,
        fileFormat="CSV",
    )
    task.start()
    return task


def _wait_for_task(task: "ee.batch.Task", output_name: str, poll_secs: int = 10):
    """Block until the task finishes. Returns True on success, False on failure."""
    while True:
        info  = task.status()
        state = info["state"]
        if state in ("READY", "RUNNING"):
            pct = info.get("progress", 0) * 100
            log.info("  %-30s  %s  %.0f%%", output_name, state, pct)
            time.sleep(poll_secs)
        elif state == "COMPLETED":
            log.info("  ✅ %-30s  COMPLETED", output_name)
            return True
        else:
            err = info.get("error_message", "unknown error")
            log.error("  ❌ %-30s  %s — %s", output_name, state, err)
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Load input file
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLS = {"min_lon", "min_lat", "max_lon", "max_lat", "start_date", "end_date", "output_name"}


def load_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xls", ".xlsx"):
        df = pd.read_excel(path, dtype=str)
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        raise ValueError(f"Unsupported file type: {suffix}  (use .csv or .xlsx)")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing columns: {sorted(missing)}\n"
            f"Required: {sorted(REQUIRED_COLS)}"
        )
    # Drop rows where ALL required columns are blank (e.g. trailing empty rows)
    df = df.dropna(subset=list(REQUIRED_COLS), how="all").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch GEE exporter — reads CSV/Excel, exports one task per row"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="batch_template.csv",
        help="Path to CSV or Excel file (default: batch_template.csv)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit all tasks and exit immediately (don't wait for completion)",
    )
    parser.add_argument(
        "--poll",
        type=int,
        default=10,
        help="Seconds between status polls when waiting (default: 10)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    # ── Authenticate ──────────────────────────────────────────────────────────
    log.info("Initialising Earth Engine (project: %s)…", GEE_PROJECT)
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception as e:
        log.error(
            "Earth Engine auth failed. Run  python -c \"import ee; ee.Authenticate()\"  first.\n%s", e
        )
        sys.exit(1)
    log.info("Earth Engine ready ✓")

    # ── Load rows ─────────────────────────────────────────────────────────────
    log.info("Loading input: %s", input_path)
    df = load_input(input_path)
    log.info("Found %d row(s) to export", len(df))

    # ── Submit tasks ──────────────────────────────────────────────────────────
    tasks = []
    for idx, row in df.iterrows():
        name = str(row["output_name"]).strip()
        log.info("[%d/%d] Submitting: %s", idx + 1, len(df), name)
        try:
            task = _build_and_start_task(
                min_lon=float(row["min_lon"]),
                min_lat=float(row["min_lat"]),
                max_lon=float(row["max_lon"]),
                max_lat=float(row["max_lat"]),
                start_date=str(row["start_date"]).strip(),
                end_date=str(row["end_date"]).strip(),
                output_name=name,
            )
            tasks.append((name, task))
            log.info("  Submitted ✓  (task id: %s)", task.id)
        except Exception as e:
            log.error("  Failed to submit %s: %s", name, e)

    if not tasks:
        log.warning("No tasks were submitted.")
        return

    log.info("Submitted %d task(s) to Earth Engine.", len(tasks))

    if args.no_wait:
        log.info("--no-wait set. Check progress at: https://code.earthengine.google.com/tasks")
        return

    # ── Wait for all tasks ────────────────────────────────────────────────────
    log.info("Waiting for all tasks to finish… (poll every %ds)", args.poll)
    log.info("(You can also monitor at: https://code.earthengine.google.com/tasks)")
    results = {}
    for name, task in tasks:
        results[name] = _wait_for_task(task, name, poll_secs=args.poll)

    # ── Summary ───────────────────────────────────────────────────────────────
    ok  = [n for n, v in results.items() if v]
    err = [n for n, v in results.items() if not v]
    print("\n" + "=" * 60)
    print(f"BATCH COMPLETE  |  ✅ {len(ok)} succeeded  |  ❌ {len(err)} failed")
    if err:
        print("Failed exports:")
        for n in err:
            print(f"  • {n}")
    print("=" * 60)
    print("Download the CSVs from your Google Drive, then use them in the Train tab.")


if __name__ == "__main__":
    main()
