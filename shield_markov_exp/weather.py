"""
SHIELD — weather.py
Open-Meteo integration for real 15-day rainfall forecasts.

Replaces SeasonalRainfallModel guessing with actual NWP model output.
No API key required. Free and open-source.

Usage
-----
    from .weather import fetch_openmeteo_rain, get_region_coords

    lat, lon = get_region_coords("barpeta_2023")   # -> (26.25, 91.05)
    rain_list = fetch_openmeteo_rain(lat, lon, start_date=date(2023, 7, 15), days=15)
    # [2.1, 0.0, 14.5, 38.2, ...]  — one value per day, mm
"""

import logging
import os
from datetime import date, timedelta
from typing import List, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)

# Default path for the region bounding-box lookup table
_DEFAULT_TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "batch_template.csv",
)


# ─────────────────────────────────────────────────────────────────────────────
# Region coordinate lookup
# ─────────────────────────────────────────────────────────────────────────────

def get_region_coords(
    filename_stem: str,
    template_csv: str = _DEFAULT_TEMPLATE,
) -> Optional[Tuple[float, float]]:
    """
    Look up the centroid (lat, lon) for a region from batch_template.csv.

    Parameters
    ----------
    filename_stem : str
        The CSV filename without extension, e.g. ``"barpeta_2023"``.
    template_csv  : str
        Path to batch_template.csv (default: auto-detected from project root).

    Returns
    -------
    (lat, lon) tuple of floats, or None if the region is not listed.
    """
    if not os.path.exists(template_csv):
        log.warning("batch_template.csv not found at %s — cannot look up coords.", template_csv)
        return None

    try:
        df = pd.read_csv(template_csv)
        row = df[df["output_name"] == filename_stem]
        if row.empty:
            log.warning("Region '%s' not found in batch_template.csv.", filename_stem)
            return None

        r = row.iloc[0]
        lat = (float(r["min_lat"]) + float(r["max_lat"])) / 2.0
        lon = (float(r["min_lon"]) + float(r["max_lon"])) / 2.0
        log.debug("Region '%s' → centroid lat=%.4f lon=%.4f", filename_stem, lat, lon)
        return lat, lon

    except Exception as exc:
        log.warning("Failed to read coords from batch_template.csv: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Open-Meteo API fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_openmeteo_rain(
    lat: float,
    lon: float,
    start_date,       # datetime.date or pd.Timestamp
    days: int = 15,
) -> List[float]:
    """
    Fetch daily precipitation forecasts from Open-Meteo (free, no API key).

    Parameters
    ----------
    lat        : Latitude (degrees north).
    lon        : Longitude (degrees east).
    start_date : First day to forecast — typically `last_hist_date + 1 day`.
    days       : Number of days to fetch (≤ 16 for free tier).

    Returns
    -------
    List of ``days`` floats representing daily rainfall in mm.

    Raises
    ------
    RuntimeError
        If the API is unreachable or returns an error. Caller should catch
        this and fall back to SeasonalRainfallModel.
    """
    try:
        import urllib.request
        import json
    except ImportError as exc:
        raise RuntimeError(f"urllib not available: {exc}") from exc

    # Convert to datetime.date if needed
    if hasattr(start_date, "date"):
        start_date = start_date.date()

    end_date = start_date + timedelta(days=days - 1)
    
    # Open-Meteo forecast API only goes back ~1-2 weeks. For historical
    # dataset evaluation (e.g. 2017-2023), we must use the archive API to
    # simulate what the forecast *would* have been.
    days_ago = (date.today() - start_date).days
    
    if days_ago > 5:
        base_url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        base_url = "https://api.open-meteo.com/v1/forecast"

    url = (
        f"{base_url}"
        f"?latitude={lat:.4f}"
        f"&longitude={lon:.4f}"
        f"&daily=precipitation_sum"
        f"&timezone=Asia%2FKolkata"
        f"&start_date={start_date.isoformat()}"
        f"&end_date={end_date.isoformat()}"
    )

    log.info("Fetching Open-Meteo forecast: lat=%.4f lon=%.4f start=%s days=%d",
             lat, lon, start_date, days)
    log.debug("URL: %s", url)

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        raise RuntimeError(f"Open-Meteo request failed: {exc}") from exc

    if "daily" not in data or "precipitation_sum" not in data["daily"]:
        raise RuntimeError(f"Unexpected Open-Meteo response format: {list(data.keys())}")

    rain_values = data["daily"]["precipitation_sum"]

    # API can return None for dates beyond its horizon — convert to 0.0
    rain_values = [float(v) if v is not None else 0.0 for v in rain_values]

    # Pad or truncate to exactly `days` values
    if len(rain_values) < days:
        log.warning("Open-Meteo returned %d values, expected %d — padding with 0.0",
                    len(rain_values), days)
        rain_values.extend([0.0] * (days - len(rain_values)))
    elif len(rain_values) > days:
        rain_values = rain_values[:days]

    log.info("✅ Open-Meteo forecast received: %d days, total=%.1f mm",
             len(rain_values), sum(rain_values))
    return rain_values
