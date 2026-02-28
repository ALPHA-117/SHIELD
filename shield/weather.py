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

def fetch_openmeteo_ensemble(
    lat: float,
    lon: float,
    start_date,       # datetime.date or pd.Timestamp
    days: int = 15,
) -> dict:
    """
    Fetch daily precipitation forecasts from Open-Meteo using multiple models.
    Returns a dictionary of {"model_name": [rain_values...]}.
    """
    try:
        import urllib.request
        import json
    except ImportError as exc:
        raise RuntimeError(f"urllib not available: {exc}") from exc

    if hasattr(start_date, "date"):
        start_date = start_date.date()

    end_date = start_date + timedelta(days=days - 1)
    days_ago = (date.today() - start_date).days
    
    if days_ago > 5:
        # Historical archive does not support multi-model ensemble easily in the free tier
        # so we fall back to the standard best-match archive.
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        models_param = ""
    else:
        # For real-time/recent predictions, explicitly request GFS and ICON models
        base_url = "https://api.open-meteo.com/v1/forecast"
        models_param = "&models=gfs_seamless,icon_seamless"

    url = (
        f"{base_url}"
        f"?latitude={lat:.4f}"
        f"&longitude={lon:.4f}"
        f"&daily=precipitation_sum"
        f"&timezone=Asia%2FKolkata"
        f"{models_param}"
        f"&start_date={start_date.isoformat()}"
        f"&end_date={end_date.isoformat()}"
    )

    log.info("Fetching Open-Meteo ensemble: lat=%.4f lon=%.4f start=%s days=%d",
             lat, lon, start_date, days)

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        raise RuntimeError(f"Open-Meteo request failed: {exc}") from exc

    if "daily" not in data:
        raise RuntimeError(f"Unexpected Open-Meteo response format: {list(data.keys())}")

    results = {}
    daily = data["daily"]
    
    # Process each requested model, or the default if it was an archive request
    keys = [k for k in daily.keys() if k.startswith("precipitation_sum")]
    for key in keys:
        rain_values = daily[key]
        rain_values = [float(v) if v is not None else 0.0 for v in rain_values]
        if len(rain_values) < days:
            rain_values.extend([0.0] * (days - len(rain_values)))
        elif len(rain_values) > days:
            rain_values = rain_values[:days]
        
        # Clean up the key name for the dictionary (e.g. 'precipitation_sum_gfs_seamless' -> 'gfs')
        model_name = key.replace("precipitation_sum", "").strip("_")
        if not model_name: 
            model_name = "default"
        elif "gfs" in model_name:
            model_name = "gfs"
        elif "icon" in model_name:
            model_name = "icon"
            
        results[model_name] = rain_values

    log.info("✅ Open-Meteo ensemble received: %s models, %d days",
             list(results.keys()), len(rain_values))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Weather Input Ensemble Class
# ─────────────────────────────────────────────────────────────────────────────

class WeatherInputEnsemble:
    """
    Orchestrates the blended weather forecast input.
    - Days 1-7: Blends GFS and ICON Open-Meteo models.
    - Days 8-15: Open-Meteo GFS (if available), or falls back to SeasonalRainfallModel.
    """

    def __init__(self, lat: float, lon: float, start_date, seasonal_model=None):
        self.lat = lat
        self.lon = lon
        self.start_date = start_date.date() if hasattr(start_date, "date") else start_date
        self.seasonal_model = seasonal_model
        
    def get_forecast(self, days: int = 15) -> List[float]:
        try:
            ensemble_data = fetch_openmeteo_ensemble(self.lat, self.lon, self.start_date, days)
        except Exception as exc:
            log.warning("Weather APIs failed entirely (%s). Falling back strictly to seasonal model.", exc)
            return self._seasonal_fallback(days)

        final_forecast = []
        for i in range(days):
            current_date = self.start_date + timedelta(days=i)
            day_idx = i + 1 # 1-indexed for logic

            vals = []
            if "icon" in ensemble_data and i < len(ensemble_data["icon"]) and ensemble_data["icon"][i] > 0.0:
                vals.append(ensemble_data["icon"][i])
            if "gfs" in ensemble_data and i < len(ensemble_data["gfs"]):
                vals.append(ensemble_data["gfs"][i])
            elif "default" in ensemble_data and i < len(ensemble_data["default"]):
                vals.append(ensemble_data["default"][i])

            if not vals:
                # API ran out of data for this day, fallback
                if self.seasonal_model:
                    val = self.seasonal_model.predict(month=current_date.month)
                else:
                    val = 0.0
            else:
                if day_idx <= 5:
                    # Short-range: prefer max of available (conservatism for flood warning)
                    val = max(vals)
                else:
                    # Medium/Long-range: average them out to reduce false noise
                    val = sum(vals) / len(vals)
            
            final_forecast.append(val)
            
        return final_forecast

    def _seasonal_fallback(self, days: int) -> List[float]:
        if not self.seasonal_model:
            log.warning("No SeasonalRainfallModel provided, fallback returns all zeros.")
            return [0.0] * days
        
        forecast = []
        for i in range(days):
            m = (self.start_date + timedelta(days=i)).month
            forecast.append(self.seasonal_model.predict(month=m))
        return forecast

