"""
SHIELD — predict.py
15-day recursive flood probability forecast.

Fixes vs. previous versions:
  ✅ Rainfall sampled from SeasonalRainfallModel (NOT random normal)
  ✅ Same FEATURES list as train.py (config.FEATURES — never out of sync)
  ✅ Same scaler loaded from disk — trained on training set (no leakage)
  ✅ No hardcoded date-range boosts
  ✅ Predictions are deterministic given fixed seed
"""

import logging
import os
from datetime import timedelta
from typing import Callable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import (
    FEATURES, XGB_FEATURES, XGB_INPUT_SIZE,
    MODEL_DIR, SEQ_LENGTH, FUTURE_DAYS, RANDOM_SEED, get_risk_level,
    THRESHOLDS,
)
from .features import create_features, validate_input_columns
from .rainfall import SeasonalRainfallModel
from .weather import get_region_coords, WeatherInputEnsemble

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

PredictionRow = Tuple[pd.Timestamp, float, str, str, float]   # (date, prob, label, colour, rain_mm)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_models(model_dir: Optional[str] = None) -> dict:
    """
    Load all saved SHIELD models from disk.

    Returns
    -------
    dict with keys: lstm, xgb, scaler, rain_model
    """
    from tensorflow.keras.models import load_model

    d = model_dir or MODEL_DIR

    lstm_path   = os.path.join(d, "shield_lstm.keras")
    xgb_path    = os.path.join(d, "shield_xgb.pkl")
    scaler_path = os.path.join(d, "shield_scaler.pkl")
    rain_path   = os.path.join(d, "shield_rain_model.pkl")

    for p in (lstm_path, xgb_path, scaler_path, rain_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Model file not found: {p}\n"
                "Please train the model first using the Train tab."
            )

    log.info("Loading SHIELD models…")
    return {
        "lstm":       load_model(lstm_path),
        "xgb":        joblib.load(xgb_path),
        "scaler":     joblib.load(scaler_path),
        "rain_model": SeasonalRainfallModel.load(rain_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prediction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def predict_flood(
    csv_path: Optional[str] = None,
    df_context: Optional[pd.DataFrame] = None,
    models: Optional[dict] = None,
    model_dir: Optional[str] = None,
    future_days: int = FUTURE_DAYS,
    seed: int = RANDOM_SEED,
    progress_cb: Optional[Callable[[str, float], None]] = None,
    future_rain_list: Optional[List[float]] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    use_weather_api: bool = True,
) -> List[PredictionRow]:
    """
    Generate a 15-day deterministic flood probability forecast.

    Parameters
    ----------
    csv_path        : Path to recent GEE-exported CSV (historical context data).
    models          : Pre-loaded model dict (output of load_models()). If None,
                      models are loaded from model_dir.
    model_dir       : Directory containing saved models (default: config.MODEL_DIR).
    future_days     : Number of days to forecast (default: config.FUTURE_DAYS = 15).
    seed            : Random seed — same seed → same output always (default: 42).
    progress_cb     : Optional callback(message, pct) for GUI progress updates.
    future_rain_list: Explicit list of future daily rainfall values (mm). If
                      provided, overrides both the API and SeasonalRainfallModel.
    lat             : Latitude of the region centroid. Used for Open-Meteo fetch.
    lon             : Longitude of the region centroid. Used for Open-Meteo fetch.
    use_weather_api : If True (default), attempt to fetch real 15-day forecasts
                      from Open-Meteo when lat/lon are available. Falls back to
                      SeasonalRainfallModel automatically on failure.

    Returns
    -------
    List of (date, probability, risk_label, colour) tuples, one per future day.
    """

    def _prog(msg: str, pct: float):
        log.info(f"[{pct:5.1f}%] {msg}")
        if progress_cb:
            progress_cb(msg, pct)

    # ── Load models ───────────────────────────────────────────────────────────
    _prog("Loading models…", 5)
    if models is None:
        models = load_models(model_dir)
    lstm_model  = models["lstm"]
    xgb_model   = models["xgb"]
    scaler      = models["scaler"]
    rain_model  = models["rain_model"]

    # ── Load & validate input ─────────────────────────────────────────────────
    _prog("Loading historical data…", 12)
    if df_context is not None:
        df = df_context.copy()
    elif csv_path is not None:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    else:
        raise ValueError("Must provide either csv_path or df_context")
    validate_input_columns(df)

    # ── Feature engineering on historical data ────────────────────────────────
    _prog("Engineering features on historical data…", 20)
    df_hist = create_features(df)

    if len(df_hist) < SEQ_LENGTH:
        raise ValueError(
            f"Input data has only {len(df_hist)} rows after feature engineering. "
            f"Need at least {SEQ_LENGTH} rows to form an LSTM input sequence."
        )

    # ── Initialise a rolling buffer starting from last historical rows ────────
    _prog("Initialising prediction buffer…", 30)
    df_rolling = df_hist.copy()
    last_date  = df_rolling["date"].iloc[-1]

    # ── Resolve future rainfall source ────────────────────────────────────────
    # Priority: 1) explicit list  2) Open-Meteo API  3) SeasonalRainfallModel
    _resolved_rain_list: Optional[List[float]] = None

    if future_rain_list is not None:
        # Caller supplied exact values (e.g. perfect-weather test)
        _resolved_rain_list = future_rain_list
        _prog("Using supplied future_rain_list (perfect-weather mode).", 32)
        
    elif use_weather_api and lat is not None and lon is not None:
        # Phase 3: Use WeatherInputEnsemble (blends GFS + ICON + Seasonal)
        from datetime import timedelta as _td
        forecast_start = (last_date + _td(days=1)).date() \
            if hasattr(last_date, "date") else last_date + _td(days=1)
            
        _prog(f"Fetching Open-Meteo ensemble for {future_days} days (lat={lat:.3f}, lon={lon:.3f})…", 32)
        ensemble = WeatherInputEnsemble(
            lat=lat, 
            lon=lon, 
            start_date=forecast_start, 
            seasonal_model=rain_model
        )
        _resolved_rain_list = ensemble.get_forecast(days=future_days)
        _prog(f"✅ Weather Ensemble: {len(_resolved_rain_list)} days resolved, "
              f"total={sum(_resolved_rain_list):.1f} mm.", 34)

    else:
        if use_weather_api and (lat is None or lon is None):
            log.info("No lat/lon provided — skipping Open-Meteo, using SeasonalRainfallModel.")

    # Static columns (single-location data — same for all future rows)
    static_cols = {
        col: float(df_rolling[col].iloc[-1])
        for col in ("elevation", "soil_texture",
                    "water_occurrence", "water_seasonality", "distance_to_water")
    }

    # Seeded RNG for deterministic sampling
    rng = np.random.default_rng(seed)

    predictions: List[PredictionRow] = []

    # ── Recursive forecast loop ────────────────────────────────────────────────
    for day in range(1, future_days + 1):
        pct = 30 + 60 * (day / future_days)
        _prog(f"Forecasting day {day}/{future_days}…", pct)

        future_date = last_date + timedelta(days=day)
        future_month = future_date.month

        # 1. Sample next-day rainfall
        # Priority: explicit list > Open-Meteo API > SeasonalRainfallModel
        if _resolved_rain_list is not None and len(_resolved_rain_list) >= day:
            next_rain = _resolved_rain_list[day - 1]
        else:
            next_rain = rain_model.predict(future_month, rng=rng)

        # 2. Append new row to rolling buffer
        # is_forecast=1 for ALL future rows — they are model-generated forecasts,
        # regardless of whether rainfall came from an API or seasonal model.
        new_row = pd.DataFrame([{
            "date":              future_date,
            "rainfall_mm":       next_rain,
            "is_forecast":       1.0,
            **static_cols,
        }])
        df_rolling = pd.concat([df_rolling, new_row], ignore_index=True)

        # 3. Re-run feature engineering on extended buffer
        #    This recomputes rolling totals, API, soil moisture, etc. consistently
        try:
            df_rolling = create_features(df_rolling)
        except Exception as e:
            log.warning(f"Feature re-engineering failed on day {day}: {e}")
            continue

        # 4. Scale the full buffer
        X_buf    = df_rolling[FEATURES].values
        X_scaled = scaler.transform(X_buf)   # uses scaler fitted only on training data

        # 5. Build LSTM sequence from the last SEQ_LENGTH rows
        if len(X_scaled) < SEQ_LENGTH:
            continue
        seq      = X_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, len(FEATURES))
        lstm_prob = float(lstm_model.predict(seq, verbose=0)[0][0])

        # 6. Build XGBoost input (column order from config.XGB_FEATURES)
        last_row  = df_rolling[XGB_FEATURES].iloc[-1].values.reshape(1, -1)
        xgb_input = np.column_stack([[lstm_prob], last_row])
        assert xgb_input.shape[1] == XGB_INPUT_SIZE, (
            f"XGB input shape mismatch: {xgb_input.shape[1]} vs {XGB_INPUT_SIZE}"
        )

        # 7. Final probability: equal-weight ensemble of LSTM + XGBoost
        xgb_prob   = float(xgb_model.predict_proba(xgb_input)[0][1])
        final_prob = 0.5 * lstm_prob + 0.5 * xgb_prob
        final_prob = float(np.clip(final_prob, 0.0, 1.0))

        if day <= 3:
            print(f"DEBUG: Day {day} len: {len(df_rolling)}, rain: {next_rain:.4f}, prob: {final_prob:.4f}")

        risk_label, colour = get_risk_level(final_prob)
        # Resolve the calibrated threshold for this lead-time day
        day_key = f"{day}_day" if f"{day}_day" in THRESHOLDS else "default"
        flood_threshold = THRESHOLDS.get(day_key, THRESHOLDS["default"])
        predictions.append((future_date, final_prob, risk_label, colour, next_rain, flood_threshold))

        log.debug(
            f"Day {day} ({future_date.date()}): "
            f"rain={next_rain:.1f}mm  lstm={lstm_prob:.3f}  "
            f"xgb={xgb_prob:.3f}  final={final_prob:.3f}  {risk_label}"
        )

    _prog(f"✅ Forecast complete — {len(predictions)} days predicted.", 100)
    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: export results to DataFrame / CSV
# ─────────────────────────────────────────────────────────────────────────────

def predictions_to_dataframe(predictions: List[PredictionRow]) -> pd.DataFrame:
    """Convert the list returned by predict_flood() into a tidy DataFrame."""
    rows = []
    for item in predictions:
        # Support both old 5-tuple and new 6-tuple (with flood_threshold)
        if len(item) == 6:
            d, p, l, c, r, th = item
        else:
            d, p, l, c, r = item
            th = 0.50
        rows.append((d.date(), round(p, 4), l, round(r, 4), round(th, 2)))
    return pd.DataFrame(
        rows,
        columns=["date", "flood_probability", "risk_level", "predicted_rainfall_mm", "flood_threshold"],
    )


if __name__ == "__main__":
    import argparse
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="SHIELD — Prediction CLI")
    parser.add_argument("--csv", required=True, help="Path to input CSV for context")
    parser.add_argument("--output", default="shield_forecast_verify.csv", help="Path to save results")
    args = parser.parse_args()

    try:
        preds = predict_flood(csv_path=args.csv)
        df_p = predictions_to_dataframe(preds)
        print("\n--- Forecast Results ---")
        print(df_p.to_string(index=False))
        df_p.to_csv(args.output, index=False)
        print(f"\n✅ Results saved to {args.output}")
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        sys.exit(1)
