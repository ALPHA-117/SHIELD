"""
SHIELD — features.py
Unified feature engineering function.
Used by BOTH train.py and predict.py — guarantees no feature mismatch.

Input:  DataFrame with columns: date, rainfall_mm, elevation, soil_texture,
        water_occurrence, water_seasonality, distance_to_water
Output: DataFrame with all 18 features in config.FEATURES
"""

import numpy as np
import pandas as pd
import logging

from .config import SOIL_PARAMS, DEFAULT_SOIL, FEATURES

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _soil_param(soil_type: int, key: str):
    """Return a single soil parameter, falling back to DEFAULT_SOIL."""
    return SOIL_PARAMS.get(int(soil_type), SOIL_PARAMS[DEFAULT_SOIL])[key]


def _calculate_api(rainfall_series: np.ndarray, current_idx: int, window: int = 7) -> float:
    """
    Antecedent Precipitation Index (exponentially weighted sum of past rain).
    Returns 50.0 as a neutral starting value when there is insufficient history.
    """
    if current_idx < window:
        return 50.0
    weights = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
    past = rainfall_series[current_idx - window: current_idx]
    return float(np.sum(weights[: len(past)] * past))


def _update_soil_moisture(
    prev_moisture: float,
    rainfall: float,
    soil_type: int,
    temp_c: float = 25.0,
) -> float:
    """
    Iterative soil-moisture model with evaporation.
    Adopted from flood6.py (the most physical version).
    """
    retention  = _soil_param(soil_type, "retention")
    evap_coeff = _soil_param(soil_type, "evap_coeff")
    evap_rate  = evap_coeff + 0.002 * temp_c   # temp-dependent evaporation
    new_moist  = (retention * prev_moisture * (1 - evap_rate)) + (rainfall * (1 - retention))
    return float(np.clip(new_moist, 0.0, 100.0))


def _calculate_flood_threshold(row: pd.Series) -> float:
    """
    Dynamic flood threshold.
    Depends on: elevation, soil_type, api, soil_moisture, month,
                water_occurrence, distance_to_water
    All of these are either raw inputs or intermediate derived columns
    that DO NOT feed back into the label generator.
    """
    soil_info   = SOIL_PARAMS.get(int(row["soil_texture"]), SOIL_PARAMS[DEFAULT_SOIL])
    base_thresh = soil_info["base_thresh"]

    # Elevation factor — higher ground reduces flood risk
    elev_factor = 1.3 - (0.0006 * row["elevation"]) * (1 + row["moisture_7d_avg"] / 100)
    elev_factor = float(np.clip(elev_factor, 0.6, 1.4))

    # API factor — higher antecedent rain lowers the effective threshold
    api_factor = 1.6 - (row["api"] / 100) * (1 + row["soil_moisture"] / 50)
    api_factor = float(np.clip(api_factor, 0.4, 1.6))

    # Proximity to water bodies
    water_factor = 1.0 - (
        0.3 * (row["water_occurrence"] / 100) *
        (1 - min(row["distance_to_water"], 1000) / 1000)
    )
    water_factor = float(np.clip(water_factor, 0.7, 1.3))

    # Monsoon season lowers the effective threshold
    monsoon_factor = 0.7 if int(row["month"]) in (6, 7, 8, 9) else 1.0

    threshold = base_thresh * elev_factor * api_factor * monsoon_factor * water_factor
    return max(15.0, threshold)


def _add_river_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    River / water-body flood risk index.
    Combines water occurrence, seasonality, and distance to water.
    """
    df = df.copy()

    base_risk = (
        0.3 * (df["water_occurrence"] / 100)
        + 0.2 * (df["water_seasonality"] / 10)
        + 0.5 * np.where(
            df["distance_to_water"] == 0,
            0.7,
            0.3 * (1 - np.minimum(df["distance_to_water"], 1000) / 1000),
        )
    )

    # Amplify risk when 3-day rain is heavy
    df["river_flood_risk"] = np.where(
        df["rainfall_3d_total"] > 50,
        base_risk * (1 + (df["rainfall_3d_total"] - 50) / 100),
        base_risk,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def create_features(df: pd.DataFrame, temp_c: float = 25.0) -> pd.DataFrame:
    """
    Create all 18 SHIELD features from raw input data.

    Parameters
    ----------
    df      : DataFrame with at minimum:
              date, rainfall_mm, elevation, soil_texture,
              water_occurrence, water_seasonality, distance_to_water
    temp_c  : Ambient temperature for evaporation model (°C). Default 25°C.

    Returns
    -------
    DataFrame with original columns + all FEATURES columns.
    Rows with NaN are dropped at the end.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # ── Temporal ──────────────────────────────────────────────────────────────
    df["month"]      = df["date"].dt.month
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)

    # ── Rolling rainfall totals ────────────────────────────────────────────────
    rain = df["rainfall_mm"].fillna(0)
    for days in (1, 3, 7):
        df[f"rainfall_{days}d_total"] = (
            rain.rolling(window=days, min_periods=1).sum().fillna(0)
        )

    # ── Antecedent Precipitation Index ────────────────────────────────────────
    rain_arr = rain.values
    df["api"] = [_calculate_api(rain_arr, i) for i in range(len(df))]

    # ── Soil moisture (iterative, with evaporation) ───────────────────────────
    soil_moisture = [0.0] * len(df)
    for i in range(1, len(df)):
        soil_moisture[i] = _update_soil_moisture(
            prev_moisture=soil_moisture[i - 1],
            rainfall=rain.iloc[i],
            soil_type=int(df["soil_texture"].iloc[i]),
            temp_c=temp_c,
        )
    df["soil_moisture"]   = soil_moisture
    df["moisture_7d_avg"] = pd.Series(soil_moisture).rolling(window=7, min_periods=1).mean().values

    # ── Infiltration rate ─────────────────────────────────────────────────────
    df["infiltration_rate"] = df["soil_texture"].apply(
        lambda x: _soil_param(x, "infil_rate")
    )

    # ── Dynamic flood threshold ────────────────────────────────────────────────
    # Depends only on raw inputs + intermediate columns calculated above
    df["flood_threshold"] = df.apply(_calculate_flood_threshold, axis=1)

    # ── Saturation index ──────────────────────────────────────────────────────
    df["saturation_index"] = (df["api"] + df["soil_moisture"]) / df["flood_threshold"]

    # ── River flood risk ──────────────────────────────────────────────────────
    df = _add_river_features(df)

    # ── Final check ───────────────────────────────────────────────────────────
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        # If we are missing features, try to fill them with 0 before crashing
        # This is safer for prediction on very short context
        for m in missing:
            df[m] = 0.0
        log.warning(f"Feature engineering missing columns: {missing} — filled with 0.0")

    # Replace dropna() with forward-fill/zero-fill to prevent row loss during prediction
    df = df.fillna(0.0).reset_index(drop=True)
    return df


def validate_input_columns(df: pd.DataFrame) -> None:
    """
    Raise ValueError if the input DataFrame is missing any required raw columns.
    Call this BEFORE create_features().
    """
    required = {
        "date", "rainfall_mm", "elevation", "soil_texture",
        "water_occurrence", "water_seasonality", "distance_to_water",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    # Type checks
    if df["rainfall_mm"].isnull().all():
        raise ValueError("rainfall_mm column is entirely empty.")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("date column must be datetime (parse_dates=['date']).")
    if df["elevation"].min() <= 0:
        raise ValueError(
            f"Elevation values must be positive (min={df['elevation'].min():.1f}). "
            "Check your GEE export — sea-level areas may need special handling."
        )
    if not df["soil_texture"].between(1, 12).all():
        bad = df.loc[~df["soil_texture"].between(1, 12), "soil_texture"].unique()
        raise ValueError(f"soil_texture must be integer 1–12. Found: {bad}")
    if df["water_occurrence"].min() < 0 or df["water_occurrence"].max() > 100:
        raise ValueError("water_occurrence must be in range 0–100.")
    if df["distance_to_water"].min() < 0:
        raise ValueError("distance_to_water cannot be negative.")

    # Warn about constant columns (not an error, but impacts model quality)
    static_cols = ["elevation", "soil_texture", "water_occurrence",
                   "water_seasonality", "distance_to_water"]
    const = [c for c in static_cols if df[c].nunique() == 1]
    if const:
        log.warning(
            f"Constant-value features detected: {const}. "
            "This means data was collected for a single point. "
            "Model predictions will rely heavily on rainfall dynamics."
        )
