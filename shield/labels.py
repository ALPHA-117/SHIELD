"""
SHIELD — labels.py
Clean, leakage-free flood label generation.

Strategy (two-tier):
  Tier 1 — Known dates (highest priority):
      User provides confirmed flood onset dates; days within ±WINDOW days = 1.

  Tier 2 — Physics thresholds (raw inputs only):
      Uses ONLY raw columns (rainfall_mm, elevation, soil_texture + its
      infiltration rate which is a direct static map, not a derived model feature).
      Does NOT use flood_threshold, saturation_index, api, soil_moisture, etc.
      because those are model features and would cause label leakage.

The labels column is named 'flood' (0 = no flood, 1 = flood event).
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    SOIL_PARAMS,
    DEFAULT_SOIL,
    FLOOD_LABEL_RAIN_MM,
    FLOOD_LABEL_RAIN_3D_MM,
    FLOOD_LABEL_ELEVATION_MAX_M,
    FLOOD_LABEL_INFIL_MAX,
    KNOWN_FLOOD_DATES,
)

log = logging.getLogger(__name__)

# Days around a confirmed flood date to label as "flood"
KNOWN_DATE_WINDOW = 2   # ± 2 days


def _infil_from_soil(soil_type: int) -> float:
    """Direct static lookup — NOT a derived/computed model feature."""
    return SOIL_PARAMS.get(int(soil_type), SOIL_PARAMS[DEFAULT_SOIL])["infil_rate"]


def generate_labels(
    df: pd.DataFrame,
    region: Optional[str] = None,
    extra_flood_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Add a 'flood' column to df using two-tier independent labeling.

    Parameters
    ----------
    df                : Feature-engineered DataFrame (output of features.create_features).
                        Must contain: date, rainfall_mm, elevation, soil_texture.
    region            : Named region key for KNOWN_FLOOD_DATES lookup (e.g. 'Barpeta').
    extra_flood_dates : Additional confirmed flood date strings 'YYYY-MM-DD'.

    Returns
    -------
    DataFrame with 'flood' column added.
    """
    df = df.copy()
    df["flood"] = 0

    # ── Compute rolling rainfall using raw rainfall_mm only ──────────────────
    rain = df["rainfall_mm"].fillna(0)
    rain_3d = rain.rolling(window=3, min_periods=1).sum()

    # ── Compute per-row infiltration (direct static map — not a model feature) ──
    infil = df["soil_texture"].apply(_infil_from_soil)

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 2: Physics-based labels (raw input columns only)
    # Condition: heavy daily rain OR heavy cumulative rain
    #            AND low elevation (flood-prone terrain)
    #            AND low infiltration (impervious soil — water can't absorb)
    # ─────────────────────────────────────────────────────────────────────────
    physics_mask = (
        ((rain > FLOOD_LABEL_RAIN_MM) | (rain_3d > FLOOD_LABEL_RAIN_3D_MM))
        & (df["elevation"] < FLOOD_LABEL_ELEVATION_MAX_M)
        & (infil < FLOOD_LABEL_INFIL_MAX)
    )
    df.loc[physics_mask, "flood"] = 1
    log.info(f"Tier-2 (physics) flood days: {physics_mask.sum()}")

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 1: Known confirmed flood dates (overrides physics, always = 1)
    # ─────────────────────────────────────────────────────────────────────────
    known_dates: List[pd.Timestamp] = []

    if region and region in KNOWN_FLOOD_DATES:
        for ds in KNOWN_FLOOD_DATES[region]:
            try:
                known_dates.append(pd.Timestamp(ds))
            except Exception:
                log.warning(f"Could not parse known flood date: {ds!r}")

    if extra_flood_dates:
        for ds in extra_flood_dates:
            try:
                known_dates.append(pd.Timestamp(ds))
            except Exception:
                log.warning(f"Could not parse extra flood date: {ds!r}")

    for fd in known_dates:
        window_mask = (
            (df["date"] >= fd - pd.Timedelta(days=KNOWN_DATE_WINDOW))
            & (df["date"] <= fd + pd.Timedelta(days=KNOWN_DATE_WINDOW))
        )
        df.loc[window_mask, "flood"] = 1
        log.info(f"Tier-1 (known date {fd.date()}) labelled {window_mask.sum()} rows as flood=1")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    counts = df["flood"].value_counts().to_dict()
    total  = len(df)
    flood_pct = 100 * counts.get(1, 0) / total if total > 0 else 0
    log.info(
        f"Label distribution → flood=0: {counts.get(0,0)}, "
        f"flood=1: {counts.get(1,0)} ({flood_pct:.1f}%)"
    )

    if counts.get(1, 0) < 3:
        log.warning(
            "Very few flood events found in labels. "
            "If this region is flood-prone, consider providing known_flood_dates "
            "or collecting more historical data. Model accuracy may be limited."
        )

    return df


def label_summary(df: pd.DataFrame) -> Dict:
    """Return a summary dict of label statistics (useful for GUI display)."""
    counts     = df["flood"].value_counts().to_dict()
    total      = len(df)
    flood_days = counts.get(1, 0)
    safe_days  = counts.get(0, 0)
    return {
        "total_rows":   total,
        "flood_days":   flood_days,
        "safe_days":    safe_days,
        "flood_pct":    round(100 * flood_days / total, 2) if total else 0,
        "class_ratio":  round(safe_days / flood_days, 1) if flood_days > 0 else float("inf"),
    }
