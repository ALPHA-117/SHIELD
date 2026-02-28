"""
SHIELD — config.py
Single source of truth for all constants, feature lists, soil params,
and known flood dates. Imported by every other module.
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "saved_models")

# ─────────────────────────────────────────────
# GEE Service Account credentials
# ─────────────────────────────────────────────
GEE_PROJECT         = "shield-488115"
GEE_SERVICE_ACCOUNT = "gee-service-account@shield-488115.iam.gserviceaccount.com"
GEE_KEY_FILE        = os.path.join(BASE_DIR, "shield-488115-ad3bb2e0adfc.json")

# ─────────────────────────────────────────────
# Sequence / forecast settings
# ─────────────────────────────────────────────
SEQ_LENGTH    = 7    # LSTM look-back window (days)
FUTURE_DAYS   = 15   # Forecast horizon (days)
MIN_DATA_ROWS = 90   # Minimum rows needed to train
RANDOM_SEED   = 42

# ─────────────────────────────────────────────
# Feature lists  ← SINGLE SOURCE OF TRUTH
# Both train.py AND predict.py import from here.
# ─────────────────────────────────────────────

# All 18 features fed into the LSTM
FEATURES = [
    "rainfall_mm",
    "elevation",
    "soil_texture",
    "water_occurrence",
    "water_seasonality",
    "distance_to_water",
    "rainfall_1d_total",
    "rainfall_3d_total",
    "rainfall_7d_total",
    "flood_threshold",
    "infiltration_rate",
    "month",
    "is_monsoon",
    "saturation_index",
    "api",
    "soil_moisture",
    "moisture_7d_avg",
    "river_flood_risk",
    "is_forecast",
]

# 12 features extracted from the scaled feature vector and combined
# with the LSTM probability output to form XGBoost input (total = 13)
XGB_FEATURES = [
    "rainfall_1d_total",
    "rainfall_3d_total",
    "rainfall_7d_total",
    "flood_threshold",
    "saturation_index",
    "is_monsoon",
    "soil_moisture",
    "moisture_7d_avg",
    "river_flood_risk",
    "water_occurrence",
    "water_seasonality",
    "distance_to_water",
    "is_forecast",
]
XGB_INPUT_SIZE = 1 + len(XGB_FEATURES)   # 14 total = 1 lstm_prob + 13

# Raw input columns used ONLY for label generation
# (must NOT include any derived/computed features to avoid label leakage)
LABEL_RAW_COLS = ["rainfall_mm", "elevation", "soil_texture"]

# ─────────────────────────────────────────────
# Soil parameters (12 USDA texture classes)
# base_thresh  : baseline daily rain (mm) above which flood risk rises
# infil_rate   : steady infiltration rate (mm/h)
# retention    : soil moisture retention coefficient [0-1]
# evap_coeff   : daily evaporation fraction
# ─────────────────────────────────────────────
SOIL_PARAMS = {
    1:  {"name": "Clay",           "base_thresh": 80, "infil_rate":  1.0, "retention": 0.95, "evap_coeff": 0.04},
    2:  {"name": "Silty clay",     "base_thresh": 78, "infil_rate":  1.2, "retention": 0.90, "evap_coeff": 0.045},
    3:  {"name": "Sandy clay",     "base_thresh": 75, "infil_rate":  3.0, "retention": 0.85, "evap_coeff": 0.05},
    4:  {"name": "Clay loam",      "base_thresh": 72, "infil_rate":  5.0, "retention": 0.80, "evap_coeff": 0.055},
    5:  {"name": "Silty clay loam","base_thresh": 70, "infil_rate":  7.0, "retention": 0.75, "evap_coeff": 0.06},
    6:  {"name": "Sandy clay loam","base_thresh": 68, "infil_rate":  8.0, "retention": 0.70, "evap_coeff": 0.065},
    7:  {"name": "Loam",           "base_thresh": 65, "infil_rate": 15.0, "retention": 0.65, "evap_coeff": 0.07},
    8:  {"name": "Silty loam",     "base_thresh": 60, "infil_rate": 10.0, "retention": 0.60, "evap_coeff": 0.075},
    9:  {"name": "Sandy loam",     "base_thresh": 58, "infil_rate": 25.0, "retention": 0.55, "evap_coeff": 0.08},
    10: {"name": "Silt",           "base_thresh": 62, "infil_rate":  8.0, "retention": 0.65, "evap_coeff": 0.07},
    11: {"name": "Loamy sand",     "base_thresh": 55, "infil_rate": 28.0, "retention": 0.50, "evap_coeff": 0.085},
    12: {"name": "Sand",           "base_thresh": 50, "infil_rate": 30.0, "retention": 0.45, "evap_coeff": 0.09},
}
DEFAULT_SOIL = 7   # Loam fallback for unknown soil type

# ─────────────────────────────────────────────
# Physics-based flood label thresholds
# These must use ONLY raw input columns (no derived features).
# ─────────────────────────────────────────────
FLOOD_LABEL_RAIN_MM          = 40.0  # Daily rainfall (mm) above which flooding is possible
FLOOD_LABEL_RAIN_3D_MM       = 100.0 # 3-day cumulative rainfall (mm)
FLOOD_LABEL_ELEVATION_MAX_M  = 300.0 # Elevation (m) below which flood risk is elevated
FLOOD_LABEL_INFIL_MAX        = 15.0  # Infiltration rate (mm/h) below which soil is prone to saturation

# ─────────────────────────────────────────────
# Known historical flood dates per region
# Add more entries as data becomes available.
# Format: "YYYY-MM-DD"
# ─────────────────────────────────────────────
KNOWN_FLOOD_DATES = {
    "Barpeta": [
        "2023-06-20",   # Confirmed flood onset (from SHIELD.txt)
    ],
    "Jaisalmer": [],    # Arid region — no confirmed flood dates
}

# ─────────────────────────────────────────────
# XGBoost hyperparameters
# ─────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":         200,
    "max_depth":            6,
    "learning_rate":        0.01,
    "subsample":            0.9,
    "colsample_bytree":     0.9,
    "objective":            "binary:logistic",
    "eval_metric":          ["logloss", "aucpr"],
    "early_stopping_rounds": 15,
    "random_state":         RANDOM_SEED,
}

# ─────────────────────────────────────────────
# LSTM hyperparameters
# ─────────────────────────────────────────────
LSTM_PARAMS = {
    "units_1":      64,
    "units_2":      32,
    "dropout":      0.3,
    "dense_units":  16,
    "lr":           0.001,
    "epochs":       50,
    "batch_size":   16,
    "patience":     5,    # EarlyStopping patience
}

# ─────────────────────────────────────────────
# Risk level thresholds for display
# ─────────────────────────────────────────────
RISK_LEVELS = [
    (0.70, "🔴 CATASTROPHIC", "red"),
    (0.50, "🟠 EXTREME",      "orange"),
    (0.30, "🟡 HIGH",         "gold"),
    (0.10, "🟢 MODERATE",     "green"),
    (0.00, "⬜ LOW",           "gray"),
]

# ─────────────────────────────────────────────
# Per-lead-time calibrated flood thresholds (Phase 1)
# Calibrated via calibrate_thresholds.py on Predict Data Rolling.
# Lower values at short lead times = higher recall for near-term warnings.
# Update by running: python calibrate_thresholds.py
# ─────────────────────────────────────────────
THRESHOLDS = {
    "1_day":   0.35,
    "2_day":   0.35,
    "3_day":   0.40,
    "4_day":   0.40,
    "5_day":   0.45,
    "6_day":   0.45,
    "7_day":   0.45,
    "8_day":   0.50,
    "9_day":   0.50,
    "10_day":  0.50,
    "11_day":  0.50,
    "12_day":  0.50,
    "13_day":  0.50,
    "14_day":  0.50,
    "15_day":  0.50,
    "default": 0.50,   # Fallback for any day > 15
}

def get_risk_level(prob: float):
    """Return (label, colour) for a given probability."""
    for threshold, label, colour in RISK_LEVELS:
        if prob >= threshold:
            return label, colour
    return "⬜ LOW", "gray"
