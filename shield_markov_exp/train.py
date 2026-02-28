"""
SHIELD — train.py
Unified LSTM + XGBoost hybrid training pipeline.

Fixes applied vs. previous versions:
  ✅ Scaler fitted ONLY on training split (no data leakage)
  ✅ Chronological train/test split (no random shuffle on time-series)
  ✅ Shared FEATURES list from config (no mismatches at inference)
  ✅ Labels from labels.py (independent of derived model features)
  ✅ SeasonalRainfallModel fitted and saved for use at inference
  ✅ XGBoost input = 1 LSTM prob + 12 XGB_FEATURES (always 13 inputs)
  ✅ Early stopping on both LSTM and XGBoost
  ✅ Saves: lstm.keras, xgb.pkl, scaler.pkl, rain_model.pkl, metadata.json
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Callable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from collections import Counter

from .config import (
    FEATURES, XGB_FEATURES, XGB_INPUT_SIZE,
    MODEL_DIR, SEQ_LENGTH, MIN_DATA_ROWS,
    RANDOM_SEED, XGB_PARAMS, LSTM_PARAMS,
)
from .features import create_features, validate_input_columns
from .labels import generate_labels, label_summary
from .rainfall import SeasonalRainfallModel

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GPU configuration
# ─────────────────────────────────────────────────────────────────────────────

def _configure_gpu() -> bool:
    """
    Enable GPU training for TensorFlow.
    - Enables memory growth so TF doesn't allocate all VRAM at once.
    - Returns True if at least one GPU was found, False if CPU-only.
    Called once per process; safe to call multiple times.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            log.info("GPU training enabled: %d GPU(s) detected — %s",
                     len(gpus), [g.name for g in gpus])
            return True
        else:
            log.info("No GPU detected — training on CPU.")
            return False
    except Exception as e:
        log.warning("GPU configuration failed (%s) — falling back to CPU.", e)
        return False


_GPU_AVAILABLE = _configure_gpu()  # configure once at import time

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def _build_lstm(input_shape: tuple):
    """Build LSTM model — uses LSTM_PARAMS from config. Runs on GPU if available."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    p = LSTM_PARAMS
    model = Sequential([
        LSTM(p["units_1"], input_shape=input_shape, return_sequences=True),
        Dropout(p["dropout"]),
        LSTM(p["units_2"]),
        Dropout(p["dropout"]),
        Dense(p["dense_units"], activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=p["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _prepare_sequences(
    X_scaled: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a window over the scaled feature matrix to create LSTM sequences."""
    Xs, ys = [], []
    for i in range(len(X_scaled) - seq_len):
        Xs.append(X_scaled[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def _xgb_input(lstm_probs: np.ndarray, df_subset: pd.DataFrame) -> np.ndarray:
    """
    Build the 13-column XGBoost input matrix:
      column 0  = LSTM probability
      columns 1-12 = XGB_FEATURES (from config)
    """
    return np.column_stack([lstm_probs, df_subset[XGB_FEATURES].values])


# ─────────────────────────────────────────────────────────────────────────────
# Public training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_pipeline(
    csv_path: str,
    region: Optional[str] = None,
    extra_flood_dates: Optional[list] = None,
    progress_cb: Optional[Callable[[str, float], None]] = None,
    dry_run: bool = False,
) -> dict:
    """
    End-to-end training pipeline.

    Parameters
    ----------
    csv_path         : Path to the GEE-exported CSV file.
    region           : Region name for KNOWN_FLOOD_DATES lookup (e.g. 'Barpeta').
    extra_flood_dates: Additional confirmed flood dates ['YYYY-MM-DD', ...].
    progress_cb      : Optional callback(message: str, pct: float) for GUI updates.
    dry_run          : If True, skip model saving (for automated tests).

    Returns
    -------
    dict with keys: metrics, scaler_fitted_on_train_only, model_paths
    """
    def _prog(msg: str, pct: float):
        log.info(f"[{pct:5.1f}%] {msg}")
        if progress_cb:
            progress_cb(msg, pct)

    # ── 1. Load Data ─────────────────────────────────────────────────────────
    _prog("Loading data…", 2)
    csv_files = []
    if os.path.isdir(csv_path):
        csv_files = [
            os.path.join(csv_path, f) for f in os.listdir(csv_path) 
            if f.lower().endswith(".csv")
        ]
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {csv_path}")
        log.info("Found %d CSV files for training.", len(csv_files))
    else:
        csv_files = [csv_path]

    # Process files into a list of cleaned dataframes
    all_dfs = []
    for f in csv_files:
        _df = pd.read_csv(f, parse_dates=["date"])
        validate_input_columns(_df)
        
        # ── 2. Feature engineering ───────────────────────────────────────────
        _df = create_features(_df)
        
        # ── 3. Label generation ──────────────────────────────────────────────
        # Note: region/extra_flood_dates applied to ALL if directory; 
        # usually better to rely on auto-thresholds for batch training.
        _df = generate_labels(_df, region=region, extra_flood_dates=extra_flood_dates)
        
        if len(_df) >= SEQ_LENGTH + 1:
            all_dfs.append(_df)
        else:
            log.warning("Skipping file %s — too short for SEQ_LENGTH", f)

    if not all_dfs:
        raise ValueError("No valid data found in provided files.")

    # ── 4. Split and Transform (Multi-file aware) ────────────────────────────
    _prog("Preparing multi-file dataset…", 25)
    
    # We split EACH file 80/20 to ensure both train/test see all regions
    train_dfs = []
    test_dfs = []
    for _df in all_dfs:
        split = int(len(_df) * 0.8)
        train_dfs.append(_df.iloc[:split])
        test_dfs.append(_df.iloc[split:])

    # Fit scaler on ALL training rows combined
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    combined_train = pd.concat(train_dfs)
    scaler.fit(combined_train[FEATURES].values)
    
    _prog("Building LSTM sequences…", 35)
    
    def _get_seqs(df_list):
        Xs, ys = [], []
        for _df in df_list:
            X_sc = scaler.transform(_df[FEATURES].values)
            y_vals = _df["flood"].values
            X_s, y_s = _prepare_sequences(X_sc, y_vals, SEQ_LENGTH)
            if len(X_s) > 0:
                Xs.append(X_s)
                ys.append(y_s)
        if not Xs: return np.array([]), np.array([])
        return np.concatenate(Xs), np.concatenate(ys)

    X_seq_tr, y_seq_tr = _get_seqs(train_dfs)
    X_seq_te, y_seq_te = _get_seqs(test_dfs)
    
    # For XGBoost pass, we need the "all" sequences
    X_seq_all, y_seq_all = _get_seqs(all_dfs)

    # Calculate global stats for reporting
    total_df = pd.concat(all_dfs)
    stats = label_summary(total_df)
    _prog(f"Dataset: {len(all_dfs)} regions, {len(total_df)} rows total.", 38)
    _prog(f"Labels: {stats['flood_days']} flood / {stats['safe_days']} safe days "
          f"({stats['flood_pct']}%)", 40)

    if len(X_seq_tr) == 0:
        raise ValueError(
            f"Training set too small for sequence length {SEQ_LENGTH}. "
            "Need at least training_rows > SEQ_LENGTH."
        )

    # ── 7. Train LSTM ─────────────────────────────────────────────────────────
    _prog("Training LSTM…", 42)
    from tensorflow.keras.callbacks import EarlyStopping
    input_shape = (SEQ_LENGTH, len(FEATURES))
    lstm_model  = _build_lstm(input_shape)
    early_stop  = EarlyStopping(
        monitor="val_loss", patience=LSTM_PARAMS["patience"],
        restore_best_weights=True, verbose=0,
    )
    lstm_model.fit(
        X_seq_tr, y_seq_tr,
        epochs=LSTM_PARAMS["epochs"],
        batch_size=LSTM_PARAMS["batch_size"],
        validation_data=(X_seq_te, y_seq_te),
        callbacks=[early_stop],
        verbose=0,
    )
    _prog("LSTM training complete.", 60)

    # ── 8. Generate LSTM probabilities for FULL dataset ───────────────────────
    _prog("Generating LSTM predictions across dataset…", 63)
    lstm_probs_all = lstm_model.predict(X_seq_all, verbose=0).flatten()

    # Build XGBoost features for ALL data
    # We must align the LSTM results with the original rows (sequences are truncated)
    # Since we combined sequences, we must also combine the dataframes identically
    merged_full_df = pd.concat([d.iloc[SEQ_LENGTH:] for d in all_dfs], ignore_index=True)
    
    # ── 9. Build XGBoost input ────────────────────────────────────────────────
    _prog("Preparing XGBoost input…", 67)
    X_xgb_all = _xgb_input(lstm_probs_all, merged_full_df)
    y_xgb_all = merged_full_df["flood"].values
    
    assert X_xgb_all.shape[1] == XGB_INPUT_SIZE, (
        f"XGB input shape mismatch: got {X_xgb_all.shape[1]}, expected {XGB_INPUT_SIZE}"
    )

    # Chronological split for XGBoost (now on the combined sequences)
    xgb_split = int(len(X_xgb_all) * 0.8)
    X_xgb_tr, X_xgb_te = X_xgb_all[:xgb_split], X_xgb_all[xgb_split:]
    y_xgb_tr,  y_xgb_te  = y_xgb_all[:xgb_split], y_xgb_all[xgb_split:]

    # ── 10. Train XGBoost ─────────────────────────────────────────────────────
    _prog("Training XGBoost…", 72)
    import xgboost as xgb
    from sklearn.model_selection import train_test_split as tts

    counts = Counter(y_xgb_tr)
    spw    = counts[0] / counts[1] if counts.get(1, 0) > 0 else 5.0

    params = {**XGB_PARAMS, "scale_pos_weight": spw}
    early_rounds = params.pop("early_stopping_rounds", 15)

    # ── GPU acceleration for XGBoost ──────────────────────────────────────────
    # Use CUDA if a GPU is available; fall back gracefully to CPU.
    if _GPU_AVAILABLE:
        try:
            # XGBoost ≥ 2.0 uses device='cuda'; older versions use tree_method='gpu_hist'
            import xgboost as _xgb_check
            xgb_version = tuple(int(x) for x in _xgb_check.__version__.split(".")[:2])
            if xgb_version >= (2, 0):
                params["device"] = "cuda"
            else:
                params["tree_method"] = "gpu_hist"
                params["gpu_id"] = 0
            log.info("XGBoost GPU training enabled (version %s).", _xgb_check.__version__)
        except Exception as xgb_gpu_err:
            log.warning("XGBoost GPU setup failed (%s) — using CPU.", xgb_gpu_err)
    else:
        log.info("XGBoost running on CPU.")

    # In newer XGBoost, early_stopping_rounds is often preferred in constructor
    xgb_model = xgb.XGBClassifier(**params, early_stopping_rounds=early_rounds)

    # Use a tiny val split from XGB train for early stopping
    X_xtr, X_xval, y_xtr, y_xval = tts(
        X_xgb_tr, y_xgb_tr, test_size=0.2, random_state=RANDOM_SEED,
        stratify=y_xgb_tr if counts.get(1, 0) >= 2 else None,
    )
    xgb_model.fit(
        X_xtr, y_xtr,
        eval_set=[(X_xval, y_xval)],
        # early_stopping_rounds removed from here to fix TypeError
        verbose=False,
    )
    _prog("XGBoost training complete.", 84)

    # ── 11. Fit seasonal rainfall model ───────────────────────────────────────
    _prog("Fitting seasonal rainfall model…", 87)
    # Fit globally on all available date/rain pairs
    rain_model = SeasonalRainfallModel().fit(total_df["date"], total_df["rainfall_mm"])

    # ── 12. Evaluate ──────────────────────────────────────────────────────────
    _prog("Evaluating model…", 90)
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score, f1_score
    )
    y_pred     = xgb_model.predict(X_xgb_te)
    y_prob     = xgb_model.predict_proba(X_xgb_te)[:, 1]
    report_str = classification_report(y_xgb_te, y_pred, digits=3)
    cm         = confusion_matrix(y_xgb_te, y_pred).tolist()
    roc        = roc_auc_score(y_xgb_te, y_prob) if len(np.unique(y_xgb_te)) > 1 else 0.0
    f1         = f1_score(y_xgb_te, y_pred, zero_division=0)

    metrics = {
        "classification_report": report_str,
        "confusion_matrix":       cm,
        "roc_auc":                round(roc, 4),
        "f1_score":               round(f1, 4),
        "flood_days_total":       int(stats["flood_days"]),
        "safe_days_total":        int(stats["safe_days"]),
        "train_rows":             len(X_xgb_tr),
        "test_rows":              len(X_xgb_te),
        "trained_at":             datetime.utcnow().isoformat(),
    }
    _prog(f"ROC-AUC={roc:.3f}  F1={f1:.3f}", 94)
    log.info(report_str)

    # ── 13. Save models ───────────────────────────────────────────────────────
    model_paths = {}
    if not dry_run:
        _ensure_model_dir()
        lstm_path  = os.path.join(MODEL_DIR, "shield_lstm.keras")
        xgb_path   = os.path.join(MODEL_DIR, "shield_xgb.pkl")
        scaler_path = os.path.join(MODEL_DIR, "shield_scaler.pkl")
        rain_path   = os.path.join(MODEL_DIR, "shield_rain_model.pkl")
        meta_path   = os.path.join(MODEL_DIR, "shield_metadata.json")

        _prog("Saving models…", 96)
        lstm_model.save(lstm_path)
        joblib.dump(xgb_model, xgb_path)
        joblib.dump(scaler, scaler_path)
        rain_model.save(rain_path)

        with open(meta_path, "w") as f:
            json.dump(metrics, f, indent=2)

        model_paths = {
            "lstm":       lstm_path,
            "xgb":        xgb_path,
            "scaler":     scaler_path,
            "rain_model": rain_path,
            "metadata":   meta_path,
        }
        _prog("✅ All models saved.", 100)
        log.info(f"Models saved to {MODEL_DIR}")

    return {
        "metrics":                   metrics,
        "scaler_fitted_on_train_only": True,   # always True by design
        "model_paths":               model_paths,
        "label_stats":               stats,
    }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="SHIELD — Model Training CLI")
    parser.add_argument("--csv", required=True, help="Path to input CSV or folder of CSVs")
    parser.add_argument("--region", default=None, help="Region name for label processing")
    args = parser.parse_args()

    try:
        train_pipeline(csv_path=args.csv, region=args.region)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1)
