"""
calibrate_thresholds.py
Phase 1: Precision-Recall calibration for SHIELD predictions.

Loads rolling-evaluation predictions (Predict Data Rolling/*.csv) and
After Data ground-truth, computes optimal thresholds per lead-time
bracket, and prints a ready-to-paste THRESHOLDS dict for shield/config.py.

Usage:
    python calibrate_thresholds.py
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from shield.features import create_features
from shield.labels import generate_labels
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

PREDICT_DIR = "Predict Data Rolling"
AFTER_DIR   = "After Data"

# Lead-time day values to report individually
LEAD_TIME_DAYS = [1, 2, 3, 4, 5, 6, 7, 10, 15]

# Lead-time brackets for the final THRESHOLDS dict
BRACKETS = [
    (1,  2,  "1–2 day"),
    (3,  4,  "3–4 day"),
    (5,  7,  "5–7 day"),
    (8,  15, "8–15 day"),
]

THRESHOLD_RANGE = np.round(np.arange(0.20, 0.81, 0.05), 2)


def load_eval_data() -> pd.DataFrame:
    """Load and merge prediction files with ground-truth labels."""
    pred_files = glob.glob(os.path.join(PREDICT_DIR, "*.csv"))
    if not pred_files:
        logging.error(f"No prediction files found in {PREDICT_DIR}/")
        return pd.DataFrame()

    all_data = []

    for pred_path in pred_files:
        filename     = os.path.basename(pred_path)
        after_path   = os.path.join(AFTER_DIR, filename.replace(".csv", "_after_data.csv"))

        if not os.path.exists(after_path):
            continue

        df_pred = pd.read_csv(pred_path)
        df_pred = df_pred.rename(columns={"target_date": "date"})
        df_pred["date"] = pd.to_datetime(df_pred["date"])

        df_after = pd.read_csv(after_path, parse_dates=["date"])

        try:
            df_after_feats   = create_features(df_after)
            df_actual_labels = generate_labels(df_after_feats)
        except Exception:
            continue

        df_merged = pd.merge(df_pred, df_actual_labels, on="date", how="inner")
        if df_merged.empty:
            continue

        anchor_date = df_actual_labels["date"].min() - pd.Timedelta(days=1)
        df_merged["lead_time_days"] = (
            (df_merged["date"] - anchor_date).dt.days - df_merged["predicted_on_day"]
        )
        df_merged = df_merged[df_merged["lead_time_days"] > 0]

        for _, row in df_merged.iterrows():
            all_data.append({
                "lead_time": int(row["lead_time_days"]),
                "prob":      float(row["flood_probability"]),
                "actual":    int(row["flood"]),
            })

    df_eval = pd.DataFrame(all_data)
    logging.info(f"Total evaluation instances loaded: {len(df_eval)}")
    return df_eval


def best_threshold_for_subset(df_lt: pd.DataFrame, prefer: str = "f1") -> tuple:
    """
    Return (best_threshold, best_precision, best_recall, best_f1).
    Prefers maximising F1. Falls back to 0.50 if no floods in subset.
    """
    if df_lt.empty or df_lt["actual"].sum() == 0:
        return 0.50, 0.0, 0.0, 0.0

    best_f1 = -1.0
    best_th = 0.50
    best_p = best_r = 0.0

    for th in THRESHOLD_RANGE:
        y_true = df_lt["actual"]
        y_pred = (df_lt["prob"] >= th).astype(int)
        p  = precision_score(y_true, y_pred, zero_division=0)
        r  = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th, best_p, best_r = f1, th, p, r

    return float(best_th), float(best_p), float(best_r), float(best_f1)


def calibrate():
    df_eval = load_eval_data()
    if df_eval.empty:
        logging.error("No data to calibrate. Aborting.")
        return

    # ── Per-day detailed table ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DETAILED PRECISION-RECALL TABLE BY LEAD TIME")
    print("=" * 65)

    for lt in LEAD_TIME_DAYS:
        df_lt     = df_eval[df_eval["lead_time"] == lt]
        n_floods  = int(df_lt["actual"].sum())
        n_total   = len(df_lt)
        if df_lt.empty or n_floods == 0:
            logging.info(f"\n→ Lead Time {lt:2d} day(s): No data / no flood events. Skipping.")
            continue

        logging.info(f"\n=== Lead Time: {lt} Day(s)  "
                     f"(Floods: {n_floods} / {n_total} = {100*n_floods/n_total:.1f}%) ===")
        print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
        print("-" * 50)

        best_f1 = 0.0
        for th in THRESHOLD_RANGE:
            y_true = df_lt["actual"]
            y_pred = (df_lt["prob"] >= th).astype(int)
            p  = precision_score(y_true, y_pred, zero_division=0)
            r  = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            marker = ""
            if f1 > best_f1:
                best_f1 = f1
                marker  = "<-- Best F1"
            print(f"{th:<10.2f} | {p:<10.3f} | {r:<10.3f} | {f1:<10.3f} {marker}")

    # ── Bracket-level calibration → THRESHOLDS dict ───────────────────────────
    print("\n" + "=" * 65)
    print("BRACKET-LEVEL CALIBRATION SUMMARY")
    print("=" * 65)

    recommended = {}
    bracket_results = {}

    for lo, hi, label in BRACKETS:
        df_bracket = df_eval[
            (df_eval["lead_time"] >= lo) & (df_eval["lead_time"] <= hi)
        ]
        th, p, r, f1 = best_threshold_for_subset(df_bracket)
        n_floods = int(df_bracket["actual"].sum())
        bracket_results[(lo, hi)] = (th, p, r, f1)
        print(f"  {label:<9}: best threshold = {th:.2f}  "
              f"(P={p:.3f} R={r:.3f} F1={f1:.3f})  floods={n_floods}")

        for day in range(lo, hi + 1):
            recommended[f"{day}_day"] = th

    recommended["default"] = 0.50

    # ── Print copy-pasteable THRESHOLDS dict ──────────────────────────────────
    print("\n" + "=" * 65)
    print("RECOMMENDED config.THRESHOLDS  (copy into shield/config.py)")
    print("=" * 65)
    print("THRESHOLDS = {")
    for day in range(1, 16):
        key = f"{day}_day"
        th  = recommended.get(key, 0.50)
        # Group bracket label
        bracket_label = next(
            (lbl for lo, hi, lbl in BRACKETS if lo <= day <= hi), "default"
        )
        print(f'    "{key}":   {th:.2f},   # {bracket_label}')
    print(f'    "default": {recommended["default"]:.2f},   # fallback')
    print("}")
    print("=" * 65)


if __name__ == "__main__":
    calibrate()
