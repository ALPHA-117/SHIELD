import os
import glob
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from shield.features import create_features
from shield.labels import generate_labels
from shield.config import THRESHOLDS

import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

AFTER_DIR = "After Data"

def evaluate_all(predict_dir="Predict Data", output_md_path="Scores.md", is_rolling=False):
    pred_files = glob.glob(os.path.join(predict_dir, "*.csv"))
    
    if not pred_files:
        logging.error(f"No prediction files found in {PREDICT_DIR}")
        return

    all_rain_actual = []
    all_rain_pred = []
    
    all_flood_actual = []
    all_flood_pred_prob = []
    
    valid_files_count = 0
    missing_after_data = 0
    
    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        # Expected after data filename convention: e.g. barpeta_2023.csv -> barpeta_2023_after_data.csv
        name, ext = os.path.splitext(filename)
        after_filename = f"{name}_after_data{ext}"
        after_path = os.path.join(AFTER_DIR, after_filename)
        
        if not os.path.exists(after_path):
            missing_after_data += 1
            logging.warning(f"Missing after data for {filename}: Expected {after_path}")
            continue

        # Load prediction file — format differs between rolling and standard modes
        if is_rolling:
            df_pred = pd.read_csv(pred_path)
            df_pred = df_pred.rename(columns={"target_date": "date"})
            df_pred["date"] = pd.to_datetime(df_pred["date"])
        else:
            df_pred = pd.read_csv(pred_path, parse_dates=["date"])
        
        # Load actuals
        df_after = pd.read_csv(after_path, parse_dates=["date"])
        
        # Generate true flood labels for actual data
        try:
            df_after_feats = create_features(df_after)
            df_actual_labels = generate_labels(df_after_feats)
        except Exception as e:
            logging.error(f"Failed to generate labels for {after_filename}: {e}")
            continue
            
        df_merged = pd.merge(
            df_pred, 
            df_actual_labels, 
            on="date", 
            how="inner",
            suffixes=("_pred", "_actual")
        )
        
        if df_merged.empty:
            logging.warning(f"No matching dates between predictions and actuals for {filename}")
            continue
            
        if is_rolling:
            anchor_date = df_actual_labels["date"].min() - pd.Timedelta(days=1)
            df_merged["lead_time_days"] = (df_merged["date"] - anchor_date).dt.days - df_merged["predicted_on_day"]
            
            # Extract just the relevant info and append to a master list
            pred_flood_prob = df_merged["flood_probability"].values
            actual_flood = df_merged["flood"].values
            lead_times = df_merged["lead_time_days"].values
            
            # Since rolling structure replaces standard rainfall collection, we skip all_rain_actual for rolling mode.
        else:
            # Extract rainfall numbers
            actual_rain = df_merged["rainfall_mm"].values
            if "predicted_rainfall_mm" in df_merged.columns:
                pred_rain = df_merged["predicted_rainfall_mm"].values
                all_rain_actual.extend(actual_rain)
                all_rain_pred.extend(pred_rain)
                
            pred_flood_prob = df_merged["flood_probability"].values
            actual_flood = df_merged["flood"].values
            
        if is_rolling:
            for pf, af, lt in zip(pred_flood_prob, actual_flood, lead_times):
                if lt > 0: # Only evaluate valid future lead times
                    all_flood_pred_prob.append((lt, pf))
                    all_flood_actual.append((lt, af))
        else:
            all_flood_actual.extend(actual_flood)
            all_flood_pred_prob.extend(pred_flood_prob)
            
        valid_files_count += 1
        
    logging.info(f"Evaluated {valid_files_count} files (skipped {missing_after_data} due to missing data)")
    
    if is_rolling:
        lt_breaks = [1, 3, 5, 7, 10, 15]
        rolling_lines = []
        df_eval = pd.DataFrame({
            "lead_time": [x[0] for x in all_flood_actual],
            "actual_flood": [x[1] for x in all_flood_actual],
            "pred_prob": [x[1] for x in all_flood_pred_prob]
        })
        df_eval["pred_flood"] = (df_eval["pred_prob"] >= 0.5).astype(int)
        
        for lt_target in lt_breaks:
            df_lt = df_eval[df_eval["lead_time"] == lt_target]
            if df_lt.empty: continue

            # Use calibrated per-lead-time threshold from config.THRESHOLDS
            lead_key = f"{lt_target}_day"
            threshold = THRESHOLDS.get(lead_key, THRESHOLDS.get("default", 0.50))

            y_t = df_lt["actual_flood"]
            y_p = (df_lt["pred_prob"] >= threshold).astype(int)
            f_prec = precision_score(y_t, y_p, zero_division=0)
            f_rec = recall_score(y_t, y_p, zero_division=0)
            f_f1 = f1_score(y_t, y_p, zero_division=0)
            f_acc = accuracy_score(y_t, y_p)
            n_floods = int(y_t.sum())

            rolling_lines.append(
                f"- **{lt_target}-Day Lead Time** (threshold={threshold:.2f}): "
                f"Accuracy: {f_acc:.3f} | Precision: {f_prec:.3f} | "
                f"Recall: {f_rec:.3f} | F1: {f_f1:.3f} (Total Floods: {n_floods})"
            )

        output_md = f"""# SHIELD Recursive Rolling Evaluation (Cron Job Simulation)
    
Based on generating rolling horizon forecasts against {valid_files_count} historical regions where the model incrementally updates its context daily.
Total evaluation prediction instances: {len(df_eval)}

## Advance Warning Detection Metrics
*How early did the model accurately classify the danger before it happened?*

""" + "\n".join(rolling_lines)

    else:
        # Rainfall
        y_true_rain = np.array(all_rain_actual)
        y_pred_rain = np.array(all_rain_pred)
        
        if len(y_true_rain) > 0:
            mae_rain = mean_absolute_error(y_true_rain, y_pred_rain)
            rmse_rain = np.sqrt(mean_squared_error(y_true_rain, y_pred_rain))
        else:
            mae_rain = rmse_rain = 0.0
            
        # Flood
        y_true_flood = np.array(all_flood_actual)
        y_pred_prob = np.array(all_flood_pred_prob)
        
        # Define a flood prediction as probability >= 0.5
        y_pred_flood = (y_pred_prob >= 0.5).astype(int)
        
        if len(y_true_flood) > 0:
            acc = accuracy_score(y_true_flood, y_pred_flood)
            prec = precision_score(y_true_flood, y_pred_flood, zero_division=0)
            rec = recall_score(y_true_flood, y_pred_flood, zero_division=0)
            f1 = f1_score(y_true_flood, y_pred_flood, zero_division=0)
            total_eval_days = len(y_true_flood)
            actual_flood_days = int(np.sum(y_true_flood))
            pred_flood_days = int(np.sum(y_pred_flood))
        else:
            acc = prec = rec = f1 = 0.0
            total_eval_days = actual_flood_days = pred_flood_days = 0

        # ── Write output to Scores.md ──
        output_md = f"""# SHIELD Model Evaluation Results (15-Day Forecast)
        
Based on generating 15 days of future prediction against {valid_files_count} historical regions.
Total evaluation days: {total_eval_days}

## Rainfall Regression Metrics
*SeasonalRainfallModel performance against actual recorded rainfall over the 15 predicted days.*
- **MAE**: {mae_rain:.2f} mm
- **RMSE**: {rmse_rain:.2f} mm

## Flood classification Metrics
*SHIELD Ensembled Binary Output (Threshold > 0.5) against actual physics-labeled events.*
- **Actual Flood Days**: {actual_flood_days}
- **Predicted Flood Days**: {pred_flood_days}
- **Accuracy**: {acc:.4f}
- **Precision**: {prec:.4f}
- **Recall**: {rec:.4f}
- **F1 Score**: {f1:.4f}
"""
    with open(output_md_path, "w") as f:
        f.write(output_md)
        
    print("\n" + "="*50)
    print("FINISHED EVALUATION")
    print(output_md)
    print("="*50)


def evaluate_operational_feedback():
    """
    Phase 4 Feedback Loop:
    Compares recent Operational Alerts against the newly downloaded GEE ground truth (mocked or real).
    Appends the evaluation to a Model_Drift.log file to track accuracy over time.
    """
    alerts_dir = "Operational_Alerts"
    alert_files = glob.glob(os.path.join(alerts_dir, "*_Alert_*.csv"))
    
    if not alert_files:
        logging.warning("No Operational Alerts found to evaluate.")
        return
        
    drift_log_path = os.path.join(alerts_dir, "Model_Drift.log")
    new_entries = 0
    
    for alert_path in alert_files:
        filename = os.path.basename(alert_path)
        # Parse region from something like 'Barpeta_Alert_2026-02-28.csv'
        parts = filename.split("_Alert_")
        if len(parts) != 2: continue
        
        region = parts[0]
        # Find the most recently downloaded GEE context for this region
        context_files = glob.glob(os.path.join(alerts_dir, f"{region}_ops_*_context.csv"))
        if not context_files:
            continue
            
        latest_context = max(context_files, key=os.path.getmtime)
        
        df_alert = pd.read_csv(alert_path, parse_dates=["forecast_date"])
        df_context = pd.read_csv(latest_context, parse_dates=["date"])
        
        # Generate true flood labels for the context
        try:
            df_context_feats = create_features(df_context)
            df_actual_labels = generate_labels(df_context_feats)
        except Exception as e:
            continue
            
        # We only want to evaluate days that have actually happened (where forecast_date exists in actuals)
        df_merged = pd.merge(
            df_alert,
            df_actual_labels,
            left_on="forecast_date",
            right_on="date",
            how="inner"
        )
        
        if df_merged.empty:
            continue
            
        for _, row in df_merged.iterrows():
            pred_prob = row["flood_probability"]
            actual_flood = row["flood"]
            lead_time = row["lead_time_days"]
            f_date = row["forecast_date"].strftime("%Y-%m-%d")
            
            # Use appropriate threshold
            tier_key = f"{lead_time}_day" if lead_time in [1,3,5,7,10,15] else "default"
            if lead_time not in [1,3,5,7,10,15]:
                # Map to closest bracket
                for b in [1,3,5,7,10,15]:
                    if lead_time <= b:
                        tier_key = f"{b}_day"
                        break
            
            threshold = THRESHOLDS.get(tier_key, 0.50)
            pred_flood = int(pred_prob >= threshold)
            
            outcome = "HIT" if pred_flood == actual_flood else ("FALSE_ALARM" if pred_flood else "MISS")
            if actual_flood == 0 and pred_flood == 0: outcome = "TRUE_NEG"
            
            log_line = f"{f_date} | Region: {region} | Lead: {lead_time}d | Prob: {pred_prob:.3f} (Thr: {threshold:.2f}) | Actual: {actual_flood} | Outcome: {outcome}\n"
            
            with open(drift_log_path, "a", encoding="utf-8") as f:
                f.write(log_line)
                
            new_entries += 1

    if new_entries > 0:
        logging.info(f"Appended {new_entries} new evaluations to {drift_log_path}")
    else:
        logging.info("No new overlapping dates between past alerts and recent ground truth.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--perfect-weather", action="store_true")
    parser.add_argument("--rolling-eval", action="store_true")
    parser.add_argument("--operational-feedback", action="store_true", help="Evaluate operational alerts against recent GEE ground truth")
    args = parser.parse_args()

    if args.operational_feedback:
        evaluate_operational_feedback()
    else:
        if args.rolling_eval:
            predict_dir = "Predict Data Rolling"
            output_md_path = "Recursive_Scores.md"
        else:
            predict_dir = "Predict Data Perfect" if args.perfect_weather else "Predict Data"
            output_md_path = "Perfect_Scores.md" if args.perfect_weather else "Scores.md"
            
        evaluate_all(predict_dir, output_md_path, is_rolling=args.rolling_eval)
