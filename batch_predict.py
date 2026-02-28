import os
import glob
import logging
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from shield.predict import predict_flood, load_models, predictions_to_dataframe
from shield.weather import get_region_coords

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# GPU configuration  (mirrors train.py — called once in the main process)
# ─────────────────────────────────────────────────────────────────────────────

def _configure_gpu() -> bool:
    """
    Enable TensorFlow GPU memory growth so VRAM is claimed on first use
    rather than pre-allocated entirely at startup.
    Returns True when at least one GPU is detected.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info("GPU enabled: %d GPU(s) detected — %s",
                         len(gpus), [g.name for g in gpus])
            return True
        logging.info("No GPU detected — running on CPU.")
        return False
    except Exception as exc:
        logging.warning("GPU configuration failed (%s) — falling back to CPU.", exc)
        return False


_GPU_AVAILABLE = _configure_gpu()  # configure once, here in the main process



def process_file(csv_path, args, out_dir, models=None):
    filename = os.path.basename(csv_path)
    out_path = os.path.join(out_dir, filename)
    
    # Skip if already predicted
    if os.path.exists(out_path):
        return f"Skipping {filename} (already exists)"
        
    try:
        name, ext = os.path.splitext(filename)

        # ── Resolve lat/lon for this region (used by Open-Meteo) ────────────
        coords = get_region_coords(name)
        lat = coords[0] if coords else None
        lon = coords[1] if coords else None

        if args.rolling_eval:
            after_filename = f"{name}_after_data{ext}"
            after_path = os.path.join("After Data", after_filename)
            
            if not os.path.exists(after_path):
                return f"Skipped {filename} (missing after data)"
                
            df_hist = pd.read_csv(csv_path, parse_dates=["date"])
            df_after = pd.read_csv(after_path, parse_dates=["date"])
            # Use pre-loaded models when provided (GPU single-process path)
            if models is None:
                models = load_models(model_dir=os.path.join("shield", "saved_models"))
            
            all_rolling_preds = []
            for d in range(15):
                if d > 0:
                    df_ctx = pd.concat([df_hist, df_after.iloc[:d]], ignore_index=True)
                else:
                    df_ctx = df_hist.copy()
                    
                future_d = 15 - d
                if future_d <= 0:
                    break
                    
                preds = predict_flood(
                    df_context=df_ctx, models=models, progress_cb=None,
                    future_days=future_d, lat=lat, lon=lon,
                )
                for date, prob, label, color, rain, threshold in preds:
                    all_rolling_preds.append({
                        "predicted_on_day": d,
                        "target_date": date.strftime("%Y-%m-%d"),
                        "flood_probability": round(prob, 4),
                        "risk_level": label,
                        "predicted_rainfall_mm": round(rain, 4),
                        "flood_threshold": round(threshold, 2),
                    })
                    
            df_out = pd.DataFrame(all_rolling_preds)
            df_out.to_csv(out_path, index=False)
            return f"Processed {filename} -> Rolling CSV"

        future_rain_list = None
        if args.perfect_weather:
            after_filename = f"{name}_after_data{ext}"
            after_path = os.path.join("After Data", after_filename)
            if os.path.exists(after_path):
                df_after = pd.read_csv(after_path)
                future_rain_list = df_after["rainfall_mm"].tolist()[:15]
            else:
                return f"Skipped {filename} (missing perfect weather data)"

        if models is None:
            models = load_models(model_dir=os.path.join("shield", "saved_models"))
        preds = predict_flood(
            csv_path=csv_path, models=models, progress_cb=None,
            future_rain_list=future_rain_list, lat=lat, lon=lon,
        )
        df_p = predictions_to_dataframe(preds)
        
        
        # In perfect-weather, predict_flood returns the actual passed future_rain_list values
        # In rolling eval and other runs, predictions_to_dataframe automatically gets rain_preds
        # So df_p already contains 'predicted_rainfall_mm' handled by predict_flood modifications.
        
        df_p.to_csv(out_path, index=False)
        return f"Processed {filename}"
        
    except Exception as e:
        return f"Error {filename}: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perfect-weather", action="store_true", help="Extract true future rain from After Data")
    parser.add_argument("--rolling-eval", action="store_true", help="Evaluate 1-day to 15-day rolling advance-warning lead time")
    args = parser.parse_args()

    train_dir = "Train Data"
    if args.rolling_eval:
        out_dir = "Predict Data Rolling"
    elif args.perfect_weather:
        out_dir = "Predict Data Perfect"
    else:
        out_dir = "Predict Data"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    csv_files = glob.glob(os.path.join(train_dir, "*.csv"))
    
    if not csv_files:
        logging.error(f"No CSVs found in {train_dir}")
        return

    print(f"Found {len(csv_files)} files in {train_dir}")

    if args.rolling_eval:
        # ── GPU single-process path ───────────────────────────────────────────
        # Load TF/XGB models ONCE here in the main process so all LSTM
        # inferences share one GPU context instead of re-initialising CUDA
        # for every worker.  ProcessPoolExecutor is intentionally avoided.
        logging.info("Rolling eval: loading models once in main process%s…",
                     " (GPU)" if _GPU_AVAILABLE else " (CPU)")
        shared_models = load_models(model_dir=os.path.join("shield", "saved_models"))

        for i, p in enumerate(csv_files, 1):
            res = process_file(p, args, out_dir, models=shared_models)
            logging.info("[%d/%d] %s", i, len(csv_files), res)
    else:
        # ── CPU-parallel path for standard / perfect-weather modes ────────────
        max_workers = min(8, os.cpu_count() or 4)
        logging.info("Starting ProcessPoolExecutor with %d workers…", max_workers)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_file, p, args, out_dir)
                       for p in csv_files]

            completed = 0
            for future in as_completed(futures):
                completed += 1
                res = future.result()
                logging.info("[%d/%d] %s", completed, len(csv_files), res)

    print("Prediction batch complete.")

if __name__ == "__main__":
    main()
