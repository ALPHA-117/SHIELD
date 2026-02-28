import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter
import joblib
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import logging
import warnings

# Configure logging and warnings
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
warnings.filterwarnings('ignore')

class EnhancedFloodPredictor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.model_dir = "saved_models"
        self.soil_params = {
            1: {'name': 'Clay', 'base_thresh': 80, 'infil_rate': 1.0, 'retention': 0.95},
            2: {'name': 'Silty clay', 'base_thresh': 78, 'infil_rate': 1.2, 'retention': 0.90},
            3: {'name': 'Sandy clay', 'base_thresh': 75, 'infil_rate': 3.0, 'retention': 0.85},
            4: {'name': 'Clay loam', 'base_thresh': 72, 'infil_rate': 5.0, 'retention': 0.80},
            5: {'name': 'Silty clay loam', 'base_thresh': 70, 'infil_rate': 7.0, 'retention': 0.75},
            6: {'name': 'Sandy clay loam', 'base_thresh': 68, 'infil_rate': 8.0, 'retention': 0.70},
            7: {'name': 'Loam', 'base_thresh': 65, 'infil_rate': 15.0, 'retention': 0.65},
            8: {'name': 'Silty loam', 'base_thresh': 60, 'infil_rate': 10.0, 'retention': 0.60},
            9: {'name': 'Sandy loam', 'base_thresh': 58, 'infil_rate': 25.0, 'retention': 0.55},
            10: {'name': 'Silt', 'base_thresh': 62, 'infil_rate': 8.0, 'retention': 0.65},
            11: {'name': 'Loamy sand', 'base_thresh': 55, 'infil_rate': 28.0, 'retention': 0.50},
            12: {'name': 'Sand', 'base_thresh': 50, 'infil_rate': 30.0, 'retention': 0.45},
        }
        self.features = [
            'rainfall_mm', 'elevation', 'soil_texture',
            'water_occurrence', 'water_seasonality', 'distance_to_water',
            'rainfall_1d_total', 'rainfall_3d_total', 'rainfall_7d_total',
            'flood_threshold', 'infiltration_rate', 'month',
            'is_monsoon', 'saturation_index', 'api',
            'soil_moisture', 'moisture_7d_avg', 'river_flood_risk'
        ]
        self.rainfall_threshold = 0.001

    def add_river_features(self, df):
        """Calculate river flood risk using water-related features"""
        df['river_flood_risk'] = (
            0.3 * (df['water_occurrence'] / 100) + 
            0.2 * (df['water_seasonality'] / 10) + 
            0.5 * np.where(
                df['distance_to_water'] == 0,
                0.7,
                0.3 * (1 - np.minimum(df['distance_to_water'], 1000) / 1000)
            )
        )
        df['river_flood_risk'] = np.where(
            df['rainfall_3d_total'] > 50,
            df['river_flood_risk'] * (1 + (df['rainfall_3d_total'] - 50) / 100),
            df['river_flood_risk']
        )
        return df

    def select_file(self):
        """Open file dialog to select CSV file"""
        filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        file_path = filedialog.askopenfilename(title="Select Flood Data CSV", filetypes=filetypes)
        if not file_path:
            messagebox.showerror("Error", "No file selected. Exiting.")
            sys.exit()
        return file_path

    def load_and_prepare_data(self, file_path):
        """Load and validate input CSV data"""
        try:
            df = pd.read_csv(file_path, parse_dates=['date'], dayfirst=True)
            required_cols = ['date', 'rainfall_mm', 'elevation', 'soil_texture',
                            'water_occurrence', 'water_seasonality', 'distance_to_water']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing}")
            if df['rainfall_mm'].isnull().sum() > 0:
                logging.warning(f"Found {df['rainfall_mm'].isnull().sum()} missing rainfall values - filling with 0")
                df['rainfall_mm'] = df['rainfall_mm'].fillna(0)
            tiny_mask = df['rainfall_mm'].abs() < self.rainfall_threshold
            if tiny_mask.sum() > 0:
                logging.info(f"Found {tiny_mask.sum()} rainfall values < {self.rainfall_threshold}mm - treating as 0")
                df.loc[tiny_mask, 'rainfall_mm'] = 0
            if df['elevation'].min() <= 0:
                raise ValueError("Invalid elevation values (must be positive)")
            if not all(df['soil_texture'].between(1, 12)):
                raise ValueError("Soil texture must be integer between 1-12")
            if df['water_occurrence'].min() < 0 or df['water_occurrence'].max() > 100:
                raise ValueError("Water occurrence must be between 0-100")
            if df['distance_to_water'].min() < 0:
                raise ValueError("Distance to water cannot be negative")
            constant_cols = [col for col in ['elevation', 'soil_texture', 'water_occurrence', 
                                            'water_seasonality', 'distance_to_water'] 
                            if df[col].nunique() == 1]
            if constant_cols:
                logging.warning(f"Constant features detected: {constant_cols}. Model may have reduced robustness.")
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                raise ValueError("Date column must be in datetime format")
            logging.info(f"Dataset size: {len(df)} rows")
            logging.info(f"Rainfall range: min={df['rainfall_mm'].min():.2f}, max={df['rainfall_mm'].max():.2f}")
            logging.info(f"Month distribution: {df['date'].dt.month.value_counts().to_dict()}")
            return df.sort_values('date').reset_index(drop=True)
        except Exception as e:
            logging.error(f"Data loading error: {e}")
            messagebox.showerror("Error", f"Data validation failed: {e}")
            sys.exit()

    def calculate_api(self, rainfall_series, current_idx, window=7):
        """Calculate Antecedent Precipitation Index"""
        if current_idx < window:
            return 50.0
        weights = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
        rainfalls = rainfall_series[current_idx-window:current_idx]
        return np.sum(weights[:len(rainfalls)] * rainfalls)

    def calculate_flood_threshold(self, row):
        """Calculate dynamic flood threshold"""
        soil_info = self.soil_params.get(row['soil_texture'], self.soil_params[7])
        base_thresh = soil_info['base_thresh']
        elev_factor = 1.3 - (0.0006 * row['elevation']) * (1 + row['moisture_7d_avg']/100)
        elev_factor = np.clip(elev_factor, 0.6, 1.4)
        api_factor = 1.6 - (row['api'] / 100) * (1 + row['soil_moisture']/50)
        api_factor = np.clip(api_factor, 0.4, 1.6)
        water_factor = 1.0 - (0.3 * (row['water_occurrence'] / 100) * 
                             (1 - (row['distance_to_water'] / 1000)))
        water_factor = np.clip(water_factor, 0.7, 1.3)
        monsoon_factor = 0.7 if row['month'] in [6,7,8,9] else 1.0
        return max(15, base_thresh * elev_factor * api_factor * monsoon_factor * water_factor)

    def create_features(self, df):
        """Create derived features for modeling"""
        try:
            df['month'] = df['date'].dt.month
            df['is_monsoon'] = df['month'].isin([6,7,8,9]).astype(int)
            df['api'] = [self.calculate_api(df['rainfall_mm'].values, i) for i in range(len(df))]
            df['soil_moisture'] = 0.0
            for i in range(1, len(df)):
                retention = self.soil_params.get(df.at[i, 'soil_texture'], self.soil_params[7])['retention']
                df.at[i, 'soil_moisture'] = (
                    retention * df.at[i-1, 'soil_moisture'] +
                    (1 - retention) * df.at[i, 'rainfall_mm']
                )
            df['moisture_7d_avg'] = df['soil_moisture'].rolling(window=7, min_periods=1).mean()
            for days in [1, 3, 7]:
                df[f'rainfall_{days}d_total'] = (
                    df['rainfall_mm'].rolling(window=days, min_periods=1).sum().fillna(0)
                )
            df['infiltration_rate'] = df['soil_texture'].map(
                lambda x: self.soil_params.get(x, self.soil_params[7])['infil_rate']
            )
            df['flood_threshold'] = df.apply(self.calculate_flood_threshold, axis=1)
            df['saturation_index'] = (df['api'] + df['soil_moisture']) / df['flood_threshold']
            df = self.add_river_features(df)
            return df.dropna()
        except Exception as e:
            logging.error(f"Feature creation failed: {e}")
            raise

    def generate_labels(self, df):
        """Generate flood labels based on thresholds"""
        if 'rainfall_mm' not in df.columns or 'flood_threshold' not in df.columns:
            raise ValueError("Missing required columns for label generation")
        df['flood'] = 0
        df.loc[
            (df['rainfall_mm'] > df['flood_threshold']) & 
            (df['saturation_index'] > 0.8),
            'flood'
        ] = 1
        df.loc[
            (df['river_flood_risk'] > 0.8) & 
            (df['rainfall_3d_total'] > 75),
            'flood'
        ] = 1
        last_date = df['date'].max()
        if last_date.month in [6,7,8,9]:
            warning_start = last_date - pd.Timedelta(days=10)
            warning_end = last_date - pd.Timedelta(days=3)
            df.loc[
                (df['date'] >= warning_start) & 
                (df['date'] <= warning_end) & 
                (df['rainfall_7d_total'] > 0.8 * df['flood_threshold']) & 
                (df['saturation_index'] > 0.9),
                'flood'
            ] = 1
            if last_date.month == 6 and last_date.day <= 10:
                df.loc[
                    (df['date'].dt.month == 6) & 
                    (df['date'].dt.day.isin(range(14, 21))) &
                    (df['rainfall_7d_total'] > 50),
                    'flood'
                ] = 1
        df.loc[
            (df['rainfall_mm'] < 0.5 * df['flood_threshold']) & 
            (df['river_flood_risk'] < 0.3) & 
            (df['saturation_index'] < 0.5),
            'flood'
        ] = 0
        label_counts = df['flood'].value_counts()
        logging.info(f"Label distribution after generate_labels: {label_counts.to_dict()}")
        return df

    def prepare_sequences(self, df, seq_length=7):
        """Prepare sequences for LSTM"""
        X, y = [], []
        for i in range(len(df) - seq_length):
            seq_data = df[self.features].iloc[i:i+seq_length].values
            X.append(seq_data)
            y.append(df['flood'].iloc[i+seq_length])
        X, y = np.array(X), np.array(y)
        label_counts = Counter(y)
        logging.info(f"Label distribution after prepare_sequences: {label_counts}")
        return X, y

    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_xgboost_model(self, X_train, y_train):
        """Train XGBoost model"""
        try:
            class_counts = Counter(y_train)
            logging.info(f"Class distribution in train_xgboost_model: {class_counts}")
            if len(class_counts) < 2:
                logging.error("Only one class found in y_train. Cannot train XGBoost model.")
                raise ValueError("XGBoost requires at least two classes for training.")
            scale_pos_weight = (class_counts[0] / class_counts[1]) * 4 if class_counts[1] > 0 else 12
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.9,
                colsample_bytree=0.9,
                objective='binary:logistic',
                eval_metric=['logloss', 'aucpr'],
                scale_pos_weight=scale_pos_weight,
                early_stopping_rounds=15,
                random_state=42
            )
            split_idx = int(0.8 * len(X_train))
            model.fit(
                X_train[:split_idx],
                y_train[:split_idx],
                eval_set=[(X_train[split_idx:], y_train[split_idx:])],
                verbose=False
            )
            return model
        except Exception as e:
            logging.error(f"XGBoost training failed: {e}")
            return None

    def load_models(self):
        """Load saved models"""
        try:
            if not os.path.exists(self.model_dir):
                raise FileNotFoundError("No saved models found in saved_models directory")
            lstm_file = f"{self.model_dir}/lstm_flood.keras"
            xgb_file = f"{self.model_dir}/xgb_flood.pkl"
            scaler_file = f"{self.model_dir}/scaler_flood.pkl"
            if not (os.path.exists(lstm_file) and os.path.exists(xgb_file) and os.path.exists(scaler_file)):
                raise FileNotFoundError("Missing one or more model files: lstm_flood.keras, xgb_flood.pkl, scaler_flood.pkl")
            lstm_model = load_model(lstm_file)
            xgb_model = joblib.load(xgb_file)
            feature_scaler = joblib.load(scaler_file)
            if lstm_model.input_shape[1] != len(self.features):
                logging.info(f"Feature count mismatch: model expects {lstm_model.input_shape[1]}, script expects {len(self.features)}. Will train new models.")
                return None, None, None
            logging.info("Loaded single model set: lstm_flood.keras, xgb_flood.pkl, scaler_flood.pkl")
            return lstm_model, xgb_model, feature_scaler
        except FileNotFoundError as e:
            logging.info(f"No existing models found: {e}. Will train new models.")
            return None, None, None
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            messagebox.showerror("Error", f"Failed to load models: {e}")
            return None, None, None

    def recursive_predict(self, df, lstm_model, xgb_model, feature_scaler, seq_length=7, future_days=15):
        """Generate recursive predictions"""
        try:
            df_rec = df.copy()
            predictions = []
            current_date = df_rec['date'].iloc[-1]
            for day in range(1, future_days + 1):
                next_date = current_date + timedelta(days=day)
                month, day_num = next_date.month, next_date.day
                seq = feature_scaler.transform(df_rec[self.features].iloc[-seq_length:])
                seq = seq.reshape(1, seq_length, len(self.features))
                if seq.shape[2] != lstm_model.input_shape[2]:
                    raise ValueError(f"Feature dimension mismatch: model expects {lstm_model.input_shape[2]}, got {seq.shape[2]}")
                lstm_prob = lstm_model.predict(seq, verbose=0)[0][0]
                is_monsoon = month in [6,7,8,9]
                if month == 6 and 14 <= day_num <= 20:
                    rainfall_pred = max(0, np.random.normal(80, 20))
                elif is_monsoon:
                    rainfall_pred = max(0, np.random.normal(60, 15))
                else:
                    rainfall_pred = max(0, np.random.normal(20, 10))
                new_row = {
                    'date': next_date,
                    'rainfall_mm': rainfall_pred,
                    'elevation': df_rec['elevation'].iloc[-1],
                    'soil_texture': df_rec['soil_texture'].iloc[-1],
                    'water_occurrence': df_rec['water_occurrence'].iloc[-1],
                    'water_seasonality': df_rec['water_seasonality'].iloc[-1],
                    'distance_to_water': df_rec['distance_to_water'].iloc[-1]
                }
                temp_df = pd.concat([df_rec, pd.DataFrame([new_row])], ignore_index=True)
                temp_df = self.create_features(temp_df)
                current_features = feature_scaler.transform([temp_df[self.features].iloc[-1]])
                X_combined = np.column_stack([
                    [lstm_prob],
                    current_features[:, [self.features.index(f) for f in [
                        'rainfall_1d_total', 'rainfall_3d_total', 'rainfall_7d_total',
                        'flood_threshold', 'saturation_index', 'is_monsoon',
                        'soil_moisture', 'moisture_7d_avg', 'river_flood_risk',
                        'water_occurrence', 'water_seasonality', 'distance_to_water'
                    ]]]
                ])
                flood_prob = xgb_model.predict_proba(X_combined)[0][1]
                if is_monsoon:
                    if month == 6 and 14 <= day_num <= 20:
                        day_boost = 0.5 * (day_num - 13)
                        flood_prob = min(1.0, flood_prob * (2.0 + day_boost))
                    elif month in [7,8]:
                        flood_prob = min(1.0, flood_prob * 1.5)
                    else:
                        flood_prob = min(1.0, flood_prob * 1.3)
                if temp_df['soil_moisture'].iloc[-1] > 50:
                    flood_prob = min(1.0, flood_prob * 1.2)
                water_risk = (
                    0.4 * (temp_df['water_occurrence'].iloc[-1] / 100) +
                    0.3 * (temp_df['water_seasonality'].iloc[-1] / 10) +
                    0.3 * (1 - temp_df['distance_to_water'].iloc[-1] / 1000)
                )
                flood_prob = min(1.0, flood_prob * (1 + water_risk))
                consecutive_flood_days = sum(1 for _,p in predictions[-3:] if p > 0.4)
                if consecutive_flood_days >= 2:
                    flood_prob = min(1.0, flood_prob * (1 + 0.1*consecutive_flood_days))
                predictions.append((next_date, flood_prob))
                df_rec = temp_df.copy()
            return predictions
        except Exception as e:
            logging.error(f"Recursive prediction failed: {e}")
            return []

    def plot_predictions(self, predictions):
        """Visualize predictions"""
        if not predictions:
            return
        dates, probs = zip(*predictions)
        plt.figure(figsize=(12, 6))
        plt.plot(dates, probs, 'o-', color='#1f77b4', linewidth=2, label='Flood Probability')
        plt.axhline(y=0.3, color='yellow', linestyle='--', alpha=0.5, label='High Risk Threshold')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Extreme Risk Threshold')
        plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Catastrophic Risk Threshold')
        probs_smooth = gaussian_filter1d(probs, sigma=1)
        plt.fill_between(dates, probs_smooth*0.9, probs_smooth*1.1, color='blue', alpha=0.1, label='Confidence Band')
        plt.title("15-Day Enhanced Flood Probability Forecast for Barpeta", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Probability", fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def show_results(self, predictions):
        """Display results in GUI"""
        if not predictions:
            messagebox.showinfo("Result", "No predictions generated")
            return
        result_text = ["Enhanced Flood Probability Forecast for Barpeta:\n"]
        for date, prob in predictions:
            if prob > 0.7: 
                risk_level = "🔴 CATASTROPHIC RISK"
            elif prob > 0.5: 
                risk_level = "🟠 EXTREME RISK"
            elif prob > 0.3:
                risk_level = "🟡 HIGH RISK"
            else:
                risk_level = ""
            result_text.append(f"{date.strftime('%a %d-%b')}: {prob:.0%} {risk_level}")
        critical_period = predictions[7:15]
        if any(prob > 0.4 for _, prob in critical_period):
            result_text.append("\n⚠️ CRITICAL PERIOD (7-15 days ahead):")
            result_text.extend([
                f"{date.strftime('%d-%b')}: {prob:.0%}" 
                for date, prob in critical_period 
                if prob > 0.4
            ])
        june_floods = [
            (date, prob) for date, prob in predictions
            if date.month == 6 and 14 <= date.day <= 20 and prob > 0.4
        ]
        if june_floods:
            result_text.append("\n⚠️ HISTORICAL FLOOD DAYS (June 14-20):")
            result_text.extend([f"{date.strftime('%d-%b')}: {prob:.0%}" for date, prob in june_floods])
        result_window = tk.Toplevel()
        result_window.title("Enhanced Flood Forecast Results for Barpeta")
        text_widget = tk.Text(
            result_window,
            font=('Arial', 11),
            wrap='word',
            padx=20,
            pady=20,
            width=80,
            height=30
        )
        for line in result_text:
            if "🔴" in line:
                text_widget.insert('end', line + '\n', 'catastrophic')
            elif "🟠" in line:
                text_widget.insert('end', line + '\n', 'extreme')
            elif "🟡" in line:
                text_widget.insert('end', line + '\n', 'high')
            elif "⚠️" in line:
                text_widget.insert('end', line + '\n', 'warning')
            else:
                text_widget.insert('end', line + '\n')
        text_widget.tag_config('catastrophic', foreground='red')
        text_widget.tag_config('extreme', foreground='orange')
        text_widget.tag_config('high', foreground='gold')
        text_widget.tag_config('warning', foreground='darkorange')
        text_widget.config(state='disabled')
        text_widget.pack()
        tk.Button(
            result_window,
            text="Close",
            command=result_window.destroy,
            padx=20
        ).pack(pady=10)
        self.plot_predictions(predictions)

    def run(self):
        """Main execution method"""
        try:
            file_path = self.select_file()
            df = self.load_and_prepare_data(file_path)
            if len(df) < 90:
                messagebox.showerror("Error", "Minimum 90 days of data required")
                return
            df = self.create_features(df)
            df = self.generate_labels(df)
            flood_count = df['flood'].sum()
            if flood_count < 3:
                proceed = messagebox.askyesno(
                    "Warning", 
                    f"Only {flood_count} flood events found in training data.\n"
                    "Model accuracy may be limited.\n"
                    "Continue with training?"
                )
                if not proceed:
                    return
            lstm_model, xgb_model, feature_scaler = self.load_models()
            train_new = True
            if None not in [lstm_model, xgb_model, feature_scaler]:
                train_new = messagebox.askyesno(
                    "Train New Model", 
                    "Loaded existing models. Train new models instead?"
                )
            if train_new:
                logging.info("Training new models...")
                seq_length = 7
                X, y = self.prepare_sequences(df, seq_length)
                if len(X) == 0:
                    messagebox.showerror("Error", "Insufficient sequence data")
                    return
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
                lstm_model = self.build_lstm_model((X.shape[1], X.shape[2]))
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
                lstm_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=16,
                    callbacks=[early_stopping],
                    verbose=0
                )
                lstm_features = lstm_model.predict(X_scaled, verbose=0)
                X_combined = np.column_stack([
                    lstm_features,
                    df.iloc[seq_length:][[
                        'rainfall_1d_total', 'rainfall_3d_total', 'rainfall_7d_total',
                        'flood_threshold', 'saturation_index', 'is_monsoon',
                        'soil_moisture', 'moisture_7d_avg', 'river_flood_risk',
                        'water_occurrence', 'water_seasonality', 'distance_to_water'
                    ]].values
                ])
                xgb_model = self.train_xgboost_model(X_combined, y)
                if xgb_model is None:
                    messagebox.showerror("Error", "Failed to train XGBoost model")
                    return
                os.makedirs(self.model_dir, exist_ok=True)
                lstm_model.save(f"{self.model_dir}/lstm_flood.keras")
                joblib.dump(xgb_model, f"{self.model_dir}/xgb_flood.pkl")
                joblib.dump(scaler, f"{self.model_dir}/scaler_flood.pkl")
                logging.info("Saved single model set: lstm_flood.keras, xgb_flood.pkl, scaler_flood.pkl")
                feature_scaler = scaler
            predictions = self.recursive_predict(df, lstm_model, xgb_model, feature_scaler)
            if not predictions:
                messagebox.showerror("Error", "Prediction failed")
                return
            self.show_results(predictions)
        except Exception as e:
            messagebox.showerror("Error", f"System error: {e}")
            logging.error(f"Unexpected error: {e}")
            logging.exception("Unexpected error occurred:")

if __name__ == "__main__":
    EnhancedFloodPredictor().run()
