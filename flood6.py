import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
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

class RealTimeFloodPredictor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
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
            'saturation_index', 'api',
            'soil_moisture', 'moisture_7d_avg', 'river_flood_risk'
        ]
        self.rainfall_threshold = 0.001
        self.model_dir = "saved_models"
        self.region_risk_factor = 1.0  # Can be adjusted based on region

    def select_file(self):
        """Select a single CSV file"""
        filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        file_path = filedialog.askopenfilename(title="Select Flood Data CSV", filetypes=filetypes)
        if not file_path:
            messagebox.showerror("Error", "No file selected. Exiting.")
            sys.exit()
        return file_path

    def load_and_prepare_data(self, file_path):
        """Load and validate CSV file"""
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
                logging.warning(f"Constant features detected: {constant_cols}. Predictions may rely heavily on rainfall.")
                
            logging.info(f"Dataset size: {len(df)} rows")
            logging.info(f"Rainfall range: min={df['rainfall_mm'].min():.2f}, max={df['rainfall_mm'].max():.2f}")
            logging.info(f"Month distribution: {df['date'].dt.month.value_counts().to_dict()}")
            
            return df.sort_values('date').reset_index(drop=True)
        except Exception as e:
            logging.error(f"Data loading error: {str(e)}")
            messagebox.showerror("Error", f"Data validation failed: {str(e)}")
            sys.exit()

    def calculate_api(self, rainfall_series, current_idx, window=7):
        """Calculate Antecedent Precipitation Index"""
        if current_idx < window:
            return 50.0
        weights = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
        rainfalls = rainfall_series[current_idx-window:current_idx]
        return np.sum(weights[:len(rainfalls)] * rainfalls)

    def predict_next_rainfall(self, last_rainfalls):
        """Predict next day's rainfall using recent trends + noise"""
        if len(last_rainfalls) == 0:
            return 0.0
        
        # Weighted average (recent days weighted higher)
        weights = np.array([0.5, 0.3, 0.2][:len(last_rainfalls)])
        weights /= weights.sum()  # Normalize
        avg_rain = np.sum(weights * last_rainfalls[-len(weights):])
        
        # Add log-normal noise (skewed right for occasional heavy rain)
        noise = np.random.lognormal(mean=0, sigma=0.3)
        predicted_rain = avg_rain * noise
        
        # Cap extreme values while allowing rare heavy rain
        return min(predicted_rain, 150)  # 150mm absolute max

    def update_soil_moisture(self, prev_moisture, rainfall, soil_type, temp_C=25):
        """Improved soil moisture model with evaporation"""
        retention = self.soil_params[soil_type]['retention']
        evap_rate = 0.05 + (0.01 * temp_C)  # Higher temp = more evaporation
        new_moisture = (retention * prev_moisture * (1 - evap_rate)) + (rainfall * (1 - retention))
        return max(0, min(100, new_moisture))

    def calculate_flood_threshold(self, row):
        """Dynamic flood threshold based on current conditions"""
        soil_info = self.soil_params.get(row['soil_texture'], self.soil_params[7])
        base_thresh = soil_info['base_thresh']
        
        # Elevation factor (higher = less risk)
        elev_factor = 1.0 - (0.0005 * row['elevation'])
        
        # Recent rain factor (3-day avg)
        rain_3d_avg = row['rainfall_3d_total'] / 3
        rain_factor = 1.0 + (rain_3d_avg / 50)
        
        # Soil saturation factor
        saturation_factor = 1.0 + (row['soil_moisture'] / 100)
        
        # River proximity factor
        river_factor = 1.0 + (0.5 * (1 - row['distance_to_water'] / 1000))
        
        # Combine all factors
        dynamic_thresh = base_thresh * elev_factor * rain_factor * saturation_factor * river_factor
        return max(10, dynamic_thresh)

    def create_features(self, df):
        """Create all derived features"""
        df['month'] = df['date'].dt.month
        
        # Antecedent Precipitation Index
        df['api'] = [self.calculate_api(df['rainfall_mm'].values, i) for i in range(len(df))]
        
        # Soil moisture with evaporation
        df['soil_moisture'] = 0.0
        for i in range(1, len(df)):
            df.at[i, 'soil_moisture'] = self.update_soil_moisture(
                df.at[i-1, 'soil_moisture'],
                df.at[i, 'rainfall_mm'],
                df.at[i, 'soil_texture']
            )
        
        # Rolling averages
        df['moisture_7d_avg'] = df['soil_moisture'].rolling(window=7, min_periods=1).mean()
        for days in [1, 3, 7]:
            df[f'rainfall_{days}d_total'] = df['rainfall_mm'].rolling(window=days, min_periods=1).sum()
        
        # Dynamic infiltration and thresholds
        df['infiltration_rate'] = df['soil_texture'].map(lambda x: self.soil_params[x]['infil_rate'])
        df['flood_threshold'] = df.apply(self.calculate_flood_threshold, axis=1)
        df['saturation_index'] = (df['api'] + df['soil_moisture']) / df['flood_threshold']
        
        # River flood risk
        df['river_flood_risk'] = (
            0.3 * (df['water_occurrence'] / 100) + 
            0.2 * (df['water_seasonality'] / 10) + 
            0.5 * np.where(
                df['distance_to_water'] == 0,
                0.7,
                0.3 * (1 - np.minimum(df['distance_to_water'], 1000) / 1000)
            )
        )
        return df.dropna()

    def load_models(self):
        """Load pre-trained models"""
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
            
            if lstm_model.input_shape[2] != len(self.features):
                logging.info(f"Feature count mismatch: model expects {lstm_model.input_shape[2]} features, script expects {len(self.features)}. Training required.")
                return None, None, None
                
            logging.info("Loaded models: lstm_flood.keras, xgb_flood.pkl, scaler_flood.pkl")
            return lstm_model, xgb_model, feature_scaler
            
        except FileNotFoundError as e:
            logging.info(f"Model files not found: {str(e)}. Training required.")
            return None, None, None
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            return None, None, None

    def recursive_predict(self, df, lstm_model, xgb_model, feature_scaler, future_days=15):
        """Generate future predictions"""
        df_rec = df.copy()
        predictions = []
        
        for day in range(1, future_days + 1):
            # 1. Predict next rainfall
            last_rain = df_rec['rainfall_mm'].tail(3).values
            next_rain = self.predict_next_rainfall(last_rain)
            
            # 2. Create new row
            new_row = {
                'date': df_rec['date'].iloc[-1] + timedelta(days=1),
                'rainfall_mm': next_rain,
                **{col: df_rec[col].iloc[-1] for col in [
                    'elevation', 'soil_texture', 'water_occurrence',
                    'water_seasonality', 'distance_to_water'
                ]}
            }
            df_rec = pd.concat([df_rec, pd.DataFrame([new_row])], ignore_index=True)
            df_rec = self.create_features(df_rec)
            
            # 3. Model predictions
            seq = feature_scaler.transform(df_rec[self.features].iloc[-7:])
            lstm_prob = lstm_model.predict(seq.reshape(1, 7, len(self.features)), verbose=0)[0][0]
            
            x_features = feature_scaler.transform([df_rec[self.features].iloc[-1]])
            xgb_prob = xgb_model.predict_proba(x_features)[0][1]
            
            # 4. Combined probability with region factor
            combined_prob = (0.6 * lstm_prob + 0.4 * xgb_prob) * self.region_risk_factor
            combined_prob = min(0.99, max(0.01, combined_prob))  # Keep within bounds
            
            predictions.append((new_row['date'], combined_prob))
            
            # Debug logging
            logging.info(
                f"Day {day}: Rain={next_rain:.1f}mm, "
                f"LSTM={lstm_prob:.2f}, XGB={xgb_prob:.2f}, "
                f"Final={combined_prob:.2f}, "
                f"Soil={df_rec['soil_moisture'].iloc[-1]:.1f}%"
            )
        
        return predictions

    def get_risk_level(self, prob):
        """Dynamic risk classification"""
        thresholds = {
            "🟢 LOW": 0.1 * self.region_risk_factor,
            "🟡 MODERATE": 0.3 * self.region_risk_factor,
            "🟠 HIGH": 0.5 * self.region_risk_factor,
            "🔴 EXTREME": 0.7 * self.region_risk_factor
        }
        for level, thresh in sorted(thresholds.items(), key=lambda x: x[1], reverse=True):
            if prob >= thresh:
                return level
        return "🟢 LOW"

    def plot_predictions(self, predictions):
        """Visualize the prediction results"""
        if not predictions:
            return
            
        dates, probs = zip(*predictions)
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, probs, 'o-', color='#1f77b4', linewidth=2, label='Flood Probability')
        
        # Dynamic threshold lines
        plt.axhline(y=0.1*self.region_risk_factor, color='green', linestyle='--', alpha=0.5, label='Low Risk Threshold')
        plt.axhline(y=0.3*self.region_risk_factor, color='yellow', linestyle='--', alpha=0.5, label='Moderate Risk Threshold')
        plt.axhline(y=0.5*self.region_risk_factor, color='orange', linestyle='--', alpha=0.5, label='High Risk Threshold')
        plt.axhline(y=0.7*self.region_risk_factor, color='red', linestyle='--', alpha=0.5, label='Extreme Risk Threshold')
        
        probs_smooth = gaussian_filter1d(probs, sigma=1)
        plt.fill_between(dates, probs_smooth*0.9, probs_smooth*1.1, color='blue', alpha=0.1, label='Confidence Band')
        
        plt.title("15-Day Flood Probability Forecast", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Probability", fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def show_results(self, predictions):
        """Display results in a GUI window"""
        if not predictions:
            messagebox.showinfo("Result", "No predictions generated")
            return
            
        result_text = ["Flood Probability Forecast:\n"]
        for date, prob in predictions:
            risk_level = self.get_risk_level(prob)
            result_text.append(f"{date.strftime('%a %d-%b')}: {prob:.0%} {risk_level}")
        
        # Highlight critical periods
        critical_days = [(date, prob) for date, prob in predictions if prob >= 0.3*self.region_risk_factor]
        if critical_days:
            result_text.append("\n⚠️ CRITICAL PERIODS:")
            result_text.extend([
                f"{date.strftime('%d-%b')}: {prob:.0%} {self.get_risk_level(prob)}" 
                for date, prob in critical_days
            ])
        
        result_window = tk.Toplevel()
        result_window.title("Flood Forecast Results")
        
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
                text_widget.insert('end', line + '\n', 'extreme')
            elif "🟠" in line:
                text_widget.insert('end', line + '\n', 'high')
            elif "🟡" in line:
                text_widget.insert('end', line + '\n', 'moderate')
            elif "⚠️" in line:
                text_widget.insert('end', line + '\n', 'warning')
            else:
                text_widget.insert('end', line + '\n')
        
        text_widget.tag_config('extreme', foreground='red')
        text_widget.tag_config('high', foreground='orange')
        text_widget.tag_config('moderate', foreground='gold')
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
            
            if len(df) < 7:
                messagebox.showerror("Error", "Minimum 7 days of data required for sequence length")
                return
                
            df = self.create_features(df)
            
            lstm_model, xgb_model, feature_scaler = self.load_models()
            
            if None in [lstm_model, xgb_model, feature_scaler]:
                messagebox.showerror("Error", "Failed to load models. Please train models first.")
                return
                
            predictions = self.recursive_predict(df, lstm_model, xgb_model, feature_scaler)
            if not predictions:
                messagebox.showerror("Error", "Prediction failed")
                return
                
            self.show_results(predictions)
            
        except Exception as e:
            messagebox.showerror("Error", f"System error: {str(e)}")
            logging.error(f"Unexpected error: {str(e)}")
            logging.exception("Unexpected error:")

if __name__ == "__main__":
    RealTimeFloodPredictor().run()
