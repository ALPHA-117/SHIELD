import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import Counter
import joblib
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class FloodPredictor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()

    def select_file(self):
        """File selection dialog with validation"""
        filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        file_path = filedialog.askopenfilename(title="Select Flood Data CSV", filetypes=filetypes)
        if not file_path:
            messagebox.showerror("Error", "No file selected. Exiting.")
            exit()
        elif not file_path.endswith('.csv'):
            messagebox.showerror("Error", "Please select a CSV file")
            exit()
        return file_path

    def create_features(self, df):
        """Enhanced feature engineering with temporal features"""
        numeric_cols = ['rainfall_mm', 'elevation', 'soil_texture']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Rolling rainfall calculations
        for days in [1, 3, 7, 15]:
            df[f'rainfall_{days}d_total'] = df['rainfall_mm'].rolling(
                window=days, min_periods=1
            ).sum().fillna(0)

        # Additional features
        df['rainfall_1h_max'] = df['rainfall_mm'] / 24.0
        soil_map = {1: 30, 2: 20, 3: 10, 4: 5, 5: 3, 6: 1.5}
        df['infiltration_rate'] = df['soil_texture'].map(soil_map).fillna(1.5)
        
        # Temporal features for monsoon season
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)  # Monsoon in Assam
        return df.dropna()

    def generate_labels(self, df):
        """Generate flood labels based on warning period"""
        df['flood'] = 0
        # Define flood period based on last date
        last_date = df['date'].max()
        flood_end = last_date
        flood_start = flood_end - pd.Timedelta(days=6)  # 7-day flood period
        warning_start = flood_start - pd.Timedelta(days=15)
        warning_end = flood_start - pd.Timedelta(days=7)
        df.loc[(df['date'] >= warning_start) & (df['date'] <= warning_end), 'flood'] = 1
        return df, warning_start

    def prepare_sequences(self, df, features, sequence_length=50):
        """Prepare sequences for LSTM input"""
        X, y = [], []
        for i in range(len(df) - sequence_length):
            X.append(df[features].iloc[i:i+sequence_length].values)
            y.append(df['flood'].iloc[i+sequence_length])
        return np.array(X), np.array(y)

    def build_lstm_model(self, input_shape):
        """Build LSTM model with optimized architecture"""
        model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_xgboost_model(self, X_train, y_train):
        """Train XGBoost model with dynamic class weighting"""
        class_counts = Counter(y_train)
        scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 5
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train if class_counts[1] >= 2 else None
        )
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.02, subsample=0.8,
            colsample_bytree=0.8, objective='binary:logistic', eval_metric='logloss',
            scale_pos_weight=scale_pos_weight, early_stopping_rounds=10, random_state=42
        )
        model.fit(X_train_split, y_train_split, eval_set=[(X_val, y_val)], verbose=False)
        joblib.dump(model, "xgb_flood_model.pkl")
        return model

    def predict_future(self, df, lstm_model, xgb_model, features, sequence_length=50, future_days=15):
        """Predict future flood probabilities aligned with XGBoost-only approach"""
        df = df.sort_values("date").reset_index(drop=True)
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[features])

        # Use last data point for predictions
        last_data = df.iloc[-1][features]
        pred_dates = pd.date_range(
            start=df['date'].iloc[-1] + pd.Timedelta(days=1),
            periods=future_days,
            freq='D'
        )

        # Prepare prediction data
        X_pred = pd.DataFrame([last_data] * len(pred_dates), columns=features)
        X_scaled = scaler.transform(X_pred)

        # Get LSTM prediction for the last sequence
        last_seq = df_scaled[-sequence_length:].reshape(1, sequence_length, len(features))
        lstm_prob = lstm_model.predict(last_seq, verbose=0)[0][0]

        # Combine LSTM probability with selected features for XGBoost
        X_combined = np.column_stack([
            np.repeat(lstm_prob, len(X_scaled)),
            X_scaled[:, [features.index(f) for f in [
                'rainfall_1d_total', 'rainfall_3d_total', 'rainfall_7d_total', 'infiltration_rate'
            ]]]
        ])
        flood_probs = xgb_model.predict_proba(X_combined)[:, 1]

        # Filter for critical period (June 10–20, adjust for your data year)
        critical_start = pd.to_datetime(f"{df['date'].max().year}-06-10")
        critical_end = pd.to_datetime(f"{df['date'].max().year}-06-20")
        predictions = [(d, p) for d, p in zip(pred_dates, flood_probs) if critical_start <= d <= critical_end]
        return predictions

    def run(self):
        """Main execution with error handling and visualization"""
        try:
            # Load and prepare data
            df = pd.read_csv(self.select_file())
            df = self.create_features(df)
            df, warning_start = self.generate_labels(df)

            # Debug: Check label distribution
            print("Flood Label Distribution:", Counter(df['flood']))

            if len(df) < 60:
                messagebox.showerror("Error", "Insufficient data: need at least 60 days")
                return
            if df['flood'].sum() == 0:
                messagebox.showerror("Error", "No flood events in the data. Please include flood-prone periods.")
                return

            features = [
                'rainfall_mm', 'elevation', 'soil_texture',
                'rainfall_1d_total', 'rainfall_3d_total', 'rainfall_7d_total',
                'rainfall_15d_total', 'rainfall_1h_max', 'infiltration_rate',
                'month', 'is_monsoon'
            ]

            # Prepare sequences for LSTM
            X, y = self.prepare_sequences(df, features, sequence_length=50)
            if len(X) == 0:
                messagebox.showerror("Error", "Not enough sequence data.")
                return

            # Normalize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

            # Train LSTM
            lstm_model = self.build_lstm_model((X.shape[1], X.shape[2]))
            lstm_model.fit(X_scaled, y, epochs=15, batch_size=32, verbose=0)
            lstm_model.save("lstm_flood_model.h5")

            # Debug: Check LSTM predictions
            lstm_preds = lstm_model.predict(X_scaled, verbose=0).flatten()
            print("LSTM Predictions (first 5):", lstm_preds[:5])

            # Prepare data for XGBoost
            X_combined = np.column_stack([lstm_preds, df.iloc[50:][[
                'rainfall_1d_total', 'rainfall_3d_total', 'rainfall_7d_total', 'infiltration_rate'
            ]].values])
            xgb_model = self.train_xgboost_model(X_combined, y)

            if xgb_model is None:
                messagebox.showerror("Error", "Failed to train XGBoost model.")
                return

            # Predict future
            future_preds = self.predict_future(df, lstm_model, xgb_model, features)
            top_dates = sorted(future_preds, key=lambda x: x[1], reverse=True)[:5]

            # Debug: Check future predictions
            print("Future Predictions:", [(d.strftime('%Y-%m-%d'), p) for d, p in top_dates])

            if not top_dates:
                messagebox.showinfo("Prediction", "No flood predicted in the future.")
                return

            # Prepare results
            result_text = [
                f"Data available up to: {df['date'].max().strftime('%B %d, %Y')}",
                "\nMost Likely Future Flood Date:",
                f"⏱️ {top_dates[0][0].strftime('%B %d')} (confidence: {top_dates[0][1]:.0%})",
                "\nOther Risky Periods:"
            ] + [f"• {d.strftime('%B %d')} ({p:.0%} confidence)" for d, p in top_dates[1:]]

            # Display results
            result_window = tk.Toplevel()
            result_window.title("Barpeta Flood Forecast")
            tk.Label(result_window, text="\n".join(result_text), font=('Arial', 12), justify='left', padx=20, pady=20).pack()
            tk.Button(result_window, text="Close", command=result_window.destroy).pack(pady=10)

            # Visualize predictions
            dates, probs = zip(*future_preds)
            plt.figure(figsize=(10, 5))
            plt.plot(dates, probs, marker='o', color='#1f77b4')
            plt.title("Future Flood Probability")
            plt.xlabel("Date")
            plt.ylabel("Flood Probability")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            self.root.mainloop()

        except Exception as e:
            messagebox.showerror("Error", f"System error:\n{str(e)}")
            return

if __name__ == "__main__":
    # Clear previous model files
    for f in ['flood_predictor.json', 'flood_scaler.pkl', 'xgb_flood_model.pkl', 'lstm_flood_model.h5']:
        if os.path.exists(f):
            os.remove(f)

    FloodPredictor().run()
