import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, messagebox
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import Counter

class FloodPredictor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()

    def select_file(self):
        filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        file_path = filedialog.askopenfilename(
            title="Select Flood Data CSV",
            filetypes=filetypes
        )
        if not file_path:
            messagebox.showerror("Error", "No file selected. Exiting.")
            exit()
        elif not file_path.endswith('.csv'):
            messagebox.showerror("Error", "Please select a CSV file")
            exit()
        return file_path

    def create_features(self, df):
        numeric_cols = ['rainfall_mm', 'elevation', 'soil_texture']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        for days in [1, 3, 7, 15]:
            df[f'rainfall_{days}d_total'] = df['rainfall_mm'].rolling(
                window=days, min_periods=1
            ).sum().fillna(0)

        df['rainfall_1h_max'] = df['rainfall_mm'] / 24.0
        soil_map = {1: 30, 2: 20, 3: 10, 4: 5, 5: 3, 6: 1.5}
        df['infiltration_rate'] = df['soil_texture'].map(soil_map).fillna(1.5)
        return df.dropna()

    def generate_labels(self, df):
        df['date'] = pd.to_datetime(df['date'])
        df['flood'] = 0
        df.loc[df['rainfall_mm'] > 50, 'flood'] = 1
        return df

    def prepare_sequences(self, df, features, sequence_length=50):
        X, y = [], []
        for i in range(len(df) - sequence_length):
            X.append(df[features].iloc[i:i+sequence_length].values)
            y.append(df['flood'].iloc[i+sequence_length])
        return np.array(X), np.array(y)

    def build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
        return model

    def train_xgboost_model(self, X_train, y_train):
        class_counts = Counter(y_train)
        print("Class distribution:", class_counts)

        if min(class_counts.values()) < 2:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        else:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=5,
            early_stopping_rounds=10,
            random_state=42
        )
        model.fit(X_train_split, y_train_split, eval_set=[(X_val, y_val)], verbose=False)
        return model

    def predict_flood_period(self, df):
        sequence_length = 50
        features = [
            'rainfall_mm', 'elevation', 'soil_texture',
            'rainfall_1d_total', 'rainfall_3d_total', 'rainfall_7d_total',
            'rainfall_15d_total', 'rainfall_1h_max', 'infiltration_rate'
        ]

        df = df.sort_values('date')
        df = df.reset_index(drop=True)

        X, y = self.prepare_sequences(df, features, sequence_length)

        # Normalize features
        scaler = StandardScaler()
        X_scaled = X.copy()
        n_samples, seq_len, n_features = X.shape
        X_scaled = scaler.fit_transform(X.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)

        # Train LSTM model
        lstm_model = self.build_lstm_model((sequence_length, len(features)))
        lstm_model.fit(X_scaled, y, epochs=10, batch_size=32, verbose=0)

        # Get LSTM predictions
        lstm_preds = lstm_model.predict(X_scaled).flatten()

        # Combine LSTM predictions with rainfall-based features
        X_combined = np.column_stack([lstm_preds, df.iloc[sequence_length:][[
            'rainfall_1d_total', 'rainfall_3d_total', 'rainfall_7d_total', 'infiltration_rate'
        ]].values])

        y_combined = y

        # Train XGBoost model on top
        xgb_model = self.train_xgboost_model(X_combined, y_combined)

        final_probs = xgb_model.predict_proba(X_combined)[:, 1]
        dates = df.iloc[sequence_length:]['date']

        top_indices = np.argsort(final_probs)[-5:][::-1]
        top_dates = [(dates.iloc[i], final_probs[i]) for i in top_indices]

        most_likely_date = top_dates[0][0] if top_dates else None
        confidence = top_dates[0][1] if top_dates else 0

        return most_likely_date, confidence, top_dates

    def run(self):
        try:
            df = pd.read_csv(self.select_file())
            df = self.create_features(df)
            df = self.generate_labels(df)

            if len(df) < 50:
                messagebox.showerror("Error", "Insufficient data: need at least 50 days")
                exit()

            pred_date, confidence, top_dates = self.predict_flood_period(df)

            if pred_date is None:
                messagebox.showerror("Error", "No predictions available for critical period")
                exit()

            result_text = [
                f"Data available up to: {df['date'].max().strftime('%B %d, %Y')}",
                "\nMost Likely Flood Date:",
                f"⏱️ {pred_date.strftime('%B %d')} (confidence: {confidence:.0%})",
                "\nOther Risky Periods:"
            ]

            for date, prob in top_dates[1:]:
                result_text.append(f"• {date.strftime('%B %d')} ({prob:.0%} confidence)")

            result_window = tk.Toplevel()
            result_window.title("Barpeta Flood Prediction")

            tk.Label(
                result_window,
                text="\n".join(result_text),
                font=('Arial', 12),
                justify='left',
                padx=20,
                pady=20
            ).pack()

            tk.Button(
                result_window,
                text="Close",
                command=result_window.destroy
            ).pack(pady=10)

            self.root.mainloop()

        except Exception as e:
            messagebox.showerror("Error", f"System error:\n{str(e)}")
            exit()


if __name__ == "__main__":
    for f in ['flood_predictor.json', 'flood_scaler.pkl']:
        if os.path.exists(f):
            os.remove(f)

    FloodPredictor().run()
