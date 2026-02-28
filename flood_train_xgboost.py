import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from tkinter import Tk, filedialog
from collections import Counter

def select_csv_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Flood Dataset CSV",
        filetypes=[("CSV files", "*.csv")]
    )
    return file_path

def main():
    # ------------------ Load Dataset ------------------ #
    csv_file = select_csv_file()
    if not csv_file:
        print("No file selected. Exiting.")
        return

    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['soil_texture'] = df['soil_texture'].astype(int)

    # Rolling rainfall sums
    for days in [1, 3, 7, 15]:
        df[f'rainfall_{days}d_total'] = df['rainfall_mm'].rolling(window=days, min_periods=1).sum()

    df['rainfall_1h_max'] = df['rainfall_mm'] / 24.0

    # Soil texture infiltration map
    soil_infiltration_map = {
        1: 30, 2: 20, 3: 10, 4: 5, 5: 3, 6: 1.5
    }
    df['infiltration_rate'] = df['soil_texture'].map(soil_infiltration_map)

    # Define flood labels (relaxed condition)
    df['flood'] = (
        (df['rainfall_3d_total'] > 100) &
        (df['elevation'] < 60) &
        (df['infiltration_rate'] < 3)
    ).astype(int)

    print("\n⚠  Flood label counts in full dataset:\n", df['flood'].value_counts())

    df = df.dropna()

    features = [
        'rainfall_1h_max', 'rainfall_1d_total', 'rainfall_3d_total',
        'rainfall_7d_total', 'rainfall_15d_total',
        'elevation', 'infiltration_rate'
    ]
    X = df[features]
    y = df['flood']

    # ------------------ Load or Fit Scaler ------------------ #
    if os.path.exists('scaler.pkl'):
        scaler = joblib.load('scaler.pkl')
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, 'scaler.pkl')

    # ------------------ Train/Test Split ------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("✅ Training label distribution:", Counter(y_train))
    print("✅ Unique labels in y_train:", np.unique(y_train))

    # ------------------ Load or Train XGBoost ------------------ #
    model_path = 'flood_predictor_xgb.json'
    if os.path.exists(model_path):
        print("🔁 Loading existing model for fine-tuning...")
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        model.load_model(model_path)
        model.fit(X_train, y_train, xgb_model=model.get_booster())
    else:
        print("🚀 No existing model found. Training new one...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            eval_metric='logloss',
            base_score=0.5  # required for logistic loss
        )
        model.fit(X_train, y_train)

    # ------------------ Evaluation ------------------ #
    y_pred = model.predict(X_test)
    print("\n✅ Metrics after training on:", os.path.basename(csv_file))

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=[0,1]))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, labels=[0,1]))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # ------------------ Save Updated Model ------------------ #
    model.save_model(model_path)
    print(f"\n✅ Model updated and saved to: {model_path}")

if __name__ == "__main__":
    main()
