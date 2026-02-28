import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from collections import Counter
import sys
from datetime import datetime, timedelta

# GUI imports
try:
    from tkinter import Tk, filedialog, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

def select_csv_file():
    """Select CSV file with GUI or console input"""
    if GUI_AVAILABLE:
        try:
            root = Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Select Flood Dataset CSV",
                filetypes=[("CSV files", "*.csv")]
            )
            return file_path
        except Exception as e:
            print(f"GUI not available, fallback to console input. ({e})")
    return input("Enter path to the flood dataset CSV: ").strip('"')

def create_features(df):
    """Create all engineered features for the model"""
    # Rolling rainfall sums
    for days in [1, 3, 7, 15]:
        df[f'rainfall_{days}d_total'] = df['rainfall_mm'].rolling(window=days, min_periods=1).sum()
    
    # Hourly max rainfall estimate
    df['rainfall_1h_max'] = df['rainfall_mm'] / 24.0
    
    # Soil infiltration characteristics
    soil_infiltration_map = {1: 30, 2: 20, 3: 10, 4: 5, 5: 3, 6: 1.5}
    df['infiltration_rate'] = df['soil_texture'].map(soil_infiltration_map)
    
    return df

def generate_flood_labels(df, prediction_window=7, threshold=100):
    """
    Generate flood labels based on future conditions
    Args:
        df: DataFrame with rainfall data
        prediction_window: Days ahead to look for flood conditions
        threshold: Rainfall threshold (mm) to consider as flood
    """
    df['flood'] = 0  # Initialize all as non-flood
    
    # Look ahead to identify precursor conditions
    for i in range(len(df) - prediction_window):
        current = df.iloc[i]
        future_window = df.iloc[i+1:i+1+prediction_window]
        
        # Flood conditions (relaxed from original criteria)
        if (future_window['rainfall_mm'].sum() > threshold and 
            current['elevation'] < 60 and 
            current['infiltration_rate'] < 3):
            df.at[i, 'flood'] = 1
    
    return df

def train_evaluate_model(X_train, y_train, X_test=None, y_test=None, model_path='flood_predictor_xgb.json'):
    """Train or load model and evaluate performance"""
    # Load or create new model
    if os.path.exists(model_path):
        print("🔁 Loading existing model for fine-tuning...")
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        model.load_model(model_path)
        model.fit(X_train, y_train, xgb_model=model.get_booster())
    else:
        print("🚀 No existing model found. Training new one...")
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=5  # Adjust for class imbalance
        )
        model.fit(X_train, y_train)
    
    # Save the trained model
    model.save_model(model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Evaluate on training data
    y_pred_train = model.predict(X_train)
    print("\nTraining Set Evaluation:")
    print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
    print("Classification Report:\n", classification_report(y_train, y_pred_train))
    print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.2f}")
    
    # Evaluate on test data if available
    if X_test is not None and y_test is not None:
        y_pred_test = model.predict(X_test)
        print("\nTest Set Evaluation:")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
        print("Classification Report:\n", classification_report(y_test, y_pred_test))
        print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.2f}")
    
    return model

def predict_future_risk(model, X_recent, days=7, threshold=3):
    """Predict flood risk for future period"""
    pred = model.predict(X_recent)
    flood_days = np.sum(pred)
    
    if flood_days >= threshold:
        warning = f"⚠️ HIGH flood risk predicted ({flood_days} flood days in next {days} days)"
    elif flood_days > 0:
        warning = f"⚠️ MODERATE flood risk predicted ({flood_days} flood days in next {days} days)"
    else:
        warning = f"✅ No flood risk predicted in next {days} days"
    
    return warning, flood_days

def main():
    """Main execution function"""
    # Load data
    csv_file = select_csv_file()
    if not csv_file or not os.path.exists(csv_file):
        print("No valid file selected. Exiting.")
        return
    
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['soil_texture'] = df['soil_texture'].astype(int)
    
    # Feature engineering
    df = create_features(df)
    
    # Generate flood labels (looking ahead 7 days)
    df = generate_flood_labels(df, prediction_window=7, threshold=100)
    
    # Check label distribution
    print("\nFlood label distribution:")
    print(df['flood'].value_counts())
    
    # Split data (last 30 days for testing)
    split_date = df['date'].max() - timedelta(days=30)
    train_df = df[df['date'] <= split_date]
    test_df = df[df['date'] > split_date]
    
    # Feature columns
    features = [
        'rainfall_1h_max', 
        'rainfall_1d_total', 
        'rainfall_3d_total',
        'rainfall_7d_total', 
        'rainfall_15d_total',
        'elevation', 
        'infiltration_rate'
    ]
    
    # Prepare data
    X_train = train_df[features]
    y_train = train_df['flood']
    X_test = test_df[features]
    y_test = test_df['flood']
    
    # Scale features
    scaler_path = 'flood_scaler.pkl'
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(X_train)
        joblib.dump(scaler, scaler_path)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate model
    model = train_evaluate_model(
        X_train_scaled, y_train, 
        X_test_scaled, y_test,
        model_path='flood_predictor_xgb.json'
    )
    
    # Predict future risk (using most recent data)
    recent_data = df[features].iloc[-7:]  # Last 7 days
    recent_scaled = scaler.transform(recent_data)
    warning, flood_days = predict_future_risk(model, recent_scaled)
    
    print(f"\nFlood Risk Assessment:")
    print(f"📅 Predicted flood days in next week: {flood_days}")
    print(warning)
    
    # Show GUI alert if available
    if GUI_AVAILABLE:
        try:
            messagebox.showinfo("Flood Forecast", warning)
        except:
            print("(GUI alert failed - printed to console instead)")

if __name__ == "__main__":
    main()
