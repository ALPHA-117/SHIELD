import pandas as pd
import numpy as np
from shield.predict import predict_flood, load_models
from shield.config import FEATURES

def debug_forecast():
    csv_path = "Rain Data/barmer_monsoon_2023.csv"
    models = load_models()
    
    print("Running debug forecast...")
    preds = predict_flood(csv_path=csv_path, models=models)
    
    for i, p in enumerate(preds[:5]):
        print(f"Day {i+1}: Date={p[0].date()}, Prob={p[1]:.6f}")

if __name__ == "__main__":
    debug_forecast()
