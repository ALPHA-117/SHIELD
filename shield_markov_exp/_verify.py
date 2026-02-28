"""
Run from d:\\PROJECTS\\Rain Predict:
    python -m shield._verify
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_config():
    from shield.config import FEATURES, XGB_FEATURES, XGB_INPUT_SIZE, SOIL_PARAMS, get_risk_level
    assert len(FEATURES) == 18,         f"FEATURES: expected 18, got {len(FEATURES)}"
    assert len(XGB_FEATURES) == 12,     f"XGB_FEATURES: expected 12, got {len(XGB_FEATURES)}"
    assert XGB_INPUT_SIZE == 13,        f"XGB_INPUT_SIZE: expected 13, got {XGB_INPUT_SIZE}"
    assert len(SOIL_PARAMS) == 12,      f"SOIL_PARAMS: expected 12 soil types"
    lbl, clr = get_risk_level(0.6)
    assert "EXTREME" in lbl or "CATASTROPHIC" in lbl, f"Unexpected label for 0.6: {lbl}"
    print(f"  [PASS] config  FEATURES={len(FEATURES)}, XGB_INPUT_SIZE={XGB_INPUT_SIZE}")

def test_label_independence():
    from shield.config import LABEL_RAW_COLS
    derived = {"flood_threshold","saturation_index","river_flood_risk","api","soil_moisture"}
    overlap = derived.intersection(set(LABEL_RAW_COLS))
    assert not overlap, f"LABEL LEAKAGE DETECTED: derived features in LABEL_RAW_COLS: {overlap}"
    print(f"  [PASS] label independence  LABEL_RAW_COLS={LABEL_RAW_COLS}")

def test_xgb_subset_of_features():
    from shield.config import FEATURES, XGB_FEATURES
    missing = set(XGB_FEATURES) - set(FEATURES)
    assert not missing, f"XGB_FEATURES not in FEATURES: {missing}"
    print(f"  [PASS] XGB_FEATURES are a subset of FEATURES")

def test_features_import():
    from shield.features import create_features, validate_input_columns
    print(f"  [PASS] features.py imports OK")

def test_labels_import():
    from shield.labels import generate_labels, label_summary
    print(f"  [PASS] labels.py imports OK")

def test_rainfall_import():
    from shield.rainfall import SeasonalRainfallModel
    m = SeasonalRainfallModel()
    print(f"  [PASS] rainfall.py  SeasonalRainfallModel instantiated OK")

def test_rainfall_predict_no_fit():
    from shield.rainfall import SeasonalRainfallModel
    m = SeasonalRainfallModel()
    try:
        m.predict(month=7)
        print("  [FAIL] rainfall predict without fit should raise RuntimeError")
    except RuntimeError:
        print("  [PASS] predict before fit correctly raises RuntimeError")

def test_feature_pipeline():
    """End-to-end feature engineering on synthetic data."""
    import pandas as pd, numpy as np
    from shield.config import FEATURES
    from shield.features import create_features
    from datetime import date, timedelta

    n = 60
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(n)]
    df = pd.DataFrame({
        "date":              pd.to_datetime(dates),
        "rainfall_mm":       np.random.default_rng(0).gamma(2, 5, n),
        "elevation":         [45.0] * n,
        "soil_texture":      [1] * n,
        "water_occurrence":  [20.0] * n,
        "water_seasonality": [5.0] * n,
        "distance_to_water": [200.0] * n,
    })
    out = create_features(df)
    missing_cols = [f for f in FEATURES if f not in out.columns]
    assert not missing_cols, f"Missing after feature engineering: {missing_cols}"
    assert len(out) > 0, "Feature engineering dropped ALL rows"
    print(f"  [PASS] feature pipeline  input={n} rows, output={len(out)} rows, all 18 cols present")

def test_label_pipeline():
    """Label generation on synthetic data."""
    import pandas as pd, numpy as np
    from shield.features import create_features
    from shield.labels import generate_labels
    from datetime import date, timedelta

    n = 60
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(n)]
    df = pd.DataFrame({
        "date":              pd.to_datetime(dates),
        "rainfall_mm":       [10.0]*30 + [100.0]*30,  # second half very rainy
        "elevation":         [45.0] * n,
        "soil_texture":      [1] * n,                  # Clay = low infiltration
        "water_occurrence":  [20.0] * n,
        "water_seasonality": [5.0] * n,
        "distance_to_water": [200.0] * n,
    })
    df = create_features(df)
    df = generate_labels(df, region="Barpeta")
    assert "flood" in df.columns
    n_flood = df["flood"].sum()
    print(f"  [PASS] label pipeline  flood events found: {n_flood}")

def test_rainfall_fit_predict():
    """Seasonal model fit and deterministic prediction."""
    import pandas as pd, numpy as np
    from shield.rainfall import SeasonalRainfallModel
    from datetime import date, timedelta

    n = 400
    dates = pd.to_datetime([date(2020, 1, 1) + timedelta(days=i) for i in range(n)])
    rain  = pd.Series(np.random.default_rng(42).gamma(2, 8, n))
    m = SeasonalRainfallModel().fit(dates, rain)

    # Same seed → same result
    r1 = m.predict(month=7, seed=42)
    r2 = m.predict(month=7, seed=42)
    assert r1 == r2, f"Non-deterministic! r1={r1}, r2={r2}"

    seq1 = m.predict_sequence([6,6,7,7,8], seed=99)
    seq2 = m.predict_sequence([6,6,7,7,8], seed=99)
    assert seq1 == seq2, "Sequence non-deterministic!"
    assert all(v >= 0 for v in seq1), "Negative rainfall values!"
    print(f"  [PASS] rainfall model  deterministic, seq sample={[round(x,1) for x in seq1]}")

if __name__ == "__main__":
    tests = [
        test_config,
        test_label_independence,
        test_xgb_subset_of_features,
        test_features_import,
        test_labels_import,
        test_rainfall_import,
        test_rainfall_predict_no_fit,
        test_feature_pipeline,
        test_label_pipeline,
        test_rainfall_fit_predict,
    ]
    print(f"\nSHIELD Smoke Tests ({len(tests)} checks)\n{'='*52}")
    failed = 0
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            test()
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print(f"\n{'='*52}")
    if failed == 0:
        print(f"All {len(tests)} tests PASSED. ✅")
    else:
        print(f"{failed}/{len(tests)} tests FAILED. ❌")
    sys.exit(failed)
