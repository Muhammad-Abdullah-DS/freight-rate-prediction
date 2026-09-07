#!/usr/bin/env python
"""
Freight Rate Prediction Challenge — Production Pipeline
Spotter Machine Learning Engineer Assessment

Executes end-to-end data preprocessing, feature engineering, 5-fold ensemble modeling,
generates validation_predictions.csv and december_predictions.csv, and runs score.py.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import RANDOM_SEED
from src.data_loader import load_raw_data, build_city_coordinate_registry
from src.feature_engineering import engineer_features, apply_target_encodings
from src.models import FreightRateEnsemble
from src.inference import generate_and_save_predictions

def main():
    print("=" * 70)
    print("      SPOTTER FREIGHT RATE PREDICTION — PRODUCTION PIPELINE")
    print("=" * 70)
    
    # 1. Load Data
    print("\n>>> STEP 1: LOADING RAW DATA")
    train_raw, val_raw, dec_raw, tmpl_raw = load_raw_data()
    
    # 2. Extract City Coordinates & Macro Signals
    print("\n>>> STEP 2: BUILDING GEOSPATIAL & MACRO REGISTRIES")
    city_coords = build_city_coordinate_registry(train_raw, val_raw)
    
    val_temp = val_raw.copy()
    val_temp['date'] = pd.to_datetime(val_temp['date'])
    val_dec = val_temp[val_temp['date'].dt.month == 12]
    
    daily_mi_dec = val_dec.groupby(val_dec['date'].dt.date)['market_index'].mean().to_dict()
    daily_qs_dec = val_dec.groupby(val_dec['date'].dt.date)['quote_signal'].mean().to_dict()
    daily_signals = (daily_mi_dec, daily_qs_dec)
    
    median_weight = float(train_raw['weight'].median())
    median_mi = float(train_raw['market_index'].median())
    
    # 3. Feature Engineering
    print("\n>>> STEP 3: EXECUTING FEATURE ENGINEERING (38 FEATURES)")
    train_feat = engineer_features(train_raw, city_coords, None, median_weight, median_mi)
    val_feat = engineer_features(val_raw, city_coords, daily_signals, median_weight, median_mi)
    dec_feat = engineer_features(dec_raw, city_coords, daily_signals, median_weight, median_mi)
    
    # 4. Target Encodings
    print("\n>>> STEP 4: APPLYING REGULARIZED TARGET ENCODINGS")
    train_processed, val_processed, dec_processed = apply_target_encodings(train_feat, val_feat, dec_feat)
    
    # 5. Model Training
    print("\n>>> STEP 5: TRAINING MULTI-OBJECTIVE ENSEMBLE (5-FOLD CV)")
    ensemble = FreightRateEnsemble()
    oof_preds = ensemble.fit_cv(train_processed)
    
    # 6. Inference
    print("\n>>> STEP 6: GENERATING SUBMISSION PREDICTIONS")
    val_sub, dec_sub = generate_and_save_predictions(
        val_raw, dec_raw, val_processed, dec_processed, ensemble
    )
    
    # 7. Verification via score.py
    print("\n>>> STEP 7: RUNNING SCORE.PY CONFORMANCE VALIDATION")
    score_script = "score.py" if os.path.exists("score.py") else "../score.py"
    
    cmd = [
        sys.executable, score_script,
        "--predictions", "validation_predictions.csv",
        "--december-predictions", "december_predictions.csv"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("[Warning] Scorer Stderr:", result.stderr)
        
    print("=" * 70)
    print("Pipeline Execution Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
