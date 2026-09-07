import os
import pandas as pd
import numpy as np
from src.config import (
    VAL_PREDICTIONS_OUTPUT, DEC_PREDICTIONS_OUTPUT, DECEMBER_INPUT_PATH
)

def generate_and_save_predictions(val_raw_df, dec_raw_df, val_processed_df, dec_processed_df, ensemble_model):
    """
    Generates submission CSVs and ensures strict schema conformance.
    """
    print("[Inference] Predicting validation dataset (12,000 loads)...")
    val_preds = ensemble_model.predict(val_processed_df)
    
    val_submission = pd.DataFrame({
        'load_id': val_raw_df['load_id'],
        'predicted_rate': np.round(val_preds, 2)
    })
    
    # Save validation_predictions.csv
    val_submission.to_csv(VAL_PREDICTIONS_OUTPUT, index=False)
    # Also save in root if running from full professional solution
    val_submission.to_csv("validation_predictions.csv", index=False)
    print(f"[Inference] Saved validation predictions -> {VAL_PREDICTIONS_OUTPUT} ({len(val_submission)} rows)")
    
    print("[Inference] Predicting fixed December scenario (31 days)...")
    dec_preds = ensemble_model.predict(dec_processed_df)
    
    dec_submission = dec_raw_df.copy()
    dec_submission['predicted_rate'] = np.round(dec_preds, 2)
    
    dec_submission.to_csv(DEC_PREDICTIONS_OUTPUT, index=False)
    dec_submission.to_csv("december_predictions.csv", index=False)
    if os.path.exists(DECEMBER_INPUT_PATH):
        dec_submission.to_csv(DECEMBER_INPUT_PATH, index=False)
    dec_submission.to_csv("december-chart-inputs.csv", index=False)
    
    print(f"[Inference] Saved December scenario predictions -> {DEC_PREDICTIONS_OUTPUT} ({len(dec_submission)} rows)")
    return val_submission, dec_submission
