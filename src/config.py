import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "report"
ASSETS_DIR = BASE_DIR / "assets"
SCORER_DIR = BASE_DIR / "scorer_results"

TRAIN_DATA_PATH = DATA_DIR / "train_test.csv"
VAL_DATA_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_INPUT_PATH = DATA_DIR / "december_chart_inputs.csv"

VAL_PREDICTIONS_OUTPUT = BASE_DIR / "validation_predictions.csv"
DEC_PREDICTIONS_OUTPUT = BASE_DIR / "december_predictions.csv"
DECEMBER_CHART_IMAGE = SCORER_DIR / "candidate_december.png"

# Modeling Hyperparameters
RANDOM_SEED = 42
N_FOLDS = 5

# LightGBM Params (L1 Loss on RPM)
LGBM_RPM_PARAMS = {
    'objective': 'regression_l1',
    'n_estimators': 1800,
    'learning_rate': 0.025,
    'num_leaves': 75,
    'min_child_samples': 20,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'verbose': -1
}

# LightGBM Params (L1 Loss on Direct Rate)
LGBM_RATE_PARAMS = {
    'objective': 'regression_l1',
    'n_estimators': 1800,
    'learning_rate': 0.025,
    'num_leaves': 75,
    'min_child_samples': 20,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'random_state': RANDOM_SEED + 10,
    'n_jobs': -1,
    'verbose': -1
}

# XGBoost Params (L1 Loss on Direct Rate)
XGB_RATE_PARAMS = {
    'objective': 'reg:absoluteerror',
    'n_estimators': 1400,
    'learning_rate': 0.025,
    'max_depth': 6,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'enable_categorical': True,
    'random_state': RANDOM_SEED + 20,
    'tree_method': 'hist',
    'n_jobs': -1
}

# Ensemble Blending Weights
ENSEMBLE_WEIGHTS = {
    'lgbm_rpm': 0.55,
    'lgbm_rate': 0.30,
    'xgb_rate': 0.15
}
