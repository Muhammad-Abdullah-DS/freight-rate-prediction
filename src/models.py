import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from src.config import (
    RANDOM_SEED, N_FOLDS, LGBM_RPM_PARAMS, LGBM_RATE_PARAMS, XGB_RATE_PARAMS, ENSEMBLE_WEIGHTS
)
from src.feature_engineering import FEATURE_COLUMNS

class FreightRateEnsemble:
    """
    Production-grade Multi-Objective Stacked Ensemble for spot freight rate prediction.
    Combines:
    1. LightGBM (L1 Loss on Rate-Per-Mile target)
    2. LightGBM (L1 Loss on Direct Posted Rate target)
    3. XGBoost (L1 Loss on Direct Posted Rate target)
    """
    def __init__(self):
        self.models_lgb_rpm = []
        self.models_lgb_rate = []
        self.models_xgb_rate = []
        self.cv_scores_mae = []
        self.feature_columns = FEATURE_COLUMNS
        self.weights = ENSEMBLE_WEIGHTS
        
    def fit_cv(self, train_df):
        """
        Trains 5-fold cross-validated models and calculates out-of-fold metrics.
        """
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        
        self.models_lgb_rpm = []
        self.models_lgb_rate = []
        self.models_xgb_rate = []
        self.cv_scores_mae = []
        
        oof_predictions = np.zeros(len(train_df))
        
        print(f"[ModelTraining] Starting {N_FOLDS}-Fold Cross-Validation...")
        for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
            tr_data = train_df.iloc[tr_idx]
            val_data = train_df.iloc[val_idx]
            
            # 1. Train LightGBM RPM
            p_rpm = LGBM_RPM_PARAMS.copy()
            p_rpm['random_state'] = RANDOM_SEED + fold
            m_rpm = lgb.LGBMRegressor(**p_rpm)
            m_rpm.fit(tr_data[self.feature_columns], tr_data['rpm'])
            self.models_lgb_rpm.append(m_rpm)
            
            # 2. Train LightGBM Direct Rate
            p_rate = LGBM_RATE_PARAMS.copy()
            p_rate['random_state'] = RANDOM_SEED + fold + 10
            m_rate = lgb.LGBMRegressor(**p_rate)
            m_rate.fit(tr_data[self.feature_columns], tr_data['posted_rate'])
            self.models_lgb_rate.append(m_rate)
            
            # 3. Train XGBoost Direct Rate
            p_xgb = XGB_RATE_PARAMS.copy()
            p_xgb['random_state'] = RANDOM_SEED + fold + 20
            m_xgb = xgb.XGBRegressor(**p_xgb)
            m_xgb.fit(tr_data[self.feature_columns], tr_data['posted_rate'])
            self.models_xgb_rate.append(m_xgb)
            
            # Compute fold OOF prediction
            fold_pred = (
                self.weights['lgbm_rpm'] * (m_rpm.predict(val_data[self.feature_columns]) * val_data['distance']) +
                self.weights['lgbm_rate'] * m_rate.predict(val_data[self.feature_columns]) +
                self.weights['xgb_rate'] * m_xgb.predict(val_data[self.feature_columns])
            )
            oof_predictions[val_idx] = fold_pred
            
            fold_mae = float(np.mean(np.abs(val_data['posted_rate'] - fold_pred)))
            self.cv_scores_mae.append(fold_mae)
            print(f"  -> Fold {fold + 1} OOF MAE: ${fold_mae:.2f}")
            
        overall_mae = float(np.mean(np.abs(train_df['posted_rate'] - oof_predictions)))
        overall_rmse = float(np.sqrt(np.mean((train_df['posted_rate'] - oof_predictions)**2)))
        overall_mape = float(np.mean(np.abs((train_df['posted_rate'] - oof_predictions) / train_df['posted_rate'])) * 100)
        
        print(f"[ModelTraining] Ensemble CV MAE: ${overall_mae:.2f} | RMSE: ${overall_rmse:.2f} | MAPE: {overall_mape:.2f}%")
        return oof_predictions
        
    def predict(self, df):
        """
        Generates ensembled predictions averaged across all cross-validation folds.
        """
        n_models = len(self.models_lgb_rpm)
        preds_lgb_rpm = np.zeros(len(df))
        preds_lgb_rate = np.zeros(len(df))
        preds_xgb_rate = np.zeros(len(df))
        
        for m_rpm in self.models_lgb_rpm:
            preds_lgb_rpm += (m_rpm.predict(df[self.feature_columns]) * df['distance']) / n_models
            
        for m_rate in self.models_lgb_rate:
            preds_lgb_rate += m_rate.predict(df[self.feature_columns]) / n_models
            
        for m_xgb in self.models_xgb_rate:
            preds_xgb_rate += m_xgb.predict(df[self.feature_columns]) / n_models
            
        final_preds = (
            self.weights['lgbm_rpm'] * preds_lgb_rpm +
            self.weights['lgbm_rate'] * preds_lgb_rate +
            self.weights['xgb_rate'] * preds_xgb_rate
        )
        return final_preds
