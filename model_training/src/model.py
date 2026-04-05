import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
# Use XGBoost if available, fallback to GradientBoosting if not installed.
try:
    import xgboost as xgb
except ImportError:
    xgb = None

def get_ensemble_model():
    """
    Constructs the Stacking Regressor utilizing XGBoost, RF, SVR, and ElasticNetCV.
    """
    base_learners = [
        ('rf', RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42)),
        ('svr', SVR(kernel='rbf', C=50, gamma='scale'))
    ]
    
    if xgb:
        base_learners.append(('xgb', xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42)))

    # Meta-Learner handling non-linear combinations flexibly
    meta_learner = ElasticNetCV(cv=5, random_state=42)
    
    ensemble = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        passthrough=True # Allow meta-learner to see original features
    )
    
    return ensemble

def evaluate_metrics(y_true, y_pred):
    """
    Calculates MAE, R², MMRE, and Pred(25)
    """
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate Magnitude of Relative Error (MRE)
    # Adding epsilon to prevent division by zero
    mre = np.abs((y_true - y_pred) / (y_true + 1e-8))
    
    mmre = np.mean(mre) * 100 # Percentage
    pred_25 = np.sum(mre <= 0.25) / len(mre) * 100 # Percentage
    
    return {
        'MAE': mae,
        'R2': r2,
        'MMRE': mmre,
        'Pred_25': pred_25
    }

def calculate_confidence_interval(prediction, std_dev=0.15):
    """
    Calculates a 95% CI heuristically based on a standard error multiplier.
    In production, use Quantile Regression.
    """
    delta = prediction * std_dev * 1.96 # 95% Confidence
    return max(0, prediction - delta), prediction + delta
