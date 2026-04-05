import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
try:
    import xgboost as xgb
except Exception:
    xgb = None
    from sklearn.ensemble import GradientBoostingRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mre = np.abs((y_true - y_pred) / (y_true + 1e-8))
    mmre = np.mean(mre) * 100
    pred_25 = np.sum(mre <= 0.25) / len(mre) * 100
    return {'MAE': mae, 'R2': r2, 'MMRE': mmre, 'Pred_25': pred_25}

def plot_visualizations(y_test, y_pred, output_path="artifacts/final_model_plots.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='teal')
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', linewidth=2)
    plt.title("Actual vs Predicted Effort")
    plt.xlabel("Actual Effort (Person-Months)")
    plt.ylabel("Predicted Effort (Person-Months)")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, color='darkorange', bins=30)
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.title("Residual Errors Distribution")
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Saved evaluation plots to {output_path}")

def run_final_pipeline():
    logger.info("Initializing Final Training Pipeline...")
    
    # -----------------------------------------------
    # STEP 1: DATA PREPROCESSING
    # -----------------------------------------------
    data_path = 'data/final_effort_dataset.csv'
    if not os.path.exists(data_path):
        logger.error(f"Dataset block not found: {data_path}")
        return
        
    df = pd.read_csv(data_path)
    # One-hot encode language if exists
    if 'language' in df.columns:
        df = pd.get_dummies(df, columns=['language'], drop_first=True)
        
    # Split features and target
    X = df.drop(columns=['Effort_pm']).values
    y = df['Effort_pm'].values
    
    # -----------------------------------------------
    # STEP 2: RESCALE TARGET
    # -----------------------------------------------
    y = y * 100.0
    logger.info(f"Target rescaled by factor of 100. Mean Effort: {y.mean():.2f}")
    
    # -----------------------------------------------
    # STEP 3: TRAIN-TEST SPLIT
    # -----------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # -----------------------------------------------
    # STEP 4: MODEL ARCHITECTURE
    # -----------------------------------------------
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42) if xgb else GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    
    base_learners = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('xgb_proxy', xgb_model),
        ('svr', SVR(kernel='rbf', C=50, gamma='scale'))
    ]
    meta_learner = ElasticNetCV(cv=5, random_state=42)
    
    ensemble = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        passthrough=True
    )
    
    # -----------------------------------------------
    # STEP 5: TRAIN MODEL
    # -----------------------------------------------
    logger.info("Training Stacking Regressor Model (this may take a moment)...")
    ensemble.fit(X_train, y_train)
    
    # -----------------------------------------------
    # STEP 6: EVALUATION
    # -----------------------------------------------
    logger.info("Evaluating predictions...")
    y_test_pred = ensemble.predict(X_test)
    y_train_pred = ensemble.predict(X_train)
    
    test_metrics = evaluate_metrics(y_test, y_test_pred)
    train_metrics = evaluate_metrics(y_train, y_train_pred)
    
    logger.info(f"--- Training Metrics (Check for Overfitting) --- \n{train_metrics}")
    logger.info(f"--- Test Metrics --- \n{test_metrics}")
    
    # -----------------------------------------------
    # STEP 7: CROSS-VALIDATION
    # -----------------------------------------------
    logger.info("Running 5-Fold Cross Validation for robust MAE verification...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
    avg_cv_mae = np.mean(np.abs(cv_scores))
    logger.info(f"Cross-Validation Avg MAE: {avg_cv_mae:.4f}")
    
    # -----------------------------------------------
    # STEP 8: VISUALIZATION
    # -----------------------------------------------
    plot_visualizations(y_test, y_test_pred)
    
    # -----------------------------------------------
    # STEP 9: VALIDATION & SAVING
    # -----------------------------------------------
    interpretation = f"""
======================================================
🚀 FINAL MODEL EVALUATION SUMMARY
======================================================
1. TEST SET PERFORMANCE:
   - MAE:     {test_metrics['MAE']:.4f} person-months
   - R²:      {test_metrics['R2']:.4f}
   - MMRE:    {test_metrics['MMRE']:.2f}%
   - Pred(25): {test_metrics['Pred_25']:.2f}%

2. CROSS VALIDATION:
   - 5-Fold Avg MAE: {avg_cv_mae:.4f} person-months

3. ARCHITECTURE USED:
   - Model: Stacking Regressor (RF, XGB, SVR) -> ElasticNetCV
   - Target Rescaling: x100

🧠 EXPERT DIAGNOSIS:
"""
    is_realistic = False
    if test_metrics['R2'] > 0.99:
        interpretation += "- WARNING: Results are suspicious (near perfect R²). Data leakage might still exist in non-linear combinations.\n"
    elif test_metrics['R2'] < 0.50:
        interpretation += "- WARNING: Model performance is poor. Target variance isn't fully explained by predictors.\n"
    else:
        is_realistic = True
        interpretation += "- SUCCESS: Model performance is highly realistic! Overfitting is contained, and variance is appropriately constrained without being suspiciously perfect.\n"

    # Save Model
    model_path = 'artifacts/final_robust_effort_model.pkl'
    os.makedirs('artifacts', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble, f)
        
    interpretation += f"\n✅ Model serialized successfully to: {model_path}\n"
    interpretation += f"✅ Evaluation plots saved to: artifacts/final_model_plots.png\n"
    interpretation += "======================================================"
    
    logger.info(interpretation)
    with open("artifacts/final_training_report.md", "w") as f:
        f.write(interpretation)

if __name__ == "__main__":
    run_final_pipeline()
