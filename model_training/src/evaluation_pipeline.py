import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sentence_transformers import SentenceTransformer

# Adjust sys.path to easily import from existing local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_ingestion import synthesize_hybrid_dataset, load_benchmark_data, generate_mock_github_data
from task import build_end_to_end_pipeline
from model import evaluate_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_visualizations(y_test, y_pred, output_path="artifacts/evaluation_plots.png"):
    """STEP 8: VISUALIZATION"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(14, 6))
    
    # 1. Actual vs Predicted Scatter
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue', edgecolor=None)
    
    # Ideal prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', linewidth=2)
    
    plt.title("Actual vs Predicted Effort")
    plt.xlabel("Actual Effort (Person-Months)")
    plt.ylabel("Predicted Effort (Person-Months)")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Residual Errors Distribution
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, color='purple', bins=30)
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.title("Residual Errors Distribution")
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Saved evaluation plots to {output_path}")

def run_evaluation_pipeline():
    logger.info("Initializing Robust Evaluation Pipeline...")
    
    # ----------------------------------------------------
    # DATA PREPARATION
    # ----------------------------------------------------
    df_numeric, texts = synthesize_hybrid_dataset()
    if df_numeric.empty:
        logger.error("Dataset could not be loaded.")
        return
    
    y = df_numeric.pop('effort_pm').values
    
    logger.info("Generating NLP Embeddings for dataset...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = sentence_model.encode(texts)
    
    X_full = np.hstack((df_numeric.values, embeddings))
    
    # ----------------------------------------------------
    # STEP 1: TRAIN-TEST SPLIT
    # ----------------------------------------------------
    logger.info("Performing 80-20 Train-Test Split BEFORE any preprocessing...")
    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
    
    pipeline = build_end_to_end_pipeline(df_numeric, embeddings)
    
    # ----------------------------------------------------
    # STEP 2 & 3: MODEL TRAINING AND PREDICTION
    # ----------------------------------------------------
    logger.info("Training Model on Training Data Only...")
    pipeline.fit(X_train, y_train)
    
    # Predict on Training Data (for Overfitting Check)
    y_train_pred = pipeline.predict(X_train)
    train_metrics = evaluate_metrics(y_train, y_train_pred)
    
    # Predict on Testing Data
    logger.info("Generating predictions on Test Data...")
    y_test_pred = pipeline.predict(X_test)
    
    # ----------------------------------------------------
    # STEP 4: EVALUATION METRICS
    # ----------------------------------------------------
    test_metrics = evaluate_metrics(y_test, y_test_pred)
    
    logger.info(f"--- Training Metrics ---\n{train_metrics}")
    logger.info(f"--- Test Metrics ---\n{test_metrics}")
    
    # ----------------------------------------------------
    # STEP 5: CROSS-VALIDATION (CRITICAL)
    # ----------------------------------------------------
    logger.info("Running 5-Fold Cross Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Using negative MAE via cross_val_score to avoid expensive nested prediction returns if not needed,
    # but we can also use cross_val_predict for full metrics.
    cv_maes = -cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
    avg_cv_mae = np.mean(cv_maes)
    logger.info(f"Cross-Validation MAE scores across folds: {cv_maes}")
    logger.info(f"Average CV MAE: {avg_cv_mae:.4f}")
    
    # ----------------------------------------------------
    # STEP 6: CHECK FOR OVERFITTING
    # ----------------------------------------------------
    logger.info("Checking for possible Overfitting or Leakage...")
    is_overfitting = False
    leakage_warning = False
    
    if train_metrics['R2'] > 0.98 and test_metrics['R2'] < 0.90:
        is_overfitting = True
        logger.warning("HIGH RISK OF OVERFITTING: Train R2 is exceptionally high compared to Test R2.")
    
    if test_metrics['R2'] > 0.99 or test_metrics['MAE'] < 0.1:
        leakage_warning = True
        logger.warning("DATA LEAKAGE DETECTED: Test metrics are unrealistically perfect. Check for overlapping samples or target-derived features in X.")
        
    if not is_overfitting and not leakage_warning:
        logger.info("Model generalization appears healthy based on metric divergence.")

    # ----------------------------------------------------
    # STEP 7: EXTERNAL VALIDATION (Train on Kaggle/GH, Test on Benchmark)
    # ----------------------------------------------------
    logger.info("Running External Validation strategy...")
    try:
        # Load Raw Kaggle / Mock GH Data
        # Using the synthetically expanded github dataset representing the "Kaggle dataset"
        # We simulate the Github-only features vs benchmark features
        df_bench_raw = load_benchmark_data()
        
        # We need a shared feature space. Both must have common numerical columns
        shared_cols = ['KLOC', 'TeamSize']
        
        # Create a tiny custom pipeline just for the shared feature external validation check
        gh_data = generate_mock_github_data(1000)
        # Map GitHub metrics to shared columns heuristically
        X_ext_train = pd.DataFrame({
            'KLOC': gh_data['total_commits'] * 50 / 1000.0,
            'TeamSize': gh_data['team_size']
        })
        y_ext_train = gh_data['total_commits'] / 50.0  # Synthetic effort representation for GH
        
        X_ext_test = df_bench_raw[shared_cols]
        y_ext_test = df_bench_raw['Effort']
        
        from sklearn.ensemble import RandomForestRegressor
        ext_model = RandomForestRegressor(random_state=42)
        ext_model.fit(X_ext_train, y_ext_train)
        
        ext_preds = ext_model.predict(X_ext_test)
        ext_metrics = evaluate_metrics(y_ext_test, ext_preds)
        
        logger.info(f"External Validation on Benchmark subset (using simplified shared features):")
        logger.info(f"External MAE: {ext_metrics['MAE']:.4f}, External R2: {ext_metrics['R2']:.4f}")
    except Exception as e:
        logger.warning(f"External Validation skipped or failed: {e}")

    # ----------------------------------------------------
    # STEP 8: VISUALIZATION
    # ----------------------------------------------------
    plot_visualizations(y_test, y_test_pred)

    # ----------------------------------------------------
    # STEP 9: INTERPRETATION & SUMMARY
    # ----------------------------------------------------
    interpretation = f"""
======================================================
📊 MODEL EVALUATION SUMMARY
======================================================
1. TEST SET PERFORMANCE:
   - MAE:     {test_metrics['MAE']:.4f} person-months
   - R²:      {test_metrics['R2']:.4f}
   - MMRE:    {test_metrics['MMRE']:.2f}%
   - Pred(25): {test_metrics['Pred_25']:.2f}%

2. CROSS VALIDATION:
   - 5-Fold Avg MAE: {avg_cv_mae:.4f} person-months

3. DIAGNOSTICS:
   - Overfitting: {'DETECTED' if is_overfitting else 'Negative'}
   - Data Leakage: {'WARNING/LIKELY' if leakage_warning else 'Negative'}

INTERPRETATION:
- Pred(25) indicates that {test_metrics['Pred_25']:.2f}% of our predictions are within 25% of the actual real effort.
- Mean Magnitude of Relative Error (MMRE) is {test_metrics['MMRE']:.2f}%.
- If R² is exactly 1.0 or MAE < 0.1, the pipeline has critically memorized target values (check DerivedFeaturesTransformer).
- A robust generalization should see Pred(25) > 30% and MMRE < 50% for software engineering tasks.
======================================================
    """
    logger.info(interpretation)
    
    # Save the interpretation report
    with open("artifacts/evaluation_report.md", "w") as f:
        f.write(interpretation)
    logger.info("Evaluation report saved to artifacts/evaluation_report.md")

if __name__ == '__main__':
    run_evaluation_pipeline()
