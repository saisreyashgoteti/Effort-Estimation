import os
import pickle
import logging
import numpy as np
import shap

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sentence_transformers import SentenceTransformer

from data_ingestion import synthesize_hybrid_dataset
from feature_engineering import DerivedFeaturesTransformer
from model import get_ensemble_model, evaluate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_end_to_end_pipeline(df_numeric, X_embeddings):
    num_indices = list(range(df_numeric.shape[1]))
    embed_indices = list(range(df_numeric.shape[1], df_numeric.shape[1] + X_embeddings.shape[1]))
    
    # 1. Feature Preprocessing (Numerical & Derived + Imputation + Scaling)
    numeric_transformer = Pipeline([
        ('derived_features', DerivedFeaturesTransformer()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # NLP embeddings pass through without scaling to avoid distortion
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_indices),
        ('embed', 'passthrough', embed_indices)
    ])
    
    ensemble_model = get_ensemble_model()
    
    main_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('ensemble', ensemble_model)
    ])
    
    return main_pipeline

def main():
    logger.info("Initializing Hybrid Software Target Estimation Pipeline...")
    
    # 1. Ingestion
    df_numeric, texts = synthesize_hybrid_dataset()
    if df_numeric.empty:
        logger.error("Failed to load dataset.")
        return
        
    y = df_numeric.pop('effort_pm').values
    
    # 2. NLP Feature Extraction (Pre-split is safe because text->vector is deterministic and unaware of y)
    logger.info("Generating NLP Embeddings...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = sentence_model.encode(texts)
    
    # Define complete feature block
    X_full = np.hstack((df_numeric.values, embeddings))
    
    # 3. Train-Test Split (BEFORE PREPROCESSING!)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
    
    # 4. Pipeline Execution & K-Fold Validation
    pipeline = build_end_to_end_pipeline(df_numeric, embeddings)
    
    logger.info("Running 5-Fold Cross Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds = cross_val_predict(pipeline, X_train, y_train, cv=kf)
    
    cv_metrics = evaluate_metrics(y_train, cv_preds)
    logger.info(f"Cross-Validation Metrics:\n{cv_metrics}")
    
    # 5. Final Model Training
    logger.info("Training Final Model on entire train split...")
    pipeline.fit(X_train, y_train)
    
    # 6. Test Set Evaluation
    test_preds = pipeline.predict(X_test)
    test_metrics = evaluate_metrics(y_test, test_preds)
    logger.info(f"Hold-out Test Metrics:\n{test_metrics}")
    
    # 7. Model Serialization
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/hybrid_effort_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    logger.info("Saved pipeline to artifacts/hybrid_effort_model.pkl")

if __name__ == '__main__':
    main()
