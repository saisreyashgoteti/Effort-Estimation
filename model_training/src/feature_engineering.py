import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DerivedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to calculate derived metrics dynamically inside a Pipeline.
    Prevents data leakage by only applying transformations row-by-row on the active split.
    """
    def __init__(self):
        # Indices or names can be tracked here
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=[
            'KLOC', 'TeamSize', 'Duration', 'ComplexityScore', 
            'total_commits', 'developer_effort_score', 
            'productivity_score', 'collaboration_index'
        ])
        
        # Calculate derived features safely
        # Replace 0s with 1s to prevent division by zero
        X_df['Effort_per_KLOC'] = X_df['TeamSize'] * X_df['Duration'] / (X_df['KLOC'].replace(0, 1))
        X_df['Team_Productivity'] = X_df['KLOC'] / (X_df['TeamSize'].replace(0, 1))
        X_df['Duration_per_KLOC'] = X_df['Duration'] / (X_df['KLOC'].replace(0, 1))
        
        # Risk score based on high complexity and high duration but low team size
        X_df['Risk_Score'] = (X_df['ComplexityScore'] * X_df['Duration']) / (X_df['TeamSize'].replace(0, 1))
        
        return X_df.values
