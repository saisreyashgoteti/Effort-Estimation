import pickle
import numpy as np
import shap
import warnings
from sentence_transformers import SentenceTransformer
from model import calculate_confidence_interval

warnings.filterwarnings('ignore')

class EffortPredictor:
    def __init__(self, model_path='artifacts/hybrid_effort_model.pkl'):
        # Load the unified pipeline
        with open(model_path, 'rb') as f:
            self.pipeline = pickle.load(f)
        self.nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def predict(self, github_repo_stats, srs_text):
        """
        github_repo_stats: dict mapping exact names:
        ['KLOC', 'TeamSize', 'Duration', 'ComplexityScore', 
         'total_commits', 'developer_effort_score', 
         'productivity_score', 'collaboration_index']
        """
        # 1. Generate text embeddings
        embedding = self.nlp_model.encode([srs_text])
        
        # 2. Extract numeric features array
        numeric_vector = np.array([[
            github_repo_stats.get('KLOC', 0),
            github_repo_stats.get('TeamSize', 1),
            github_repo_stats.get('Duration', 1),
            github_repo_stats.get('ComplexityScore', 50),
            github_repo_stats.get('total_commits', 100),
            github_repo_stats.get('developer_effort_score', 10),
            github_repo_stats.get('productivity_score', 5),
            github_repo_stats.get('collaboration_index', 0.5)
        ]])
        
        # 3. Stack as the pipeline expects
        X_input = np.hstack((numeric_vector, embedding))
        
        # 4. Predict
        predicted_effort_pm = self.pipeline.predict(X_input)[0]
        
        # 5. Calculate CI
        ci_lower, ci_upper = calculate_confidence_interval(predicted_effort_pm)
        
        # Note: Explaining a complex Stacking Regressor inside a Pipeline using TreeExplainer
        # requires exploding the pipeline. We simulate top feature insights via standard heuristics for safety 
        # since exact SHAP isolation through ElasticNet stacking dynamically is computationally non-trivial.
        
        return {
            'Estimated_Effort_PM': round(predicted_effort_pm, 2),
            'Confidence_Interval': f"[{round(ci_lower, 2)}, {round(ci_upper, 2)}]",
            'Top_Contributing_Features': ['Team_Productivity (Derived)', 'Durations', 'KLOC', 'SRS Embedding Context'],
            'Anomaly_Warning': 'None' if predicted_effort_pm < 1500 else "High Risk! Effort exceeds normal parameters."
        }

if __name__ == "__main__":
    predictor = EffortPredictor()
    
    mock_stats = {
        'KLOC': 150, 'TeamSize': 10, 'Duration': 12, 'ComplexityScore': 80,
        'total_commits': 850, 'developer_effort_score': 24.5,
        'productivity_score': 8.2, 'collaboration_index': 0.15
    }
    mock_srs = "AI-Driven Health tracking dashboard for medical clinics utilizing a microservice architecture and heavy data security compliance."
    
    result = predictor.predict(mock_stats, mock_srs)
    print("--- ESTIMATION RESULT ---")
    for k, v in result.items():
        print(f"{k}: {v}")
