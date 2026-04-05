import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def load_benchmark_data():
    """Loads the core benchmark dataset."""
    csv_path = 'external_data/manual_datasets/research_benchmarks.csv'
    if not os.path.exists(csv_path):
        logger.error(f"Benchmark file not found at {csv_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path).drop_duplicates()
    return df

def generate_mock_github_data(n_samples=1000):
    """
    Simulates the 11k+ GitHub repository dataset processed via Octokit.
    In production, this would read from a database or JSON chunks.
    """
    np.random.seed(42)
    
    data = {
        'total_commits': np.random.randint(50, 5000, n_samples),
        'commit_frequency': np.random.uniform(0.5, 20.0, n_samples), # Commits per day
        'issues_count': np.random.randint(0, 500, n_samples),
        'PR_count': np.random.randint(5, 1000, n_samples),
        'project_duration': np.random.randint(3, 48, n_samples), # Months
        'team_size': np.random.randint(1, 30, n_samples)
    }
    
    df_github = pd.DataFrame(data)
    
    # Compute proxy metrics
    df_github['developer_effort_score'] = (df_github['total_commits'] * 1.5 + df_github['issues_count'] * 2.5) / df_github['team_size']
    df_github['productivity_score'] = df_github['PR_count'] / (df_github['project_duration'] + 1)
    df_github['collaboration_index'] = df_github['PR_count'] / (df_github['total_commits'] + 1)
    
    return df_github

def synthesize_hybrid_dataset():
    """
    Fuses real benchmarks with structural distributions from GitHub data.
    Provides SRS text descriptions.
    """
    df_bench = load_benchmark_data()
    if df_bench.empty:
        return pd.DataFrame(), []
    
    # Map Benchmark 'Effort' to target
    df_bench['effort_pm'] = df_bench['Effort']
    
    srs_texts = df_bench.apply(
        lambda row: f"{row['Project']} built in {row['Language']} using {row['Methodology']}. Scale: {row['KLOC']} KLOC.", 
        axis=1
    ).tolist()
    
    # Expand slightly to simulate merged datasets combining GitHub scale with benchmark truth
    df_expanded = []
    final_texts = []
    
    for _ in range(10): # Simulate expanding dataset matching GitHub structural variances
        mock_gh = generate_mock_github_data(len(df_bench))
        
        df_combo = df_bench.copy()
        df_combo['total_commits'] = mock_gh['total_commits'].values
        df_combo['developer_effort_score'] = mock_gh['developer_effort_score'].values
        df_combo['productivity_score'] = mock_gh['productivity_score'].values
        df_combo['collaboration_index'] = mock_gh['collaboration_index'].values
        
        numeric_features = df_combo[['KLOC', 'TeamSize', 'Duration', 'ComplexityScore', 
                                     'total_commits', 'developer_effort_score', 
                                     'productivity_score', 'collaboration_index', 'effort_pm']].copy()
        
        df_expanded.append(numeric_features)
        final_texts.extend(srs_texts)
        
    final_df = pd.concat(df_expanded, ignore_index=True)
    return final_df, final_texts
