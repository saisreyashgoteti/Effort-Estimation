
"""
Analysis Processor - Data Ingestion for Gemma 3
This script ingests the JSON output from `data_collection/scraper.js` and prepares it for
analysis by the Gemma 3 (via Vertex AI) and Claude 4.5 models.
"""

import json
import os
import time


RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data_collection/raw_data')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), 'processed_metrics')

def process_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Simulated Gemma 3 Analysis Logic
    # In a real deployment, this would call the Vertex AI generate_content API
    print(f"Processing {data['repository']['full_name']}...")

    commits = data.get('commit_history', [])
    
    # Analyze timestamp differences (Core Gemma function)
    avg_hours_between_commits = 0
    if len(commits) > 1:
        deltas = [float(c['time_since_last_commit_hours']) for c in commits if c['time_since_last_commit_hours']]
        if deltas:
            avg_hours_between_commits = sum(deltas) / len(deltas)

    # Claude 4.5 Complexity Analysis (Placeholder)
    # This would be an API call to assess 'difficulty'
    complexity_score = len(commits) * 0.5  # Simplified heuristic

    # Metrics Output
    metrics = {
        'repo_name': data['repository']['full_name'],
        'scraped_at': data['scraped_at'],
        'contributor_count': len(data['contributors']),
        'commit_velocity_avg_hours': avg_hours_between_commits,
        'complexity_score': complexity_score,
        'agile_readiness': 'High' if avg_hours_between_commits < 48 else 'Medium'
    }

    # Save Results
    output_file = os.path.join(PROCESSED_DATA_DIR, os.path.basename(file_path))
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved analysis for {metrics['repo_name']}")

def main():
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.json')]
    print(f"Found {len(files)} raw data files to process.")

    for file in files:
        process_file(os.path.join(RAW_DATA_DIR, file))

if __name__ == "__main__":
    main()
