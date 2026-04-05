"""
Mock Data Generator

Generates sample repository JSON files in `data_collection/raw_data/`
to simulate the output of the GitHub scraper. Useful for testing the analysis pipeline
without needing a valid GitHub API token.
"""

import json

import os
import random
from datetime import datetime, timedelta

# Ensure output is in data_collection/raw_data regardless of execution path
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'raw_data')
NUM_REPOS = 10

def generate_mock_repo(index):
    repo_name = f"project-{index}"
    owner = "mock-user"
    
    # Generate random commits
    commits = []
    base_time = datetime.now() - timedelta(days=365)
    
    for i in range(random.randint(50, 200)):
        # Random time gap between commits (hours)
        hours_gap = random.expovariate(1/24) # avg 24 hours between commits
        commit_time = base_time + timedelta(hours=hours_gap*i)
        
        commits.append({
            "sha": f"sha{i}",
            "author": f"dev-{random.randint(1, 5)}",
            "email": f"dev-{random.randint(1, 5)}@example.com",
            "date": commit_time.isoformat(),
            "message": f"update feature {i}",
            "time_since_last_commit_hours": round(hours_gap, 2),
            "files_modified_count": random.randint(1, 10)
        })
        
    return {
        "repository": {
            "id": index,
            "name": repo_name,
            "full_name": f"{owner}/{repo_name}",
            "owner": owner,
            "stars": random.randint(100, 5000),
            "forks": random.randint(10, 1000),
            "language": random.choice(["JavaScript", "Python", "Java", "Go"]),
            "created_at": (datetime.now() - timedelta(days=400)).isoformat(),
            "pushed_at": datetime.now().isoformat(),
            "description": "A mock repository for testing effort estimation.",
            "topics": ["ai", "machine-learning", "web"]
        },
        "commit_history": commits,
        "contributors": [
            {"login": f"dev-{i}", "contributions": random.randint(10, 100)} for i in range(1, 6)
        ],
        "analysis_status": "pending",
        "scraped_at": datetime.now().isoformat()
    }

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Generating {NUM_REPOS} mock repositories in {OUTPUT_DIR}...")
    
    for i in range(NUM_REPOS):
        data = generate_mock_repo(i)
        filename = f"mock-user_project-{i}.json"
        
        with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
            json.dump(data, f, indent=2)
            
    print("Mock data generation complete.")

if __name__ == "__main__":
    main()
