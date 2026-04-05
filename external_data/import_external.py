
"""
External Data Integration - Benchmarks

This script normalizes and processes manually entered benchmark data (e.g., from research papers like COCOMO, NASA data)
so it can be used to pre-train or calibrate the effort estimation models alongside the scraped GitHub data.

Input: CSV files in `external_data/manual_datasets/`
Output: `external_data/normalized_benchmarks.json`
"""

import os
import pandas as pd
import json

MANUAL_DATA_DIR = os.path.join(os.path.dirname(__file__), 'manual_datasets')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'normalized_benchmarks.json')

def process_benchmark_data():
    if not os.path.exists(MANUAL_DATA_DIR):
        print(f"Directory {MANUAL_DATA_DIR} not found.")
        return

    csv_files = [f for f in os.listdir(MANUAL_DATA_DIR) if f.endswith('.csv')]
    
    all_data = []

    for file in csv_files:
        print(f"Processing manual dataset: {file}...")
        df = pd.read_csv(os.path.join(MANUAL_DATA_DIR, file))
        
        # Normalize columns to match our scraped data schema where possible
        # Our target model uses: contributor_count, commit_velocity, complexity_score
        # Benchmarks use: TeamSize, Duration, KLOC (proxy for complexity/velocity)
        
        # Heuristic Mapping:
        # TeamSize -> contributor_count
        # KLOC/Duration -> commit_velocity_avg_hours (Very rough proxy)
        # ComplexityScore (if present) -> complexity_score
        
        for _, row in df.iterrows():
            # Heuristic Calculation for velocity proxy from benchmarks
            # Assume ~200 working hours/month per person
            # If KLOC is high and Duration is low, velocity must be high (low hours between 'units of work')
            # This is a synthetic mapping to allow model pre-training
            
            velocity_proxy = 40.0 # Default fallback
            if 'KLOC' in row and 'Duration' in row and row['Duration'] > 0:
                 # Invert: Higher duration for same size = Slower velocity
                 velocity_proxy = (row['KLOC'] * 1000) / (row['Duration'] * 20 * 8) 
                 # This is lines per hour. Not hours per commit. Let's invert again.
                 # Hours per 'unit':
                 if velocity_proxy > 0:
                     velocity_proxy = 1 / velocity_proxy

            normalized_entry = {
                "source": f"Research-{file.split('.')[0]}",
                "repo_name": row.get('Project', 'Unknown'),
                "contributor_count": row.get('TeamSize', 1),
                "commit_velocity_avg_hours": abs(velocity_proxy * 10), # Scaling factor
                "complexity_score": row.get('ComplexityScore', row.get('KLOC', 10) * 1.5), # KLOC proxy
                "agile_score": 100 - (row.get('Effort', 100) / 100) # Dummy target calculation
            }
            all_data.append(normalized_entry)

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Normalized {len(all_data)} benchmark entries. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_benchmark_data()
