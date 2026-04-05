"""
LinkedIn Profile Mapper (Simulation)

This script attempts to map GitHub contributor emails to LinkedIn profiles.
Due to strict anti-scraping measures on LinkedIn, this script uses a placeholder
approach or would require a paid API like Proxycurl or a custom scraper with residential proxies.

Input: 
    - List of emails/names from `data_collection/raw_data/*.json`
Output: 
    - `external_data/social_profiles.json` mapping emails to LinkedIn URLs and skills.
"""

import json
import os
import random

RAW_DATA_DIR = '../data_collection/raw_data'
OUTPUT_FILE = 'social_profiles.json'

# Mock Database of Skills (Simulating LinkedIn Profile Data)
MOCK_SKILLS_DB = {
    'javascript': ['React', 'Node.js', 'Vue.js', 'TypeScript'],
    'python': ['Django', 'Flask', 'Pandas', 'NumPy', 'TensorFlow'],
    'java': ['Spring Boot', 'Hibernate', 'Maven', 'Gradle'],
    'go': ['Docker', 'Kubernetes', 'gRPC'],
    'rust': ['Actix', 'Tokyo', 'WebAssembly']
}

def mock_enrich_profile(email, name):
    """
    Simulates fetching profile data from LinkedIn based on email/name.
    In a real scenario, this would call an external API.
    """
    # Randomly assign skills for demonstration
    primary_lang = random.choice(list(MOCK_SKILLS_DB.keys()))
    skills = MOCK_SKILLS_DB[primary_lang]
    
    return {
        'github_email': email,
        'name': name,
        'linkedin_url': f"https://www.linkedin.com/in/{name.replace(' ', '-').lower()}",
        'skills': skills,
        'experience_years': random.randint(1, 15),
        'education': 'B.Tech Computer Science',
        'verified': True
    }

def process_contributors():
    profiles = []
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Directory {RAW_DATA_DIR} not found. Run scraper first.")
        return

    # iterate through scraped repositories to find contributors
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith('.json'):
            with open(os.path.join(RAW_DATA_DIR, filename), 'r') as f:
                data = json.load(f)
                
                # Extract unique commit authors (better than just top contributors)
                # In a real app, we'd dedup by email
                seen_emails = set()
                for commit in data.get('commit_history', []):
                    email = commit.get('email')
                    name = commit.get('author')
                    
                    if email and email not in seen_emails and 'users.noreply' not in email:
                        seen_emails.add(email)
                        profile = mock_enrich_profile(email, name)
                        profiles.append(profile)
                        print(f"Enriched profile for {name} ({email})")

    # Save Mapping
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(profiles, f, indent=2)
    
    print(f"Successfully mapped {len(profiles)} profiles.")

if __name__ == "__main__":
    process_contributors()
