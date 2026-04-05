
"""
Vertex AI Model Prediction

This script sends a prediction request to a deployed Vertex AI endpoint.
It assumes the model (v1) is deployed and running.
"""

import os
from google.cloud import aiplatform

PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'effort-estimation-project-id')
REGION = os.environ.get('GCP_REGION', 'us-central1')

def predict(endpoint_name, input_data):
    # Initialize Vertex AI SDK
    aiplatform.init(project=PROJECT_ID, location=REGION)

    print(f"Sending Prediction Request to Endpoint '{endpoint_name}'...")
    print(f"Input: {input_data}")

    endpoint = aiplatform.Endpoint(endpoint_name)

    # In Scikit-Learn Container: Input is array-like: [[f1, f2, f3]]
    prediction = endpoint.predict(instances=input_data)

    print(f"Prediction Result: {prediction.predictions}")
    return prediction.predictions

if __name__ == "__main__":
    # Example input: [[contributor_count, velocity_hours, complexity_score]]
    # Example: 10 contributors, avg 24 hours commit interval, 85 complexity
    input_sample = [[10, 24.0, 85.0]]
    endpoint_res_name = "projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/YOUR_ENDPOINT_ID"

    try:
        predict(endpoint_res_name, input_sample)
    except Exception as e:
        print(f"Error calling endpoint: {e}")
        print("Set valid endpoint_res_name.")
