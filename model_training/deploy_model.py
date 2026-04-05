
"""
Vertex AI Model Deployment

This script deploys a trained model artifact to a Vertex AI endpoint.
It assumes the model has been registered in the Vertex AI Model Registry.
"""

import os
from google.cloud import aiplatform

PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'effort-estimation-project-id')
REGION = os.environ.get('GCP_REGION', 'us-central1')

def deploy_model(model_resource_name, endpoint_name="production-effort-api"):
    # Initialize Vertex AI SDK
    aiplatform.init(project=PROJECT_ID, location=REGION)

    print(f"Deploying model '{model_resource_name}' to endpoint '{endpoint_name}'...")

    # Load Model (by Resource Name or ID)
    model = aiplatform.Model(model_resource_name)

    # Create an Endpoint
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)

    # Deploy Model to Endpoint
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name="v1-effort-estimation",
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3,
    )

    print(f"Model Deployed. Endpoint Name: {endpoint.resource_name}")
    return endpoint

if __name__ == "__main__":
    # Example Resource Name (would be retrieved from training job or user input)
    model_res_name = "projects/YOUR_PROJECT_ID/locations/us-central1/models/YOUR_MODEL_ID"
    
    try:
        deploy_model(model_res_name)
    except Exception as e:
        print(f"Error deploying model: {e}")
        print("Set valid model_res_name.")
