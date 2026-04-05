
"""
Vertex AI Pipeline - Model Training Initiator.

This script triggers the Vertex AI training job using the source code in `model_training/src/task.py`.
It assumes `model_training/src` is the Python source directory.

Requirements:
  - `gcloud` authenticated with a project
  - `google-cloud-aiplatform` python package
"""

import os
import argparse
from google.cloud import aiplatform

PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'effort-estimation-project-id')
REGION = os.environ.get('GCP_REGION', 'us-central1')
BUCKET_NAME = f"gs://{PROJECT_ID}-effort-estimation"  # Example bucket

def run_training_job():
    # Initialize Vertex AI SDK
    aiplatform.init(project=PROJECT_ID, location=REGION)

    print(f"Submitting Custom Training Job to '{REGION}'...")

    job = aiplatform.CustomTrainingJob(
        display_name="effort-estimation-v1",
        script_path="model_training/src/task.py",  # The script we just created
        container_uri="us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest",
        requirements=["pandas", "scikit-learn", "google-cloud-storage"],
        model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/scikit-learn-cpu.0-23:latest",
    )

    # In a real scenario, we'd pass the input dataset URI
    # For now, we simulate data generation inside the task
    model = job.run(
        model_display_name="effort-estimation-model",
        args=[
            f"--model-dir={BUCKET_NAME}/models/v1",
        ],
        replica_count=1,
        machine_type="n1-standard-4",
        sync=True,  # Wait for job to complete
    )

    print("Model Training Job Completed.")
    print(f"Model Resource Name: {model.resource_name}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    try:
        run_training_job()
    except Exception as e:
        print(f"Error submitting job: {e}")
        print("Note: Ensure you have authenticated with `gcloud auth login` and set `GCP_PROJECT_ID`.")
