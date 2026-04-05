
# Model Training Module (`model_training`)

This directory contains scripts for training and managing the Effort Estimation model on **Google Cloud Vertex AI**.

## Directory Structure
-   `src/task.py`: The core training logic (Random Forest Regressor) that executes **inside** the Vertex AI container.
-   `train_model.py`: Automates the job submission to Vertex AI. It bundles `src/task.py` and waits for completion.
-   `deploy_model.py`: Deploys a trained model artifact to a managed endpoint for real-time inference.
-   `predict_model.py`: Example script to send prediction requests (Contributor Count, Velocity, Complexity) to the endpoint.

## Prerequisites
1.  **GCP Project**: Set `GCP_PROJECT_ID` in your `.env` or environment.
2.  **Service Account**: Ensure `GOOGLE_APPLICATION_CREDENTIALS` points to a JSON key with `Vertex AI Admin` and `Storage Admin` roles.
3.  **Authentication**: Run `gcloud auth application-default login`.

## Usage Workflow

### 1. Train Model
Run the training job. This will launch a container on Vertex AI.
```bash
python3 train_model.py
```
*Output: A `Model Resource Name` (e.g., `projects/123.../models/456`)*.

### 2. Deploy Model
Update `deploy_model.py` with the `Model Resource Name` from step 1.
```bash
python3 deploy_model.py
```
*Output: An `Endpoint Resource Name` (e.g., `projects/123.../endpoints/789`)*.

### 3. Predict Effort (Inference)
Update `predict_model.py` with the `Endpoint Resource Name` from step 2.
```bash
python3 predict_model.py
```

## Model Architecture
The current model uses a `RandomForestRegressor` (Scikit-Learn) trained on:
-   **Contributor Count**: Number of developers.
-   **Commit Velocity**: Avg. hours between commits (from Gemma analysis).
-   **Complexity Score**: Codebase difficulty (from Claude analysis).

**Target**: `Agile Score` (or `Effort/Efficiency`).
