# Project: Automated Codebase Analysis and Talent Identification Pipeline

## Overview
This project leverages advanced AI models and automated scraping tools to analyze thousands of GitHub repositories. By integrating GitHub's Metadata API, LinkedIn profiles, and social data, we aim to derive comprehensive metrics on developer skills, project difficulty, and team dynamics. The ultimate goal is to build a robust dataset and model capable of predicting project outcomes, developer efficiency, and aptitude.

## Architecture & Technology Stack

### 1. Data Collection & Ingestion
- **Source**: GitHub Repositories (11,000+ entries targeted).
- **Tooling**:
  - **Octokit**: Official GitHub API toolkit for accessing repositories, user data, and contribution stats.
  - **Google Opal**: Automation driver for high-throughput data collection.

### 2. Analysis Engine
- **Core Engine**: **Google Gemini 3 Pro**.
- **Model**: **Gemma 3** (Open Source Model).
  - Specialized in identifying key timestamp differences across commits.
  - Utilizes file modification metadata via GitHub Metadata API.
- **Complexity Analysis**: **Claude 4.5 Pro** (used for assessing codebase difficulty).

### 3. Data Enrichment
- **Sources**:
  - **Social Profiles**: LinkedIn and other social platforms to map developer skills.
  - **External Data**: Manual scraping from secondary datasets and research papers.
- **Metrics Derived**:
  - **Developer**: Coding time, efficiency, performance, aptitude, contribution, collaborativeness.
  - **Project**: Delivery time, agile score, productivity, KPIs, deliverables.

### 4. Human Verification
- A manual intervention layer ensures data scrutinized by the AI models is verified before final processing.

### 5. Model Training & Deployment
- **Platform**: **Google Cloud Platform (GCP) VertexAI**.
- **Infrastructure**: High-level GPUs for processing complex relations within the data.
- **Output**: Multi-modular models trained on custom datasets.

## Development Status
- **Current Progress**: 36% complete.
- **Target**: Completion within 4 weeks.

## Roadmap
1. **Weeks 1-3**: Complete data collection, pipelines, and model training.
2. **Week 4**: Testing, validation, and research articulation.

## Directory Structure
- `data_collection/`: Scripts for GitHub scraping and Octokit integration.
- `analysis/`: Gemma 3 model implementation and metadata processing.
- `external_data/`: Tools for scraping LinkedIn and social profiles.
- `model_training/`: VertexAI integration and training scripts.
- `docs/`: Project documentation and research papers.
