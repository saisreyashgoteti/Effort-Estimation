# Project Plan: Automated Codebase Analysis and Talent Identification Pipeline

## Overview
This roadmap outlines the remaining 64% of the work required to complete the project within the next 4 weeks.

## Timeline: 4 Weeks

### Week 1: Data Collection & Ingestion (Completion)
**Goal**: Finalize scraping and ingestion of all relevant repository data.

- [x] **GitHub Scraping Optimization**:
  - Optimize `Octokit` scripts to handle rate limiting and maximize throughput.
  - Scale scraping to reach 11,000+ entries.
- [x] **Metadata Extraction**:
  - Ensure all commit timestamps and file modification metadata are captured correctly.
  - Implement automated scraping for user social profiles (LinkedIn).
- **Deliverable**: Complete dataset ingested into the staging environment.

### Week 2: Analysis & Metrics Engine
**Goal**: Process raw data to derive meaningful metrics.

- [x] **Complexity Analysis**:
  - Integrate **Claude 4.5 Pro** prompts to assess codebase difficulty at scale.
- [x] **Gemma 3 Integration**:
  - Deploy **Gemma 3** model pipeline to identify key timestamp differences.
  - Apply logic to correlate commit activity with project milestones.
- [x] **Profile Mapping**:
  - Automate linking of GitHub contributors to their LinkedIn profiles.
  - Develop scoring algorithm for developer proficiency (coding time, efficiency).
- **Deliverable**: Analyzed data ready for model training, with all metrics calculated.

### Week 3: Model Training & Integration
**Goal**: Train the multi-modular models using VertexAI.

- [ ] **Data Verification**:
  - Conduct manual human intervention on a subset of data to verify accuracy.
  - Validate derived metrics against known benchmarks.
- [x] **VertexAI Pipeline**:
  - Set up training jobs on Google Cloud Platform with high-level GPUs.
  - Train the model to predict project outcomes based on the aggregated metrics.
- [x] **External Data Integration**:
  - Incorporate manually scraped secondary datasets and research paper data.
- **Deliverable**: Trained models ready for validation.

### Week 4: Testing & Research Articulation
**Goal**: Validate the entire system and prepare final documentation.

- [ ] **System Testing**:
  - End-to-end testing of the pipeline from scraping to prediction.
  - Performance tuning and optimization.
- [x] **Research Articulation**:
  - Document methodology, analysis techniques, and findings.
  - Prepare final project report and presentation materials.
- **Deliverable**: Final Project Report, Presentation, and deployed system.
