# Capstone Project Report: Effort Estimation of Software Projects

## 1. Introduction
Effort estimation is a process that forms part of the software development life cycle and is key to the assessment of the probable number of hours that may be required for the accomplishment of particular tasks in a software development project. 

In high-level definition, effort estimation in software development is the process of quantifying the amount of work that can be done in terms of person-hours or person-days needed to accomplish a given task or the whole project. This involves the whole process of the SDLC, that is, the gathering of requirements and preparation of specifications, the design, the coding and testing of the software, and the maintenance of the software. 

Effort estimation is an essential part of project management, as it enables one to predict the amount of time necessary to complete a project and coordinate the costs and quality of a project to meet customer expectations.

## 2. Problem Statement
Traditional estimation models (like COCOMO) and static linear regression models often fail with modern software complexities because they heavily rely on static lines of code (LOC). They frequently ignore the semantic nuance, the inherent difficulty of the architecture, and the actual varying capability of developers, which leads to 30-60% of software projects failing due to poor planning and inaccurate estimates.

## 3. Project Objectives
To solve these inaccuracies, this project aims to bridge the gap between static estimation models and modern AI potential. Our primary objectives include:
1. Developing an automated pipeline to scrape and analyze large-scale GitHub repositories.
2. Integrating Large Language Models (LLMs) to extract semantic features like a 'Complexity Score' from commits.
3. Quantifying developer efficiency by mapping and simulating social profiling (LinkedIn data).
4. Training a robust prediction model on Google Cloud Vertex AI to dynamically estimate project effort.

## 4. Proposed System Architecture & Methodology
We conceptualize the solution in a three-tier pipeline:

### 4.1. Data Collection Layer
Targeting a massive pool of datasets, we utilize a custom Node.js script built on **GitHub's Octokit REST API** to scrape over 11,000 repositories. Key metrics extracted include commit timestamps (to calculate coding velocity and gaps) and file addition/deletion metadata. In parallel, a simulated module maps these repository contributors to social network profiles to measure team size and developer collaborative skills.

### 4.2. Analysis Engine (AI Automation)
We utilize **Google Opal** as an automation throughput driver. This engine processes the raw data using state-of-the-art AI:
- **Claude 4.5 Pro Profiling:** Takes in repository structure and commit logs to dynamically output a 'Codebase Difficulty' score.
- **Gemma 3 Integration:** Processes file modification metadata and timestamp differences to derive accurate 'Coding Velocity' and efficiency heuristics.

### 4.3. Prediction Model (Vertex AI)
Finally, these refined, AI-driven metrics (Velocity, Complexity, Team Size) are combined into a standardized vector. This dataset is fed into a Random Forest Regressor trained on **Google Cloud Vertex AI** to output accurate predictions of Project Delivery Time and a final Effort 'Agile Score'.

## 5. Conclusion
By correlating commit velocity via Gemma 3 heuristics with semantic complexity via Claude 4.5 Pro, this architecture achieves a highly dynamic and accurate "Effort Score". The deployment onto Google Cloud Vertex AI ensures that the estimation pipeline is fully scalable to an enterprise level, directly mitigating the risks of poor project planning.
