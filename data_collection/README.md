# Data Collection Module

## Setup
1.  **Install Dependencies**:
    ```bash
    npm install
    ```
2.  **Configure API Keys**:
    -   Rename `.env.example` to `.env` (if applicable) or edit `.env` in the root directory.
    -   Add your `GITHUB_TOKEN`.

## Usage
Run the scraper to fetch repository data:
```bash
node scraper.js
```

Or generate mock data for testing (doesn't require a GitHub Token):
```bash
python3 generate_mock_data.py
```

## Configuration
Edit `scraper.js` to change:
-   `SEARCH_QUERY`: The GitHub search query (default: `stars:>1000 language:javascript`).
-   `MAX_REPOS`: The number of repositories to scrape (default: `11000`).
-   `OUTPUT_DIR`: Where the raw JSON files are saved (default: `raw_data/`).

## Output
The script generates one JSON file per repository in `raw_data/`. Each file contains:
-   **Repository Metadata**: Stars, forks, owner, etc.
-   **Commit History**: Timestamps, authors, and calculated time differences (for Gemma model).
-   **Contributors**: List of contributors for team analysis.
