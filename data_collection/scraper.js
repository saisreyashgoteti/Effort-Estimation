require('dotenv').config();
const { Octokit } = require('@octokit/rest');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

// Initialize Octokit client
const octokit = new Octokit({
  auth: process.env.GITHUB_TOKEN || 'YOUR_GITHUB_TOKEN', // Replace with your token
});

// Configure scraping parameters
const CONFIG = {
  SEARCH_QUERY: 'stars:>1000 language:javascript pushed:>2024-01-01', // Example query to filter repositories
  MAX_REPOS: 11000, // Target: 11,000+ entries
  BATCH_SIZE: 100,  // Process in batches to avoid memory issues
  OUTPUT_DIR: path.join(__dirname, '../data_collection/raw_data'),
  METADATA_RATE_LIMIT_DELAY: 2000,
};

// Ensure output directory exists
if (!fs.existsSync(CONFIG.OUTPUT_DIR)) {
  fs.mkdirSync(CONFIG.OUTPUT_DIR, { recursive: true });
}

/**
 * Identify Key Timestamp Differences (for Gemma model input)
 * @param {Array} commits - Array of commit objects
 * @returns {Array} - Processed commit metadata with time deltas
 */
function processCommitTimestamps(commits) {
  return commits.map((commit, index) => {
    const currentCommitDate = new Date(commit.commit.author.date);
    let timeDiff = 0;
    
    if (index < commits.length - 1) {
      const prevCommitDate = new Date(commits[index + 1].commit.author.date);
      timeDiff = (currentCommitDate - prevCommitDate) / (1000 * 60 * 60); // Difference in hours
    }

    return {
      sha: commit.sha,
      author: commit.commit.author.name,
      email: commit.commit.author.email, // Use for LinkedIn mapping
      date: currentCommitDate.toISOString(),
      message: commit.commit.message,
      time_since_last_commit_hours: timeDiff.toFixed(2),
      files_modified_count: commit.files ? commit.files.length : 0 // Metadata API hooks here
    };
  });
}

/**
 * Fetch detailed metrics for a single repository
 * @param {Object} repo - Repository object from search results
 */
async function processRepository(repo) {
  try {
    console.log(`Processing repo: ${repo.full_name}...`);

    // 1. Fetch Repository Metadata
    const repoMetadata = {
      id: repo.id,
      name: repo.name,
      full_name: repo.full_name,
      owner: repo.owner.login,
      stars: repo.stargazers_count,
      forks: repo.forks_count,
      language: repo.language,
      created_at: repo.created_at,
      pushed_at: repo.pushed_at,
      description: repo.description,
      topics: repo.topics
    };

    // 2. Fetch Recent Commits (for Timestamp Analysis)
    const { data: commits } = await octokit.repos.listCommits({
      owner: repo.owner.login,
      repo: repo.name,
      per_page: 50 // Fetch last 50 commits for analysis
    });

    const commitMetrics = processCommitTimestamps(commits);

    // 3. (Optional) Fetch Contributors (for Team Dynamics)
    const { data: contributors } = await octokit.repos.listContributors({
      owner: repo.owner.login,
      repo: repo.name,
      per_page: 10
    });

    // Structure for Analysis Engine (Gemma 3 / Claude 4.5)
    const analysisPayload = {
      repository: repoMetadata,
      commit_history: commitMetrics,
      contributors: contributors.map(c => ({ login: c.login, contributions: c.contributions })),
      analysis_status: 'pending', // To be picked up by the Analysis module
      scraped_at: new Date().toISOString()
    };

    // Save to JSON file for processing by the Analysis Engine
    const fileName = `${repo.owner.login}_${repo.name}.json`.replace(/[\/\\?%*:|"<>]/g, '-');
    fs.writeFileSync(path.join(CONFIG.OUTPUT_DIR, fileName), JSON.stringify(analysisPayload, null, 2));
    
    console.log(`Saved data for ${repo.full_name}`);

  } catch (error) {
    console.error(`Error processing repo ${repo.full_name}:`, error.message);
  }
}

/**
 * Main scraping function driven by search query
 */
async function scrapeGithubData() {
  console.log('Starting GitHub Data Scraping Pipeline...');
  console.log(`Target: ${CONFIG.MAX_REPOS} repositories`);

  let processedCount = 0;
  let page = 1;

  while (processedCount < CONFIG.MAX_REPOS) {
    try {
      const { data } = await octokit.search.repos({
        q: CONFIG.SEARCH_QUERY,
        sort: 'stars',
        order: 'desc',
        per_page: CONFIG.BATCH_SIZE,
        page: page
      });

      if (!data.items || data.items.length === 0) {
        console.log('No more repositories found.');
        break;
      }

      for (const repo of data.items) {
        if (processedCount >= CONFIG.MAX_REPOS) break;
        
        await processRepository(repo);
        processedCount++;
        
        // Rate limiting strategy
        await new Promise(resolve => setTimeout(resolve, CONFIG.METADATA_RATE_LIMIT_DELAY));
      }

      console.log(`Batch ${page} complete. Total processed: ${processedCount}`);
      page++;

    } catch (error) {
      console.error('Error in main loop:', error.message);
      // Wait longer on robust error (likely rate limit)
      await new Promise(resolve => setTimeout(resolve, 60000));
    }
  }

  console.log('Scraping Pipeline Complete.');
}

// Execute the scraper
scrapeGithubData();
