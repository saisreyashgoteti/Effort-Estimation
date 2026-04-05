require('dotenv').config({ path: '../.env' });
const { Octokit } = require('@octokit/rest');
const fs = require('fs');
const path = require('path');

const owner = process.argv[2];
const repo = process.argv[3];

if (!owner || !repo) {
  console.error(JSON.stringify({ error: "Usage: node fetch_single_repo.js <owner> <repo>" }));
  process.exit(1);
}

const octokit = new Octokit({
  auth: process.env.GITHUB_TOKEN || '',
});

async function fetchRepo() {
  try {
    const { data: repoData } = await octokit.repos.get({ owner, repo });
    
    // Fetch last 50 commits
    const { data: commits } = await octokit.repos.listCommits({
      owner,
      repo,
      per_page: 50
    });

    const commitMetrics = commits.map((commit, index) => {
      const currentCommitDate = new Date(commit.commit.author.date);
      let timeDiff = 0;
      
      if (index < commits.length - 1) {
        const prevCommitDate = new Date(commits[index + 1].commit.author.date);
        timeDiff = (currentCommitDate - prevCommitDate) / (1000 * 60 * 60);
      }
      return {
        sha: commit.sha,
        author: commit.commit?.author?.name || 'Unknown',
        email: commit.commit?.author?.email || 'Unknown',
        date: currentCommitDate.toISOString(),
        message: commit.commit?.message || '',
        time_since_last_commit_hours: timeDiff.toFixed(2),
        files_modified_count: commit.files ? commit.files.length : 0
      };
    });

    let contributors = [];
    try {
        const { data: contribs } = await octokit.repos.listContributors({ owner, repo, per_page: 50 });
        contributors = contribs.map(c => ({ login: c.login, contributions: c.contributions }));
    } catch (e) {
        // Just ignore contributor errors if they happen (e.g. rate limit specifically on contributors)
    }

    const payload = {
      repository: {
        id: repoData.id,
        name: repoData.name,
        full_name: repoData.full_name,
        owner: repoData.owner.login,
        stars: repoData.stargazers_count,
        forks: repoData.forks_count,
        language: repoData.language,
        created_at: repoData.created_at,
        pushed_at: repoData.pushed_at,
        description: repoData.description
      },
      commit_history: commitMetrics,
      contributors: contributors,
      scraped_at: new Date().toISOString()
    };

    const outDir = path.join(__dirname, 'raw_data');
    if (!fs.existsSync(outDir)) {
      fs.mkdirSync(outDir, { recursive: true });
    }
    
    // Create safely formatted filename
    const safeName = `${owner}_${repo}.json`.replace(/[\/\\?%*:|"<>]/g, '-');
    const outFile = path.join(outDir, safeName);
    fs.writeFileSync(outFile, JSON.stringify(payload, null, 2));
    
    // Output valid JSON to be parsed by Python
    console.log(JSON.stringify({ success: true, file: safeName }));
  } catch (err) {
    if (err.status === 404) {
      console.error(JSON.stringify({ error: `Repository ${owner}/${repo} not found.` }));
    } else {
      console.error(JSON.stringify({ error: err.message }));
    }
    process.exit(1);
  }
}

fetchRepo();
