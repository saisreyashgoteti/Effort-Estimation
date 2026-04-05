require('dotenv').config({ path: '../.env' });
const { Octokit } = require('@octokit/rest');
const fs = require('fs');
const path = require('path');

// "We have used Octakit to help us scrape through thousands of repositories gathering project source code."
// "Octakit is GitHub’s official API toolkit chain that helps users access repositories, GitHub’s insights into user data, contributions"

const octokit = new Octokit({
  auth: process.env.GITHUB_TOKEN || '',
});

const MAX_REPOS = 11000;
const BATCH_SIZE = 100;
const OUT_DIR = path.join(__dirname, 'raw_data');

if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

async function scrapeRepositories() {
  console.log("Starting bulk scrape of up to 11,000 repositories using Octokit...");
  let count = 0;
  let page = 1;

  while (count < MAX_REPOS) {
    try {
      const { data } = await octokit.search.repos({
        q: 'stars:>500 pushed:>2024-01-01',
        sort: 'stars',
        order: 'desc',
        per_page: BATCH_SIZE,
        page: page
      });

      if (!data.items || data.items.length === 0) break;

      for (const repo of data.items) {
        if (count >= MAX_REPOS) break;
        
        console.log(`Processing [${count + 1}/${MAX_REPOS}]: ${repo.full_name}`);
        
        // 1. Fetch File Modification Metadata (Metadata API)
        const { data: commits } = await octokit.repos.listCommits({
          owner: repo.owner.login,
          repo: repo.name,
          per_page: 50
        });

        // "identify the key difference in timestamps across commits, and using identification of file modification metadata provided by the Metadata API"
        const commitMetrics = await Promise.all(commits.map(async (commit, index) => {
          const currentDate = new Date(commit.commit.author.date);
          let timeDiffHours = 0;
          
          if (index < commits.length - 1) {
            const prevDate = new Date(commits[index + 1].commit.author.date);
            timeDiffHours = (currentDate - prevDate) / (1000 * 60 * 60);
          }

          // Fetch per-commit file modification metadata
          let filesModified = 0;
          let additions = 0;
          let deletions = 0;
          try {
            const { data: detailedCommit } = await octokit.repos.getCommit({
              owner: repo.owner.login,
              repo: repo.name,
              ref: commit.sha
            });
            filesModified = detailedCommit.files ? detailedCommit.files.length : 0;
            additions = detailedCommit.stats ? detailedCommit.stats.additions : 0;
            deletions = detailedCommit.stats ? detailedCommit.stats.deletions : 0;
          } catch(e) { /* Limit exceeded */ }

          return {
            sha: commit.sha,
            author_email: commit.commit?.author?.email,
            date: currentDate.toISOString(),
            message: commit.commit?.message,
            time_since_last_commit_hours: timeDiffHours.toFixed(2),
            metadata_api_files_modified: filesModified,
            additions: additions,
            deletions: deletions
          };
        }));

        // 2. Fetch User Data & Contributions
        let contributors = [];
        try {
          const { data: contribs } = await octokit.repos.listContributors({
            owner: repo.owner.login,
            repo: repo.name,
            per_page: 30
          });
          contributors = contribs.map(c => ({ login: c.login, contributions: c.contributions }));
        } catch(e) {}

        const payload = {
          repository: repo.full_name,
          language: repo.language,
          commit_history: commitMetrics,
          contributors: contributors
        };

        const safeName = repo.full_name.replace(/[\/\\?%*:|"<>]/g, '-');
        fs.writeFileSync(path.join(OUT_DIR, `${safeName}.json`), JSON.stringify(payload, null, 2));
        count++;
        
        // Throttle to respect rate limits
        await new Promise(r => setTimeout(r, 1000));
      }
      page++;
    } catch (err) {
      console.error("Octokit Bulk Fetch Error: ", err.message);
      await new Promise(r => setTimeout(r, 60000)); // wait 1 minute on limit Hit
    }
  }
}

scrapeRepositories();
