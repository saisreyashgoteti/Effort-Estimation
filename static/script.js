document.addEventListener('DOMContentLoaded', () => {
    const repoInput = document.getElementById('repo-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const loader = analyzeBtn.querySelector('.loader');
    const resultsPanel = document.getElementById('results-panel');
    
    // UI Elements for Data
    const elRepoName = document.getElementById('res-repo-name');
    const elStars = document.getElementById('res-stars');
    const elForks = document.getElementById('res-forks');
    const elContributors = document.getElementById('metric-contributors');
    const elVelocity = document.getElementById('metric-velocity');
    const elEfficiency = document.getElementById('metric-efficiency');
    const elPerformance = document.getElementById('metric-performance');
    const elAptitude = document.getElementById('metric-aptitude');
    const elCollaborativeness = document.getElementById('metric-collaborativeness');
    const elDelivery = document.getElementById('metric-delivery');
    const elKpis = document.getElementById('metric-kpis');
    const elComplexity = document.getElementById('metric-complexity');
    const elProductivity = document.getElementById('metric-productivity');
    const elLlmSummary = document.getElementById('llm-summary-text');
    
    const elReadinessDot = document.getElementById('readiness-dot');
    const elReadiness = document.getElementById('res-readiness');
    const elEffortRange = document.getElementById('effort-range-text');

    const srsInput = document.getElementById('srs-input');
    const teamRoster = document.getElementById('team-roster');
    const srsComplexity = document.getElementById('srs-complexity');
    const initialHours = document.getElementById('initial-hours');

    function checkInputs() {
        // Evaluate if GitHub URL path
        if (repoInput.value.trim().length > 0) {
            analyzeBtn.disabled = false;
        } 
        // Evaluate if SRS MVP path
        else if (srsInput.value.trim().length > 0 && teamRoster.value.trim().length > 0 && srsComplexity.value > 0 && initialHours.value > 0) {
            analyzeBtn.disabled = false;
        } else {
            analyzeBtn.disabled = true;
        }
    }

    repoInput.addEventListener('input', () => {
        // Clear MVP inputs if using GitHub
        srsInput.value = '';
        teamRoster.value = '';
        srsComplexity.value = '';
        initialHours.value = '';
        checkInputs();
    });

    [srsInput, teamRoster, srsComplexity, initialHours].forEach(el => {
        el.addEventListener('input', () => {
            // Clear github if using MVP 
            repoInput.value = '';
            checkInputs();
        });
    });

    analyzeBtn.addEventListener('click', () => {
        const payload = {};
        if (repoInput.value.trim()) {
            payload.repo_url = repoInput.value.trim();
        } else if (srsInput.value.trim()) {
            payload.srs_text = srsInput.value.trim();
            // Assign team roster by splitting raw string
            payload.team_roster = teamRoster.value.split(',').map(s => s.trim()).filter(s => s.length > 0);
            payload.team_size = payload.team_roster.length || 1;
            payload.complexity = parseInt(srsComplexity.value);
            payload.initial_hours = parseInt(initialHours.value);
        } else {
            return;
        }

        // UI Loading State
        analyzeBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        resultsPanel.classList.add('hidden'); // Reset panel

        // Simulate a slight delay to make the UX feel like heavy compute
        setTimeout(() => {
            fetch('/api/estimate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                
                populateDashboard(data);
                
                // Show dashboard 
                resultsPanel.classList.remove('hidden');
            })
            .catch(err => {
                alert("Error during analysis: " + err.message);
            })
            .finally(() => {
                analyzeBtn.disabled = false;
                btnText.classList.remove('hidden');
                loader.classList.add('hidden');
            });
        }, 800);
    });

    function populateDashboard(data) {
        // Meta
        elRepoName.textContent = data.repository.name;
        elStars.textContent = data.repository.stars.toLocaleString();
        elForks.textContent = data.repository.forks.toLocaleString();
        elLlmSummary.textContent = data.llm_analysis;

        // Animate metrics counter
        animateCounter(elContributors, 0, data.metrics.contributor_count, 1000);
        animateCounter(elVelocity, 0, data.metrics.commit_velocity_avg_hours, 1000, 1);
        animateCounter(elEfficiency, 0, data.metrics.efficiency, 1000, 1);
        animateCounter(elPerformance, 0, data.metrics.performance, 1000, 1);
        animateCounter(elAptitude, 0, data.metrics.aptitude, 1000, 1);
        animateCounter(elCollaborativeness, 0, data.metrics.collaborativeness, 1000, 1);
        animateCounter(elComplexity, 0, data.metrics.complexity_score, 1000, 1);
        
        // Delivery time: show range label (e.g. "44–73 days") from /predict or legacy project block
        if (elDelivery) {
            const dt = data.delivery_time || data.project;
            const label = dt && dt.label
                ? dt.label                                                   // /predict: "44–73 days"
                : dt && dt.delivery_time_label
                    ? dt.delivery_time_label                                 // /api/estimate: "44–73 days"
                    : dt && dt.delivery_time_days !== undefined
                        ? dt.delivery_time_days.toFixed(1) + ' days'        // fallback single value
                        : '—';
            elDelivery.innerHTML = label;
        }
        animateCounter(elKpis, 0, data.project.kpis_met, 1000, 1);
        animateCounter(elProductivity, 0, data.project.productivity, 1000, 1);

        // Update Effort Range Text
        if (elEffortRange) {
            elEffortRange.innerText = data.prediction.effort_range || "Pending";
            
            // Add a small fade-in animation
            elEffortRange.style.opacity = 0;
            setTimeout(() => {
                elEffortRange.style.transition = 'opacity 0.6s ease-in';
                elEffortRange.style.opacity = 1;
            }, 100);
        }

        // Status 
        const readiness = data.prediction.agile_readiness;
        elReadiness.textContent = readiness;
        
        // Remove old classes
        elReadinessDot.className = 'pulse-dot';
        elReadiness.className = 'status-text';
        
        if (readiness === "High") {
            elReadinessDot.classList.add('status-high');
            elReadiness.classList.add('status-high');
        } else if (readiness === "Medium") {
            elReadinessDot.classList.add('status-medium');
            elReadiness.classList.add('status-medium');
        } else {
            elReadinessDot.classList.add('status-low');
            elReadiness.classList.add('status-low');
        }
    }

    function animateCounter(el, start, end, duration, decimals = 0) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            // ease out cubic
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = (start + (end - start) * easeOut);
            
            el.innerHTML = current.toFixed(decimals);
            
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                el.innerHTML = end.toFixed(decimals);
            }
        };
        window.requestAnimationFrame(step);
    }
});
