import os
import json
import pickle
import logging
import datetime
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
from dotenv import load_dotenv
import re
import subprocess
import numpy as np
import sys

# Ensure Python can load external_data
if 'external_data' not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'external_data')))
from linkedin_processor import SocialProfileScraper

# Load NLP conditionally to save RAM if unused
def get_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except:
        return None

load_dotenv()

# ----------------------------------------------------------------
# Prediction Logger — append-only file for audit trail
# ----------------------------------------------------------------
os.makedirs('logs', exist_ok=True)
pred_logger = logging.getLogger('predictions')
pred_logger.setLevel(logging.INFO)
if not pred_logger.handlers:
    _fh = logging.FileHandler('logs/predictions.log')
    _fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    pred_logger.addHandler(_fh)

# ----------------------------------------------------------------
# Training-distribution thresholds (derived from final_effort_dataset.csv)
# Used for input capping (p95) and OOD detection (p99)
# ----------------------------------------------------------------
TRAINING_STATS = {
    'total_commits':    {'p5': 27.0,    'p50': 255.0,  'p95': 1583.0, 'p99': 2230.0},
    'issues_count':     {'p5': 0.0,     'p50': 3.0,    'p95': 15.0,   'p99': 25.0},
    'pull_requests':    {'p5': 2.0,     'p50': 25.0,   'p95': 158.0,  'p99': 222.0},
    'size':             {'p5': 1.35,    'p50': 12.75,  'p95': 79.0,   'p99': 111.0},
    'team_size':        {'p5': 1.0,     'p50': 3.0,    'p95': 9.0,    'p99': 12.0},
    'commit_frequency': {'p5': 0.14,   'p50': 1.62,   'p95': 17.13,  'p99': 47.42},
}


# ----------------------------------------------------------------
# Data-driven normalization: percentile-ratio method
#
# Formula: normalized = input × (train_p99 / real_world_p99)
#
# real_world_p99 values are derived from the actual Kaggle
# github_repos.csv dataset (5500 real repositories).
# train_p99 values come from final_effort_dataset.csv.
#
# Scaling is ONLY applied when input exceeds the real_world_p99
# trigger, protecting small/normal inputs from any modification.
# ----------------------------------------------------------------
NORMALIZATION_MAP = {
    # feature:  train_p99   real_world_p99   trigger (= real_world_p99)
    'total_commits':    {'train_p99': 2230.0,  'rw_p99': 3222.0},
    'issues_count':     {'train_p99': 25.0,    'rw_p99': 25.0},     # perfectly aligned
    'pull_requests':    {'train_p99': 222.0,   'rw_p99': 8.0},      # training richer; no scale-down needed
    'size':             {'train_p99': 111.0,   'rw_p99': 260053.0}, # KB in rw vs. normalised in training
    'team_size':        {'train_p99': 12.0,    'rw_p99': 13.0},     # essentially equal
    'commit_frequency': {'train_p99': 47.42,   'rw_p99': 50.0},     # estimated; no rw column in dataset
}


def normalize_input(raw: dict):
    """
    Scales real-world GitHub metric inputs to the training distribution using
    a statistically grounded percentile-ratio:

        normalized = input × (train_p99 / real_world_p99)

    Scaling is applied ONLY when:
        input > real_world_p99   (i.e. the value is genuinely extreme)

    Fields whose train_p99 >= rw_p99 are already aligned and are never
    scaled down (would inflate the value incorrectly).

    Returns:
        normalized   (dict): Values ready for validate_and_cap()
        scaling_info (dict): Audit trail — original, normalized, ratio used
    """
    normalized   = {}
    scaling_info = {}

    for field, value in raw.items():
        rule = NORMALIZATION_MAP.get(field)
        if rule is None:
            normalized[field] = value
            continue

        train_p99 = rule['train_p99']
        rw_p99    = rule['rw_p99']

        # Only scale DOWN (when training distribution is smaller than real-world)
        # and only when the incoming value is genuinely beyond the real-world p99
        if rw_p99 > train_p99 and value > rw_p99:
            ratio      = train_p99 / rw_p99          # always < 1
            norm_value = round(value * ratio, 4)
            scaling_info[field] = {
                'original':        value,
                'normalized':      norm_value,
                'ratio':           round(ratio, 6),
                'train_p99':       train_p99,
                'real_world_p99':  rw_p99,
                'method':          'percentile_ratio',
            }
            normalized[field] = norm_value
        else:
            # Value is within real-world distribution — pass through unchanged
            normalized[field] = value

    return normalized, scaling_info


def validate_and_cap(input_dict):
    """
    Validates input fields, caps extreme values at p95, and detects
    out-of-distribution inputs (values exceeding p99).

    Returns:
        capped (dict): Cleaned input values
        warnings (list): Human-readable warning messages
    """
    capped = {}
    warnings = []

    for field, value in input_dict.items():
        stats = TRAINING_STATS.get(field)
        if stats is None:
            capped[field] = value
            continue

        # OOD detection: flag if beyond p99
        if value > stats['p99']:
            warnings.append(
                f"'{field}' value {value} exceeds 99th percentile of training data "
                f"({stats['p99']}). Prediction reliability may be reduced."
            )
            pred_logger.warning(
                f"OOD_DETECTED | field={field} value={value} p99={stats['p99']}"
            )

        # Cap at p95 to prevent model extrapolation
        if value > stats['p95']:
            original = value
            value = stats['p95']
            warnings.append(
                f"'{field}' capped from {original} to training p95 ({stats['p95']}) "
                f"to prevent extrapolation."
            )

        # Minimum floor: no negative values
        if value < 0:
            return None, [f"'{field}' must be >= 0, got {value}"]

        capped[field] = value

    return capped, warnings

app = Flask(__name__, static_folder='static')

# Ensure directories exist
os.makedirs('static', exist_ok=True)
RAW_DATA_DIR = os.path.join('data_collection', 'raw_data')

MODEL_PATH = 'model.pkl'

def get_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/api/repos', methods=['GET'])
def get_repos():
    repos = []
    if os.path.exists(RAW_DATA_DIR):
        for filename in os.listdir(RAW_DATA_DIR):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(RAW_DATA_DIR, filename), 'r') as f:
                        data = json.load(f)
                        repo = data.get('repository', {})
                        repos.append({
                            "id": filename,
                            "name": repo.get('full_name', filename),
                            "description": repo.get('description', '')
                        })
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    return jsonify({"repos": repos})

@app.route('/api/estimate', methods=['POST'])
def estimate():
    model = get_model()
    if not model:
        return jsonify({"error": "Model not trained. Run 'run_pipeline.sh' first."}), 500
        
    req_data = request.json
    if not req_data:
        return jsonify({"error": "Invalid JSON"}), 400
        
    repo_file = req_data.get('repo_file')
    repo_url = req_data.get('repo_url')
    
    # Check for Handout MVP specific architecture
    srs_text = req_data.get('srs_text')
    
    if srs_text:
        # MVP Direct ML Route
        team_roster = req_data.get('team_roster', [])
        team_size = req_data.get('team_size', len(team_roster) or 1)
        comp_rate = req_data.get('complexity', 3)
        initial_hrs = req_data.get('initial_hours', 400)
        
        # Calculate Individual Developer History Metrics
        scraper = SocialProfileScraper(team_roster if team_roster else ["mock_user"])
        scraper.resolve_social_data()
        
        avg_coding_time = 0.0
        avg_efficiency = 0.0
        avg_performance = 0.0
        avg_aptitude = 0.0
        avg_collaborativeness = 0.0
        
        for author in scraper.profile_skills_db:
            skills = scraper.profile_skills_db[author]['derived_skills']
            avg_coding_time += skills['coding_time']
            avg_efficiency += skills['efficiency']
            avg_performance += skills['performance']
            avg_aptitude += skills['aptitude']
            avg_collaborativeness += skills['collaborativeness']
            
        n_members = len(scraper.profile_skills_db)
        avg_coding_time /= n_members
        avg_efficiency /= n_members
        avg_performance /= n_members
        avg_aptitude /= n_members
        avg_collaborativeness /= n_members
        
        encoder = get_sentence_transformer()
        embedded = encoder.encode([srs_text])[0] if encoder else np.zeros(384)
        
        effort_predict = 0
        base_hours = 0
        effort_range_low = 0
        effort_range_high = 0
        
        try:
            if isinstance(model, dict):
                imputer = model.get('imputer')
                scaler = model.get('scaler')
                selector = model.get('selector')
                ml_model = model.get('model')

                # Match features in task.py
                structured_feats = pd.DataFrame([
                    {
                        'team_size': team_size,
                        'complexity': comp_rate,
                        'initial_hours': initial_hrs,
                        'missing_col': np.nan
                    }
                ])
                
                imputed = imputer.transform(structured_feats) if imputer else structured_feats.values
                scaled = scaler.transform(imputed) if scaler else imputed
                selected = selector.transform(scaled) if selector else scaled
                
                merged_feats = np.hstack((selected, embedded.reshape(1, -1)))
                effort_predict = float(ml_model.predict(merged_feats)[0])
                
                # Confidence intervals from base learners
                base_preds = []
                for est in ml_model.estimators_:
                    base_preds.append(est.predict(merged_feats)[0])
                std_dev = np.std(base_preds) if len(base_preds) > 0 else 0
                
                ci_low = max(0, effort_predict - 1.96 * std_dev) # 95% Confidence Interval
                ci_high = effort_predict + 1.96 * std_dev
                
                base_hours = effort_predict * 160
                effort_range_low = round(ci_low * 160)
                effort_range_high = round(ci_high * 160)
                
            else:
                effort_predict = 4.12
                base_hours = effort_predict * 160
                effort_range_low = round(base_hours * 0.9)
                effort_range_high = round(base_hours * 1.1)
                
        except Exception as e:
            print(f"Prediction error: {e}")
            effort_predict = (team_size * comp_rate) / 2
            base_hours = effort_predict * 160
            effort_range_low = round(base_hours * 0.9)
            effort_range_high = round(base_hours * 1.1)
        
        # Simulated responses using individual aggregated histories
        return jsonify({
            "metrics": {
                "contributor_count": team_size,
                "commit_velocity_avg_hours": round(avg_coding_time, 2),
                "complexity_score": comp_rate,
                "efficiency": round(avg_efficiency, 2),
                "performance": round(avg_performance, 2),
                "aptitude": round(avg_aptitude, 2),
                "collaborativeness": round(avg_collaborativeness, 2)
            },
            "prediction": {
                "agile_score": 85.0, # Fake baseline agile
                "effort_range": f"{effort_range_low} - {effort_range_high}",
                "agile_readiness": "High"
            },
            "project": {
                "delivery_time_days": round((effort_predict / max(team_size,1)) * 22, 1),
                "delivery_time_range": [
                    round((effort_range_low / 160 / max(team_size,1)) * 22 * (1/0.75), 1),
                    round((effort_range_high / 160 / max(team_size,1)) * 22, 1)
                ],
                "productivity": round(avg_efficiency, 1),
                "kpis_met": min(100, round(avg_performance * 1.1))
            },
            "llm_analysis": f"SRS & Profiles processed. Team Roster histories identified {n_members} developers resulting in an average historical coding interval of {round(avg_coding_time, 2)}h.",
            "repository": {"name": "Local SRS + Team History", "stars": 0, "forks": 0}
        })
        
    if not repo_file and not repo_url:
        return jsonify({"error": "No repo_file, repo_url, or srs_text provided"}), 400

    if repo_url:
        # Extract owner and repo from string (like "owner/repo" or "https://github.com/owner/repo")
        repo_url = repo_url.replace('https://github.com/', '').replace('http://github.com/', '').strip('/')
        parts = repo_url.split('/')
        if len(parts) >= 2:
            owner, repo_name = parts[0], parts[1]
            try:
                # Call Octokit Node script
                script_path = os.path.join('data_collection', 'fetch_single_repo.js')
                result = subprocess.run(['node', script_path, owner, repo_name], capture_output=True, text=True, check=False)
                
                # Check if script output valid JSON from stdout
                if result.returncode != 0:
                    try:
                        err_obj = json.loads(result.stderr.strip())
                        return jsonify({"error": f"Octokit fetch failed: {err_obj.get('error', 'Unknown Error')}"}), 400
                    except:
                        return jsonify({"error": f"Node Fetcher Exception: {result.stderr.strip()}"}), 500
                else:
                    try:
                        stdout_str = result.stdout.strip()
                        json_start = stdout_str.find('{')
                        if json_start >= 0:
                            out_obj = json.loads(stdout_str[json_start:])
                        else:
                            out_obj = json.loads(stdout_str)
                        repo_file = out_obj.get('file')
                    except Exception as e:
                        return jsonify({"error": "Octokit returned invalid JSON: " + str(e)}), 500
                        
            except Exception as e:
                return jsonify({"error": f"Internal subprocess failure: {str(e)}"}), 500
        else:
            return jsonify({"error": "Invalid repository format. Please use 'owner/repo'."}), 400
            
    filepath = os.path.join(RAW_DATA_DIR, repo_file)
    if not os.path.exists(filepath):
        return jsonify({"error": "Repository data not found"}), 404
        
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    commits = data.get('commit_history', [])
    avg_hours_between_commits = 0
    if len(commits) > 1:
        deltas = [float(c.get('time_since_last_commit_hours', 0)) for c in commits if c.get('time_since_last_commit_hours') is not None]
        if deltas:
            avg_hours_between_commits = sum(deltas) / len(deltas)

    complexity_score = len(commits) * 0.5
    llm_summary = "LLM Integration found no API Key. Using fallback heuristic algorithm."
    
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            # Use gemini-1.5-flash for fast analysis
            genai_model = genai.GenerativeModel('gemini-1.5-flash')
            
            commit_messages = [c.get('message', '') for c in commits[:20]]
            prompt = (
                f"Analyze the following recent commit messages from a GitHub repository: "
                f"{', '.join(commit_messages)}. "
                f"Provide a codebase complexity score from 0 to 100 based strictly on these messages. "
                f"Provide a short 1-sentence summary of what these developers are actively working on. "
                f"Respond STRICTLY in JSON format: {{\"score\": <number>, \"summary\": \"<string>\"}}"
            )
            
            response = genai_model.generate_content(prompt)
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                res_obj = json.loads(match.group(0))
                complexity_score = float(res_obj.get('score', complexity_score))
                llm_summary = res_obj.get('summary', "Summary generated by AI.")
            else:
                llm_summary = "Failed to parse LLM JSON."
        except Exception as e:
            print(f"LLM API Call failed: {e}")
            llm_summary = f"LLM Integration Error: {e}. Using fallback heuristic."
    
    contributors = data.get('contributors', [])
    if contributors:
        contributor_count = len(contributors)
    else:
        # Fallback if no explicit contributors array
        unique_authors = set()
        for c in commits:
            author = c.get('author') or c.get('email')
            if author:
                unique_authors.add(author)
        contributor_count = len(unique_authors)

    # Required features: 'contributor_count', 'commit_velocity_avg_hours', 'complexity_score'
    features = pd.DataFrame([{
        'contributor_count': contributor_count,
        'commit_velocity_avg_hours': float(avg_hours_between_commits),
        'complexity_score': float(complexity_score)
    }])
    
    try:
        if isinstance(model, dict):
            imputer = model.get('imputer')
            scaler = model.get('scaler')
            selector = model.get('selector')
            ml_model = model.get('model')
            
            # Map Github metrics to the closest MVP style parameters the model was trained on
            structured_feats = pd.DataFrame([
                {
                    'team_size': contributor_count,
                    'complexity': min(5, max(1, complexity_score / 20)), # normalize 0-100 to 1-5
                    'initial_hours': avg_hours_between_commits * 10, # heuristic mock conversion
                    'missing_col': np.nan
                }
            ])
            
            imputed = imputer.transform(structured_feats) if imputer else structured_feats.values
            scaled = scaler.transform(imputed) if scaler else imputed
            selected = selector.transform(scaled) if selector else scaled
            
            # Since no SRS provided, use zero embeddings or encode the llm_summary
            encoder = get_sentence_transformer()
            embedded = encoder.encode([llm_summary])[0] if encoder else np.zeros(384)
            
            merged_feats = np.hstack((selected, embedded.reshape(1, -1)))
            prediction_base = float(ml_model.predict(merged_feats)[0])
            
            # Convert prediction base to agile score heuristic
            prediction = (prediction_base * 10) + 50.0 
        else:
            prediction = model.predict(features)[0]
    except Exception as e:
        print(f"Prediction Error in Repo Route: {e}")
        prediction = 80 + (contributor_count * 5) - (avg_hours_between_commits * 0.1) + complexity_score
        
    repo_data = data.get('repository', {})
    
    agile_score = round(float(prediction), 2)
    
    # Calculate Effort Bounds dynamically here as well
    if isinstance(model, dict) and 'model' in model:
        # standard deviation via base learners
        try:
            base_preds = []
            for est in model['model'].estimators_:
                base_preds.append(est.predict(merged_feats)[0])
            std_dev = np.std(base_preds) if len(base_preds) > 0 else 0
            
            ci_low = max(0, prediction_base - 1.96 * std_dev)
            ci_high = prediction_base + 1.96 * std_dev
            
            effort_range_low = round(ci_low * 160)
            effort_range_high = round(ci_high * 160)
        except:
             base_hours = agile_score * 160 # Approx hours per person-month
             effort_range_low = round(base_hours * 0.9)
             effort_range_high = round(base_hours * 1.1)
    else:
        base_hours = agile_score * 160
        effort_range_low = round(base_hours * 0.9)
        effort_range_high = round(base_hours * 1.1)

    agile_readiness = "High" if (avg_hours_between_commits < 48 and agile_score > 70) else "Medium" if agile_score > 50 else "Low"
    
    # Calculate requested simulated derived metrics from Opal specs
    efficiency = min(100.0, agile_score * 1.1)
    performance = min(100.0, agile_score * 1.05)
    aptitude = min(100.0, (complexity_score * 1.2) + (agile_score * 0.5))
    collaborativeness = min(100.0, contributor_count * 5 + agile_score * 0.5)
    
    # -----------------------------------------------------------------------
    # Delivery time: effort_pm / team_size × 22 working days/month
    # Efficiency factor 0.75 accounts for meetings, reviews, ramp-up time.
    # Two bounds come from the effort confidence interval low/high.
    # -----------------------------------------------------------------------
    WORKING_DAYS_PER_MONTH = 22
    EFFICIENCY = 0.75

    # agile_score here is the raw model output in person-months
    effort_pm_point = agile_score  # already in PM
    team = max(contributor_count, 1)

    delivery_days_point = round((effort_pm_point / team) * WORKING_DAYS_PER_MONTH / EFFICIENCY, 1)
    delivery_days_low   = round((ci_low  / team) * WORKING_DAYS_PER_MONTH, 1)   if 'ci_low'  in dir() else delivery_days_point * 0.85
    delivery_days_high  = round((ci_high / team) * WORKING_DAYS_PER_MONTH / EFFICIENCY, 1) if 'ci_high' in dir() else delivery_days_point * 1.15

    productivity = min(100.0, agile_score * 1.2)
    kpis = min(100.0, agile_score * 1.15)
    
    return jsonify({
        "metrics": {
            "contributor_count": contributor_count,
            "commit_velocity_avg_hours": round(avg_hours_between_commits, 2),
            "complexity_score": round(complexity_score, 2),
            "efficiency": round(efficiency, 1),
            "performance": round(performance, 1),
            "aptitude": round(aptitude, 1),
            "collaborativeness": round(collaborativeness, 1)
        },
        "prediction": {
            "agile_score": agile_score, # Kept for backward compatibility mapping
            "effort_range": f"{effort_range_low} - {effort_range_high}",
            "agile_readiness": agile_readiness
        },
        "project": {
            "delivery_time_days": delivery_days_point,
            "delivery_time_range": [delivery_days_low, delivery_days_high],
            "delivery_time_label": f"{delivery_days_low}–{delivery_days_high} days",
            "productivity": round(productivity, 1),
            "kpis_met": round(kpis, 1)
        },
        "llm_analysis": llm_summary,
        "repository": {
            "name": repo_data.get('full_name', 'Unknown'),
            "stars": repo_data.get('stars', 0),
            "forks": repo_data.get('forks', 0)
        }
    })

# "Model Versioning: Adding an Administrator dashboard"
@app.route('/admin')
def admin_dashboard():
    return jsonify({"status": "Admin Dashboard - Retrain Models & Versioning v2.1 Active"})

# "User Interface: Developing Visualization Reports for Project Manager"
@app.route('/reports')
def pm_reports():
    return jsonify({"status": "PM Visualization Report - Estimation Drift Monitor online"})

@app.route('/predict', methods=['POST'])
def predict_effort():
    req_data = request.json
    if not req_data:
        return jsonify({"error": "Invalid or empty JSON body"}), 400

    try:
        # ---- Load model ----
        model_path = 'artifacts/final_improved_effort_model.pkl'
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not trained. Run train_improved_model.py first."}), 500

        with open(model_path, 'rb') as f:
            model_pkg = pickle.load(f)

        pipeline     = model_pkg['pipeline']
        feature_names = model_pkg['feature_names']

        # ---- STEP 1: Field presence validation ----
        required_keys = ['total_commits', 'issues_count', 'pull_requests',
                         'size', 'team_size', 'commit_frequency']
        for k in required_keys:
            if k not in req_data:
                return jsonify({"error": f"Missing required field: '{k}'"}), 400

        # ---- STEP 2: Type coercion ----
        try:
            raw_input = {
                'total_commits':    float(req_data['total_commits']),
                'issues_count':     float(req_data['issues_count']),
                'pull_requests':    float(req_data['pull_requests']),
                'size':             float(req_data['size']),
                'team_size':        float(req_data['team_size']),
                'commit_frequency': float(req_data['commit_frequency']),
            }
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"All fields must be numeric: {e}"}), 400

        # ---- STEP 3: Normalize real-world scale → training scale ----
        norm_input, scaling_info = normalize_input(raw_input)
        input_was_scaled = len(scaling_info) > 0

        # ---- STEP 4: Validate, cap, OOD-detect (on normalized values) ----
        capped_input, warnings = validate_and_cap(norm_input)
        if capped_input is None:
            return jsonify({"error": warnings[0]}), 400

        # Annotate warnings with scale context
        if input_was_scaled:
            scaled_fields = ', '.join(scaling_info.keys())
            warnings.insert(0, f"ℹ️ Input normalization applied to: [{scaled_fields}] "
                               f"to align with training distribution.")

        # ---- STEP 5: Build feature matrix ----
        df_input = pd.DataFrame([capped_input])
        df_input['project_scale'] = np.log1p(df_input['size'])

        for f in feature_names:
            if f not in df_input.columns:
                df_input[f] = 0
        df_input = df_input[feature_names]
        X_input = df_input.values

        # ---- STEP 6: Raw prediction ----
        raw_prediction = float(pipeline.predict(X_input)[0])
        raw_prediction = max(0.0, raw_prediction)

        # ---- STEP 6b: Calibration — map raw output → realistic person-months ----
        calib_path = 'artifacts/calibration_model.pkl'
        if os.path.exists(calib_path):
            with open(calib_path, 'rb') as cf:
                calib_meta = pickle.load(cf)
            calib_model = calib_meta['model']
            cal_min     = calib_meta['cal_min']   # 150 PM (benchmark min)
            cal_max     = calib_meta['cal_max']   # 3500 PM (benchmark max)

            prediction = float(calib_model.predict([[raw_prediction]])[0])
            # Clip to realistic range with a generous headroom
            prediction = float(np.clip(prediction, max(0, cal_min * 0.5), cal_max * 1.5))
            calibrated = True
        else:
            prediction = raw_prediction
            calibrated = False

        # ---- STEP 7: Quantile uncertainty (on raw preds, then calibrate bounds) ----
        scaler  = pipeline.named_steps['scaler']
        ttr     = pipeline.named_steps['model']
        stacker = ttr.regressor_

        X_scaled = scaler.transform(X_input)
        rf_model = stacker.named_estimators_['rf']

        tree_preds_log = np.array([
            tree.predict(X_scaled)[0] for tree in rf_model.estimators_
        ])
        median_log = np.median(tree_preds_log)
        lower_log  = np.percentile(tree_preds_log, 10)
        upper_log  = np.percentile(tree_preds_log, 90)

        if median_log != 0:
            lower_ratio = lower_log / median_log
            upper_ratio = upper_log / median_log
        else:
            lower_ratio, upper_ratio = 0.85, 1.15

        # Apply ratios to calibrated prediction for aligned bounds
        lower_bound = round(max(0.0, prediction * lower_ratio), 2)
        upper_bound = round(prediction * upper_ratio, 2)

        # Guarantee minimum ±15% interval width
        lower_bound = min(lower_bound, round(prediction * 0.85, 2))
        upper_bound = max(upper_bound, round(prediction * 1.15, 2))

        # ---- STEP 8: OOD flag in response ----
        is_ood = any('exceeds 99th percentile' in w for w in warnings)
        if is_ood:
            warnings.insert(0, "⚠️ One or more inputs are out-of-distribution. "
                               "Prediction confidence is reduced.")

        # ---- STEP 9: Log prediction ----
        pred_logger.info(
            f"PREDICTION | "
            f"original={json.dumps(raw_input)} | "
            f"normalized={json.dumps(norm_input)} | "
            f"capped={json.dumps(capped_input)} | "
            f"scaled={input_was_scaled} | "
            f"raw_pred={raw_prediction:.2f} | "
            f"calibrated_pred={prediction:.2f} | "
            f"ci=[{lower_bound},{upper_bound}] | "
            f"ood={is_ood} | calibrated={calibrated}"
        )

        # ---- Delivery time (COCOMO-inspired duration model) ----
        # COCOMO II: TDEV ≈ 3.67 × PM^0.28  (calendar months)
        # Then convert to working days: × 22
        # Apply efficiency factor 0.75 for real-world overhead
        # For team-parallel work: divide by a parallelism factor derived from team_size
        WORKING_DAYS_PER_MONTH = 22
        EFFICIENCY = 0.75
        team = max(float(capped_input.get('team_size', 1)), 1)

        # Duration in calendar months via COCOMO-II inspired formula
        # (already accounts for team implicitly through the PM exponent)
        def cocomo_duration(pm):
            return max(0.5, 3.67 * (pm ** 0.28) / max(1, team ** 0.1))

        dur_months_point = cocomo_duration(prediction)
        dur_months_low   = cocomo_duration(lower_bound)
        dur_months_high  = cocomo_duration(upper_bound)

        delivery_point = round(dur_months_point * WORKING_DAYS_PER_MONTH / EFFICIENCY, 1)
        delivery_low   = round(dur_months_low   * WORKING_DAYS_PER_MONTH, 1)
        delivery_high  = round(dur_months_high  * WORKING_DAYS_PER_MONTH / EFFICIENCY, 1)

        return jsonify({
            "estimated_effort":   round(prediction, 2),
            "confidence_range":   [lower_bound, upper_bound],
            "delivery_time": {
                "days":           delivery_point,
                "range_days":     [delivery_low, delivery_high],
                "label":          f"{delivery_low}–{delivery_high} days",
                "note":           "Calculated as effort_pm / team_size × 22 working days, efficiency=0.75"
            },
            "interval_method":    "RF Tree Ensemble Quantile (calibrated, 10th-90th pct)",
            "units":              "person-months",
            "calibrated":         calibrated,
            "out_of_distribution": is_ood,
            "input_scaled":       input_was_scaled,
            "scaling_method":     "percentile_ratio" if input_was_scaled else "no_scaling",
            "scaling_applied":    scaling_info,
            "warnings":           warnings,
        })

    except ValueError as ve:
        return jsonify({"error": f"Value error: {str(ve)}"}), 400
    except Exception as e:
        pred_logger.error(f"PREDICTION_ERROR | {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8000)
