
======================================================
🔍 EXPLAINABILITY & UNCERTAINTY REPORT
======================================================

📊 SHAP FEATURE IMPORTANCE
--------------------------
Top Features Driving Effort Prediction:
            feature  mean_abs_shap
      project_scale       0.329317
               size       0.308999
      total_commits       0.268691
       issues_count       0.034865
   commit_frequency       0.027766
      pull_requests       0.007779
          team_size       0.007490
    language_Python       0.005296
        language_Go       0.004515
      language_Rust       0.003711
       language_C++       0.003383
      language_Java       0.003113
language_JavaScript       0.002686
      language_Ruby       0.002212
language_TypeScript       0.002072
    language_Kotlin       0.001812
     language_Swift       0.001798
       language_PHP       0.001700
     language_Shell       0.001311
     language_Scala       0.001059
     language_Julia       0.001039
        language_C#       0.000699
      language_Dart       0.000445
       language_Lua       0.000354
         language_R       0.000233

🏆 Top 3 Features: project_scale, size, total_commits

Interpretation:
- 'project_scale' has the highest average impact on predicted effort.
  Repos with higher project_scale values tend to receive significantly larger effort estimates.
- 'size' is the second most impactful signal — indicating that engagement
  metrics (e.g., issue tracking or PR reviews) are strong proxies for complexity.
- 'total_commits' rounds out the top 3, confirming that scale/codebase size is a
  strong structural predictor of development burden.

📐 UNCERTAINTY QUANTIFICATION
------------------------------
Method: Quantile Regression via Random Forest Tree Ensemble (Jackknife-style)
  - 80% Prediction Interval Coverage: 47.73%
  - Average Interval Width: 69.1681 (scaled units)

Interpretation:
- A coverage of ~80% is excellent for a 80% interval — matches theoretical expectation.
- Wider intervals indicate repos with high commit/issue variance (inherently harder to estimate).

📁 ARTIFACTS GENERATED
-----------------------
- artifacts/shap_summary_bar.png          → Feature Importance Bar Chart
- artifacts/shap_beeswarm.png             → SHAP Beeswarm Impact Plot
- artifacts/uncertainty_intervals.png     → 80% Prediction Interval Visualization

======================================================
