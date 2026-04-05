
======================================================
📊 MODEL EVALUATION SUMMARY
======================================================
1. TEST SET PERFORMANCE:
   - MAE:     1.7557 person-months
   - R²:      1.0000
   - MMRE:    0.26%
   - Pred(25): 100.00%

2. CROSS VALIDATION:
   - 5-Fold Avg MAE: 2.9230 person-months

3. DIAGNOSTICS:
   - Overfitting: Negative
   - Data Leakage: WARNING/LIKELY

INTERPRETATION:
- Pred(25) indicates that 100.00% of our predictions are within 25% of the actual real effort.
- Mean Magnitude of Relative Error (MMRE) is 0.26%.
- If R² is exactly 1.0 or MAE < 0.1, the pipeline has critically memorized target values (check DerivedFeaturesTransformer).
- A robust generalization should see Pred(25) > 30% and MMRE < 50% for software engineering tasks.
======================================================
    