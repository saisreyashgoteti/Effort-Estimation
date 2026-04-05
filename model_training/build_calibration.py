"""
build_calibration.py
─────────────────────
Trains a calibration regressor that maps raw model outputs → realistic
person-month values, using benchmark data as ground truth.

Run once after training the improved model:
    python3 model_training/build_calibration.py

Outputs: artifacts/calibration_model.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

ARTIFACTS = 'artifacts'
MODEL_PKL  = os.path.join(ARTIFACTS, 'final_improved_effort_model.pkl')
BENCH_CSV  = 'external_data/manual_datasets/research_benchmarks.csv'
CALIB_PKL  = os.path.join(ARTIFACTS, 'calibration_model.pkl')


def build_pairs():
    """Generate (raw_prediction, actual_effort) pairs from benchmark data."""
    with open(MODEL_PKL, 'rb') as f:
        pkg = pickle.load(f)
    pipeline      = pkg['pipeline']
    feature_names = pkg['feature_names']

    df = pd.read_csv(BENCH_CSV)
    y_actual = df['Effort'].values  # ground truth (person-months)

    rows = []
    for _, row in df.iterrows():
        kloc = float(row['KLOC'])
        dur  = max(float(row['Duration']), 1.0)
        rec  = {
            'total_commits':    kloc * 10,
            'issues_count':     kloc * 0.5,
            'pull_requests':    kloc * 0.8,
            'size':             kloc,          # KLOC matches training 'size' scale
            'team_size':        float(row['TeamSize']),
            'commit_frequency': kloc / dur,
            'project_scale':    np.log1p(kloc),
        }
        rows.append(rec)

    X_bench = pd.DataFrame(rows)
    for f in feature_names:
        if f not in X_bench.columns:
            X_bench[f] = 0
    X_bench = X_bench[feature_names].values

    y_raw = pipeline.predict(X_bench)
    return y_raw, y_actual


def train_calibration(y_raw, y_actual):
    """
    Fit a linear Ridge regression:  actual ≈ a * raw + b

    Linear is intentional: polynomial calibrators produce negative values
    when extrapolating below the training range of raw predictions.
    An origin-anchored linear fit is stable and monotone across all inputs.
    """
    # Anchor at origin: if raw=0 then calibrated=0
    # fit_intercept=False forces the line through the origin, preventing
    # the large negative intercept that clips small predictions to the floor.
    X_aug = np.concatenate([[0], y_raw]).reshape(-1, 1)
    y_aug = np.concatenate([[0], y_actual])

    calib_model = Ridge(alpha=0.1, fit_intercept=False)
    calib_model.fit(X_aug, y_aug)

    y_cal = calib_model.predict(y_raw.reshape(-1, 1))
    mae = mean_absolute_error(y_actual, y_cal)
    r2  = r2_score(y_actual, y_cal)

    print("Calibration training results:")
    for raw, actual, cal in zip(y_raw, y_actual, y_cal):
        print(f"  raw={raw:.1f}  actual={actual:.0f}  calibrated={cal:.1f}")
    print(f"\nCalibration slope : {calib_model.coef_[0]:.4f}")
    print(f"Calibration intercept: {calib_model.intercept_:.4f}")
    print(f"Calibration MAE : {mae:.2f} PM")
    print(f"Calibration R²  : {r2:.4f}")

    return calib_model


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)

    print("Building calibration pairs from benchmark data …")
    y_raw, y_actual = build_pairs()

    print("Training calibration model …")
    calib = train_calibration(y_raw, y_actual)

    # Also store the raw→actual range for clipping in the API
    meta = {
        'model':   calib,
        'raw_min': float(y_raw.min()),
        'raw_max': float(y_raw.max()),
        'cal_min': float(y_actual.min()),
        'cal_max': float(y_actual.max()),
    }
    with open(CALIB_PKL, 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nCalibration model saved → {CALIB_PKL}")


if __name__ == '__main__':
    main()
