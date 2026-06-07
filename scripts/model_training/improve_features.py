#!/usr/bin/env python3
"""
DS4CS — Feature & Model Improvement Script
===========================================
Implements a hybrid anomaly-supervised pipeline.
1. Trains an unsupervised Isolation Forest strictly on benign training layouts.
2. Calculates anomaly scores for all training and test records.
3. Appends the anomaly score as a new feature to the training and test sets.
4. Retrains the model suite to evaluate performance improvements.
"""

import subprocess
import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR = BASE_DIR / 'models'

EXCLUDE_COLS = {'Label', 'Category', 'Technique', 'URL', 'HTML_Content', 'Source', 'Payload'}


def improve_dataset(num_benign, ratio_pct):
    train_path = DATA_DIR / f'features_train.csv'
    test_path = DATA_DIR / f'features_test.csv'

    if not train_path.exists() or not test_path.exists():
        print(f"Skipping {num_benign}_r{ratio_pct} (files not found).")
        return False

    print(f"\nProcessing Dataset: {num_benign} benign, ratio {ratio_pct}%")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Identify features
    feature_cols = [c for c in df_train.columns if c not in EXCLUDE_COLS and c != 'anomaly_score']
    X_train = df_train[feature_cols].fillna(0)
    X_test = df_test[feature_cols].fillna(0)

    # Fit Isolation Forest strictly on benign training layouts
    benign_mask = df_train['Label'] == 0
    X_train_benign = X_train[benign_mask]

    print(f"  Training Isolation Forest on {len(X_train_benign)} benign records...")
    clf = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_benign)

    # Save Isolation Forest model
    iso_model_path = MODELS_DIR / f'isolation_forest.joblib'
    joblib.dump(clf, iso_model_path)
    print(f"  Saved Isolation Forest model to {iso_model_path}")

    # decision_function returns negative values for anomalies, positive for inliers.
    # We invert it so higher score = more anomalous.
    df_train['anomaly_score'] = -clf.decision_function(X_train)
    df_test['anomaly_score'] = -clf.decision_function(X_test)

    # Save features
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"  Successfully saved updated feature matrices with 'anomaly_score' column.")

    # Backward compatibility copy
    if num_benign == 3000 and ratio_pct == 20:
        import shutil
        shutil.copyfile(iso_model_path, MODELS_DIR / 'isolation_forest.joblib')
        print("  Saved backward-compatible isolation forest model.")
    return True


def run_cmd(args):
    print(f"Executing: {' '.join(args)}")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(args)}")
        print("STDERR:")
        print(result.stderr)
        sys.exit(1)
    return result.stdout


def main():
    print("=== Phase 5: Improving Features via Unsupervised Anomaly Detection ===")

    # 1. Update datasets with anomaly scores
    updated = []
    for num_benign, ratio_pct in [(10000, 10), (10000, 20)]:
        if improve_dataset(num_benign, ratio_pct):
            updated.append((num_benign, ratio_pct))

    if not updated:
        print("No feature files updated. Ensure features are extracted first.")
        return

    # 2. Retrain models to see if the anomaly score feature improves class bounds
    print("\n=== Retraining Models with Improved Features ===")
    for num_benign, ratio_pct in updated:
        ratio_val = float(ratio_pct) / 100.0
        run_cmd([
            sys.executable,
            str(BASE_DIR / 'scripts' / 'model_training' / 'train_evaluate.py'),
            '--num-benign', str(num_benign),
            '--ratio', str(ratio_val)
        ])

    # 3. Re-run WAF3LLM evaluation with the updated tuned XGBoost
    print("\n=== Re-evaluating WAF3LLM with Improved Models ===")
    run_cmd([sys.executable, str(BASE_DIR / 'scripts' / 'model_training' / 'waf3llm.py')])

    print("\nAll improvements successfully incorporated and evaluated!")


if __name__ == '__main__':
    main()
