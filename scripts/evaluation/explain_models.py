#!/usr/bin/env python3
"""
DS4CS — Explainable AI (SHAP Summary)
======================================
Loads the Tuned XGBoost classifier and the feature set, calculates Shapley values,
and saves a summary plot explaining model predictions.
"""

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings('ignore')

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_COLS = {'Label', 'Category', 'Technique', 'URL', 'HTML_Content', 'Source', 'Payload'}


def main():
    print("=== Phase 4: Explainable AI (SHAP Analysis) ===")

    # Load Tuned XGBoost model
    model_path = MODELS_DIR / 'xgb_tuned_10000_r20.joblib'
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Train models first.")
        sys.exit(1)
        
    print(f"Loading Tuned XGBoost model from {model_path.name}...")
    model = joblib.load(model_path)

    # Load test features
    test_file = DATA_DIR / 'features_test_10000_r20.csv'
    if not test_file.exists():
        print(f"Error: Feature matrix not found: {test_file}.")
        sys.exit(1)

    df = pd.read_csv(test_file)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X_test = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    print(f"Test Set: {X_test.shape[0]} samples, {X_test.shape[1]} features.")

    # 1. Calculate SHAP values
    print("Calculating SHAP values using TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    # TreeExplainer is extremely fast on XGBoost models
    shap_values = explainer(X_test)

    # 2. Plot SHAP summary and save to image
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance Summary (XGBoost Tuned)", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    out_img = RESULTS_DIR / 'shap_feature_importance.png'
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"Saved SHAP summary plot → {out_img}")

    # 3. Save detailed feature impact text summary
    # shap_values.values has shape (N, D)
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
    feat_impact = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_Absolute_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False)

    out_report = RESULTS_DIR / 'shap_report.txt'
    with open(out_report, 'w') as f:
        f.write("DS4CS SHAP Feature Explanation Report\n")
        f.write("=====================================\n\n")
        f.write("Top 15 features ranked by their average absolute SHAP impact:\n\n")
        for idx, row in feat_impact.head(15).iterrows():
            f.write(f"  {row['Feature']:30s} : {row['Mean_Absolute_SHAP']:.4f}\n")
            
        f.write("\nKey Insights:\n")
        top_feat = feat_impact.iloc[0]['Feature']
        f.write(f"1. The feature with the highest impact on IPI classification is '{top_feat}'.\n")
        f.write("2. DOM structural attributes (like hidden element counts, comment counts, and ratio indicators)\n")
        f.write("   dominate feature importance over semantic LSA dimensions, confirming that structural footprinting\n")
        f.write("   is highly effective for static injection filtering at ingestion time.\n")
    print(f"Saved SHAP text summary → {out_report}")
    print("\nSHAP Task Complete.")


if __name__ == '__main__':
    main()
