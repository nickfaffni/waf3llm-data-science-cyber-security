#!/usr/bin/env python3
"""
DS4CS — Ratio Experiment Orchestrator
======================================
Automates the generation, feature extraction, training, and evaluation of
models across different malicious ratio configurations (10%, 20%, 30%, 40%)
for 10,000 base benign HTML records.

Aggregates all metrics into a clean markdown comparison table.
"""

import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON_EXE = sys.executable

RATIOS = [0.1, 0.2, 0.3, 0.4]
NUM_BENIGN = 10000

def run_cmd(args):
    print(f"\nExecuting: {' '.join(args)}")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(args)}")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        sys.exit(1)
    return result.stdout

def main():
    print(f"=== Starting Ratio Experiments (Benign base: {NUM_BENIGN}) ===")
    
    # 1. Run generation, feature extraction, training, and evaluation for each ratio
    for ratio in RATIOS:
        ratio_pct = int(ratio * 100)
        print(f"\n==========================================")
        print(f"Running Experiment for Ratio: {ratio_pct}%")
        print(f"==========================================")
        
        # A. Generation
        run_cmd([
            PYTHON_EXE,
            str(BASE_DIR / 'scripts' / 'data_generation' / 'generate_dataset_real_html.py'),
            '--num-benign', str(NUM_BENIGN),
            '--ratio', str(ratio)
        ])
        
        # B. Feature Extraction
        run_cmd([
            PYTHON_EXE,
            str(BASE_DIR / 'scripts' / 'feature_engineering' / 'extract_features.py'),
            '--num-benign', str(NUM_BENIGN),
            '--ratio', str(ratio)
        ])
        
        # C. Model Training & Evaluation
        run_cmd([
            PYTHON_EXE,
            str(BASE_DIR / 'scripts' / 'model_training' / 'train_evaluate.py'),
            '--num-benign', str(NUM_BENIGN),
            '--ratio', str(ratio)
        ])
        
        # D. OOD Evaluation
        run_cmd([
            PYTHON_EXE,
            str(BASE_DIR / 'scripts' / 'evaluation' / 'evaluate_ood.py'),
            '--num-benign', str(NUM_BENIGN),
            '--ratio', str(ratio)
        ])

    # 2. Parse and aggregate results
    print("\n==========================================")
    print("Aggregating Results...")
    print("==========================================")
    
    aggregated = []
    
    for ratio in RATIOS:
        ratio_pct = int(ratio * 100)
        
        # Load ID (In-Distribution) metrics
        id_path = BASE_DIR / 'results' / f'metrics_report_{NUM_BENIGN}_r{ratio_pct}.json'
        ood_path = BASE_DIR / 'results' / f'ood_evaluation_{NUM_BENIGN}_r{ratio_pct}.json'
        
        if not id_path.exists() or not ood_path.exists():
            print(f"Warning: Missing results for ratio {ratio_pct}%")
            continue
            
        with open(id_path) as f:
            id_data = json.load(f)
        with open(ood_path) as f:
            ood_data = json.load(f)
            
        # Extract Tuned XGBoost metrics
        xgb_tuned_id = next((m for m in id_data['models'] if m['Model'] == 'XGBoost (tuned)'), None)
        xgb_tuned_ood_adv = ood_data['adversarial'].get('xgb_tuned', {})
        xgb_tuned_ood_nat = ood_data['natural_benign'].get('xgb_tuned', {})
        
        # Extract Hard-OR metrics
        or_ens_id = next((m for m in id_data['models'] if m['Model'] == 'Hard-OR Ensemble'), None)
        or_ens_ood_adv = ood_data['adversarial'].get('or_ensemble', {})
        or_ens_ood_nat = ood_data['natural_benign'].get('or_ensemble', {})

        # Extract true LOTO (Leave-One-Technique-Out) mean recall
        loto_mean = id_data.get('true_leave_one_technique_out_xgb_tuned', {}).get('_mean', 0.0)

        aggregated.append({
            'ratio': ratio_pct,
            'id_xgb_f1': xgb_tuned_id.get('F1_Score', 0.0) if xgb_tuned_id else 0.0,
            'id_xgb_tpr': xgb_tuned_id.get('TPR', 0.0) if xgb_tuned_id else 0.0,
            'id_xgb_fpr': xgb_tuned_id.get('FPR', 0.0) if xgb_tuned_id else 0.0,
            'id_xgb_prauc': xgb_tuned_id.get('PR_AUC', 0.0) if xgb_tuned_id else 0.0,
            
            'loto_mean_recall': loto_mean,
            
            'ood_adv_xgb_recall': xgb_tuned_ood_adv.get('recall', 0.0),
            'ood_nat_xgb_fpr': xgb_tuned_ood_nat.get('fpr', 0.0),
            
            'id_or_f1': or_ens_id.get('F1_Score', 0.0) if or_ens_id else 0.0,
            'id_or_tpr': or_ens_id.get('TPR', 0.0) if or_ens_id else 0.0,
            'id_or_fpr': or_ens_id.get('FPR', 0.0) if or_ens_id else 0.0,
            'ood_adv_or_recall': or_ens_ood_adv.get('recall', 0.0),
            'ood_nat_or_fpr': or_ens_ood_nat.get('fpr', 0.0),
        })

    # Write Markdown summary report
    markdown_path = BASE_DIR / 'results' / f'ratio_experiments_summary_{NUM_BENIGN}.md'
    with open(markdown_path, 'w') as f:
        f.write(f"# Ratio Experiment Results Summary (10k records)\n\n")
        f.write("This report evaluates the impact of class imbalance (malicious to benign ratio) on WAF detection capabilities and false positive rates. The datasets use **10,000 base benign HTML records**.\n\n")
        
        f.write("### In-Distribution & OOD Generalization (Tuned XGBoost)\n\n")
        f.write("| Malicious Ratio | ID TPR (Recall) | ID FPR | ID F1-Score | ID PR-AUC | LOTO Mean Recall | OOD Adv Recall | Natural-Benign FPR |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for a in aggregated:
            f.write(f"| **{a['ratio']}%** | {a['id_xgb_tpr']:.2%} | {a['id_xgb_fpr']:.2%} | {a['id_xgb_f1']:.2%} | {a['id_xgb_prauc']:.4f} | {a['loto_mean_recall']:.2%} | {a['ood_adv_xgb_recall']:.2%} | {a['ood_nat_xgb_fpr']:.2%} |\n")
            
        f.write("\n### High-Recall WAF Operational Comparison (Hard-OR Ensemble)\n\n")
        f.write("| Malicious Ratio | ID TPR (Recall) | ID FPR | ID F1-Score | OOD Adv Recall | Natural-Benign FPR |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for a in aggregated:
            f.write(f"| **{a['ratio']}%** | {a['id_or_tpr']:.2%} | {a['id_or_fpr']:.2%} | {a['id_or_f1']:.2%} | {a['ood_adv_or_recall']:.2%} | {a['ood_nat_or_fpr']:.2%} |\n")

        f.write("\n### Key Takeaways & Observations\n\n")
        f.write("1. **Class Weight Balancing (`scale_pos_weight`)**: Model scaling weight scales inversely with the malicious portion, preventing low ratio models (e.g. 10%) from collapsing. However, higher malicious portions generally lead to more stable training signals and better decision boundaries.\n")
        f.write("2. **False Positive Rates (FPR)**: Compare the false alarm rates on the natural-benign web traffic dataset across the varying ratios. Typically, lower training-time malicious ratios can reduce the models' over-sensitivity to code-like indicators, mitigating the 'Readability Trap'.\n")
        f.write("3. **OOD Generalization**: Higher malicious representation during training may help or hurt OOD generalization depending on structural diversity. Check LOTO and OOD Adversarial columns to observe generalization trends.\n")

    print(f"\nSuccessfully generated aggregated experiment results: {markdown_path}")

if __name__ == "__main__":
    main()
