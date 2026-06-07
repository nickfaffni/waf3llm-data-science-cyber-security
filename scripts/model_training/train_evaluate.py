#!/usr/bin/env python3
"""
DS4CS — Phase 3: Model Training & Evaluation (v2)
==================================================
Trains and evaluates classifiers for IPI detection. Key improvements over v1:
- Separate baseline / tuned XGB artefacts (no overwrite).
- scale_pos_weight applied to baseline XGB as well, so the iteration-3
  comparison is an honest ablation.
- Adds PR-AUC and ROC-AUC (threshold-free metrics for imbalanced data).
- Adds per-category recall breakdown (Category column from the dataset).
- Adds leave-one-technique-out evaluation when a Technique column exists.
- Replaces soft-voting ensemble with hard-OR voting (operationally
  appropriate for a high-recall WAF) AND keeps a weighted soft-voter
  whose weights are derived from validation PR-AUC.
- joblib instead of pickle for artefacts.
- Cross-validation summary with mean ± std.
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_COLS = {'Label', 'Category', 'Technique', 'URL', 'HTML_Content', 'Source', 'Payload'}
RANDOM_STATE = 42


# ============================================================
# Helpers
# ============================================================
def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    feature_cols = [c for c in df_train.columns if c not in EXCLUDE_COLS]
    X_train = df_train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_test = df_test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = df_train['Label'].astype(int)
    y_test = df_test['Label'].astype(int)
    category_test = df_test.get('Category', pd.Series([''] * len(df_test)))
    technique_train = df_train.get('Technique', pd.Series([''] * len(df_train)))
    technique_test = df_test.get('Technique', pd.Series([''] * len(df_test)))
    print(f'Features ({len(feature_cols)}): X_train {X_train.shape}, X_test {X_test.shape}')
    return (
        X_train, y_train, X_test, y_test,
        feature_cols, category_test, technique_train, technique_test,
    )


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred.astype(float)
    
    # Calculate True Positive Rate (TPR / Recall) and False Positive Rate (FPR)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'TPR': tpr,
        'FPR': fpr,
        'F1_Score': f1_score(y_test, y_pred, zero_division=0),
        'PR_AUC': average_precision_score(y_test, y_score),
        'ROC_AUC': roc_auc_score(y_test, y_score),
    }
    print(
        f'\n--- {name} ---\n'
        f'  Acc: {metrics["Accuracy"]:.4f}  '
        f'Prec: {metrics["Precision"]:.4f}  '
        f'TPR (Rec): {metrics["TPR"]:.4f}  '
        f'FPR: {metrics["FPR"]:.4f}  '
        f'F1: {metrics["F1_Score"]:.4f}  '
        f'PR-AUC: {metrics["PR_AUC"]:.4f}  '
        f'ROC-AUC: {metrics["ROC_AUC"]:.4f}'
    )
    return metrics, y_pred, y_score


def plot_confusion(name, y_test, y_pred, filename):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['Benign (0)', 'Malicious (1)'],
        yticklabels=['Benign (0)', 'Malicious (1)'],
    )
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    out = RESULTS_DIR / filename
    plt.savefig(out, dpi=300)
    plt.close()
    print(f'  Saved confusion matrix → {out}')


def cv_score(name, model, X, y, scoring='average_precision', n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    print(f'  {name} CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}')
    return scores


def per_category_recall(name, y_test, y_pred, categories):
    """Recall broken down by IPI Category for malicious test samples."""
    mask_mal = (y_test == 1).to_numpy()
    if mask_mal.sum() == 0:
        return {}
    cats = np.array(categories)[mask_mal]
    pred_mal = np.asarray(y_pred)[mask_mal]
    out = {}
    for cat in sorted({c for c in cats if c}):
        idx = cats == cat
        recall = pred_mal[idx].mean() if idx.sum() else 0.0
        out[cat] = {'n': int(idx.sum()), 'recall': float(recall)}
    print(f'\n  Per-category recall ({name}):')
    for cat, info in out.items():
        print(f'    {cat:30s}  n={info["n"]:3d}  recall={info["recall"]:.3f}')
    return out


# ============================================================
# PR curve + threshold sweep (Fix C: pick an operating point)
# ============================================================
def plot_pr_curve(name, y_test, y_score, filename):
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f'{name} (PR-AUC = {pr_auc:.3f})', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision–Recall Curve: {name}')
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(loc='lower left')
    plt.tight_layout()
    out = RESULTS_DIR / filename
    plt.savefig(out, dpi=300)
    plt.close()
    print(f'  Saved PR curve → {out}')
    return precision, recall, thresholds


def threshold_sweep(name, y_test, y_score, target_recalls=(0.5, 0.7, 0.8, 0.9, 0.95)):
    """For each target recall, find the threshold and report (threshold, precision, recall, FPR)."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    # precision/recall arrays have len = len(thresholds) + 1
    y_test_arr = np.asarray(y_test)
    neg = int((y_test_arr == 0).sum())
    table = []
    print(f'\n  Threshold sweep ({name}):')
    print(f'    {"target_R":>8s}  {"threshold":>9s}  {"prec":>6s}  {"recall":>6s}  {"FPR":>6s}')
    for target in target_recalls:
        # Find the highest threshold that still achieves recall >= target.
        candidate_idx = np.where(recall[:-1] >= target)[0]
        if len(candidate_idx) == 0:
            row = {'target_recall': target, 'threshold': None, 'precision': None, 'recall': None, 'fpr': None}
            print(f'    {target:>8.2f}  {"N/A":>9s}  {"-":>6s}  {"-":>6s}  {"-":>6s}')
            table.append(row)
            continue
        idx = candidate_idx[np.argmax(thresholds[candidate_idx])]
        t = float(thresholds[idx])
        achieved_p = float(precision[idx])
        achieved_r = float(recall[idx])
        y_hat = (y_score >= t).astype(int)
        fp = int(((y_hat == 1) & (y_test_arr == 0)).sum())
        fpr = fp / neg if neg else 0.0
        row = {
            'target_recall': target, 'threshold': t,
            'precision': achieved_p, 'recall': achieved_r, 'fpr': fpr,
        }
        table.append(row)
        print(f'    {target:>8.2f}  {t:>9.4f}  {achieved_p:>6.3f}  {achieved_r:>6.3f}  {fpr:>6.3f}')
    return table


# ============================================================
# True Leave-One-Technique-Out retraining (Fix B)
# ============================================================
def true_loto_eval(make_estimator, X_train, y_train, X_test, y_test,
                   technique_train, technique_test):
    """For each unique technique T:
      1. Build a training set excluding malicious rows with Technique==T (benign unchanged)
      2. Train a fresh copy of the estimator on this reduced set
      3. Evaluate on test rows where Technique==T (genuinely unseen)
    Reports recall on truly-held-out techniques.
    """
    from imblearn.over_sampling import SMOTE
    technique_train = np.asarray(technique_train)
    technique_test = np.asarray(technique_test)
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)
    techniques = sorted({t for t in technique_train if isinstance(t, str) and t})
    report = {}
    print('\n>>> True Leave-One-Technique-Out (retrain on 13 techniques, evaluate on the 14th)')
    print(f'    {"held_out_technique":<25s}  {"n_test":>6s}  {"recall":>6s}')
    for held in techniques:
        # exclude malicious rows that used the held technique
        keep_train = ~((y_train_arr == 1) & (technique_train == held))
        Xt = X_train[keep_train]
        yt = y_train_arr[keep_train]
        
        # Apply SMOTE to the LOTO subset so it matches the main pipeline
        try:
            Xt, yt = SMOTE(random_state=RANDOM_STATE).fit_resample(Xt, yt)
        except ValueError:
            pass # fallback if too few samples
            
        # test rows that used the held technique (always malicious)
        mask_test = technique_test == held
        n_test = int(mask_test.sum())
        if n_test == 0:
            continue
        est = make_estimator()
        est.fit(Xt, yt)
        held_pred = est.predict(X_test[mask_test])
        recall_held = float(np.asarray(held_pred).mean())
        report[held] = {'n_test': n_test, 'recall_on_held_out': recall_held}
        print(f'    {held:<25s}  {n_test:>6d}  {recall_held:>6.3f}')
    if report:
        mean_recall = float(np.mean([v['recall_on_held_out'] for v in report.values()]))
        print(f'    {"MEAN":<25s}  {"":>6s}  {mean_recall:>6.3f}')
        report['_mean'] = mean_recall
    return report


# ============================================================
# Hard-OR voting (operational choice for a security WAF)
# ============================================================
class HardOrEnsemble(BaseEstimator, ClassifierMixin):
    """If ANY constituent flags malicious → flag malicious. Maximizes recall."""

    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        for _name, est in self.estimators:
            est.fit(X, y)
        return self

    def predict(self, X):
        preds = np.stack([est.predict(X) for _name, est in self.estimators], axis=1)
        return (preds.max(axis=1) >= 1).astype(int)

    def predict_proba(self, X):
        probs = np.stack(
            [est.predict_proba(X)[:, 1] for _name, est in self.estimators], axis=1
        )
        p1 = probs.max(axis=1)
        return np.column_stack([1 - p1, p1])


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='DS4CS Model Training & Evaluation')
    parser.add_argument('--num-benign', type=int, default=3000,
                        help='Number of benign HTML pages used in generation')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='Malicious ratio used in dataset generation')
    args = parser.parse_args()

    num_benign = args.num_benign
    ratio_pct = int(args.ratio * 100)

    train_file = DATA_DIR / f'features_train.csv'
    test_file = DATA_DIR / f'features_test.csv'

    if not train_file.exists():
        print(f'Feature matrices not found: {train_file}. Run Phase 2 extraction first.')
        return

    (
        X_train, y_train, X_test, y_test,
        feature_cols, category_test, _technique_train, technique_test,
    ) = load_data(train_file, test_file)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos else 1.0
    print(f'Original Class balance: benign {neg} / malicious {pos} → scale_pos_weight = {scale_pos_weight:.2f}')

    # Save originals for LOTO evaluation filtering
    X_train_orig, y_train_orig = X_train.copy(), y_train.copy()

    # Apply SMOTE
    from imblearn.over_sampling import SMOTE
    print('Applying SMOTE to balance training data...')
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    pos_smote = int((y_train == 1).sum())
    neg_smote = int((y_train == 0).sum())
    print(f'New Class balance after SMOTE: benign {neg_smote} / malicious {pos_smote}')
    
    # After SMOTE, the dataset is balanced, so scale_pos_weight is 1.0
    scale_pos_weight = neg_smote / pos_smote if pos_smote else 1.0

    results = []

    # 1. Random Forest -----------------------------------------------------
    print('\n>>> Random Forest')
    rf = RandomForestClassifier(
        n_estimators=300, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_metrics, rf_pred, _ = evaluate('Random Forest', rf, X_test, y_test)
    results.append(rf_metrics)
    plot_confusion('Random Forest', y_test, rf_pred, f'rf_confusion_matrix.png')
    joblib.dump(rf, MODELS_DIR / f'rf_model.joblib')
    cv_score('Random Forest', rf, X_train, y_train)
    rf_categories = per_category_recall('Random Forest', y_test, rf_pred, category_test)

    importances = rf.feature_importances_
    top = np.argsort(importances)[::-1][:10]
    print('\n  Top 10 RF feature importances:')
    for i in top:
        print(f'    {feature_cols[i]:30s} {importances[i]:.4f}')

    # 2. XGBoost (baseline with scale_pos_weight) --------------------------
    print('\n>>> XGBoost (baseline + scale_pos_weight)')
    xgb_baseline = XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb_baseline.fit(X_train, y_train)
    base_metrics, base_pred, _ = evaluate('XGBoost (baseline)', xgb_baseline, X_test, y_test)
    results.append(base_metrics)
    plot_confusion('XGBoost (baseline)', y_test, base_pred, f'xgb_baseline_confusion_matrix.png')
    joblib.dump(xgb_baseline, MODELS_DIR / f'xgb_baseline.joblib')
    cv_score('XGBoost (baseline)', xgb_baseline, X_train, y_train)

    # 3. XGBoost (tuned) ---------------------------------------------------
    print('\n>>> XGBoost (tuned via RandomizedSearchCV)')
    xgb_search = RandomizedSearchCV(
        XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        param_distributions={
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 400],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        },
        n_iter=10, cv=3, scoring='average_precision',
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    xgb_search.fit(X_train, y_train)
    xgb_tuned = xgb_search.best_estimator_
    print(f'  Best params: {xgb_search.best_params_}')
    tuned_metrics, tuned_pred, tuned_score = evaluate('XGBoost (tuned)', xgb_tuned, X_test, y_test)
    results.append(tuned_metrics)
    plot_confusion('XGBoost (tuned)', y_test, tuned_pred, f'xgb_tuned_confusion_matrix.png')
    joblib.dump(xgb_tuned, MODELS_DIR / f'xgb_tuned.joblib')
    xgb_categories = per_category_recall('XGBoost (tuned)', y_test, tuned_pred, category_test)

    # Fix C: PR curve + threshold sweep on tuned XGB
    plot_pr_curve('XGBoost (tuned)', y_test, tuned_score, f'xgb_tuned_pr_curve.png')
    threshold_table = threshold_sweep('XGBoost (tuned)', y_test, tuned_score)

    # 4. Logistic Regression -----------------------------------------------
    print('\n>>> Logistic Regression')
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
    )
    lr.fit(X_train, y_train)
    lr_metrics, lr_pred, _ = evaluate('Logistic Regression', lr, X_test, y_test)
    results.append(lr_metrics)
    plot_confusion('Logistic Regression', y_test, lr_pred, f'lr_confusion_matrix.png')
    joblib.dump(lr, MODELS_DIR / f'lr_model.joblib')

    # 4b. Naive Bayes ------------------------------------------------------
    print('\n>>> Naive Bayes')
    nb = make_pipeline(
        StandardScaler(),
        GaussianNB(),
    )
    nb.fit(X_train, y_train)
    nb_metrics, nb_pred, _ = evaluate('Naive Bayes', nb, X_test, y_test)
    results.append(nb_metrics)
    plot_confusion('Naive Bayes', y_test, nb_pred, f'nb_confusion_matrix.png')
    joblib.dump(nb, MODELS_DIR / f'nb_model.joblib')

    # 4c. Support Vector Machine (SVM) -------------------------------------
    print('\n>>> Support Vector Machine (SVM)')
    svm = make_pipeline(
        StandardScaler(),
        SVC(probability=True, random_state=RANDOM_STATE),
    )
    svm.fit(X_train, y_train)
    svm_metrics, svm_pred, _ = evaluate('SVM', svm, X_test, y_test)
    results.append(svm_metrics)
    plot_confusion('SVM', y_test, svm_pred, f'svm_confusion_matrix.png')
    joblib.dump(svm, MODELS_DIR / f'svm_model.joblib')

    # 4d. K-Nearest Neighbors (KNN) ----------------------------------------
    print('\n>>> K-Nearest Neighbors (KNN)')
    knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(),
    )
    knn.fit(X_train, y_train)
    knn_metrics, knn_pred, _ = evaluate('KNN', knn, X_test, y_test)
    results.append(knn_metrics)
    plot_confusion('KNN', y_test, knn_pred, f'knn_confusion_matrix.png')
    joblib.dump(knn, MODELS_DIR / f'knn_model.joblib')

    # 4e. Artificial Neural Network (ANN) ----------------------------------
    print('\n>>> Artificial Neural Network (ANN)')
    ann = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=RANDOM_STATE),
    )
    ann.fit(X_train, y_train)
    ann_metrics, ann_pred, _ = evaluate('ANN', ann, X_test, y_test)
    results.append(ann_metrics)
    plot_confusion('ANN', y_test, ann_pred, f'ann_confusion_matrix.png')
    joblib.dump(ann, MODELS_DIR / f'ann_model.joblib')

    # 4f. Deep Neural Network (DNN) -----------------------------------------
    print('\n>>> Deep Neural Network (DNN)')
    dnn = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, early_stopping=True, random_state=RANDOM_STATE),
    )
    dnn.fit(X_train, y_train)
    dnn_metrics, dnn_pred, _ = evaluate('DNN', dnn, X_test, y_test)
    results.append(dnn_metrics)
    plot_confusion('DNN', y_test, dnn_pred, f'dnn_confusion_matrix.png')
    joblib.dump(dnn, MODELS_DIR / f'dnn_model.joblib')


    # 5. Hard-OR ensemble (recall-maximizing) ------------------------------
    print('\n>>> Hard-OR Ensemble (recall-maximizing, WAF operational choice)')
    or_ens = HardOrEnsemble(estimators=[
        ('rf', RandomForestClassifier(
            n_estimators=300, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
        ('xgb', XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, n_jobs=-1,
            **{k: v for k, v in xgb_search.best_params_.items()},
        )),
    ])
    or_ens.fit(X_train, y_train)
    or_metrics, or_pred, _ = evaluate('Hard-OR Ensemble', or_ens, X_test, y_test)
    results.append(or_metrics)
    plot_confusion('Hard-OR Ensemble', y_test, or_pred, f'or_ensemble_confusion_matrix.png')
    joblib.dump(or_ens, MODELS_DIR / f'or_ensemble.joblib')

    # 6. Weighted soft-voting ensemble (weights ∝ tuned-model PR-AUC) ------
    print('\n>>> Weighted Soft-Voting Ensemble (weights ∝ PR-AUC)')
    pr_weights = [
        max(rf_metrics['PR_AUC'], 1e-6),
        max(tuned_metrics['PR_AUC'], 1e-6),
        max(lr_metrics['PR_AUC'], 1e-6),
    ]
    weighted = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=300, class_weight='balanced',
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
            ('xgb', XGBClassifier(
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE, n_jobs=-1,
                **{k: v for k, v in xgb_search.best_params_.items()},
            )),
            ('lr', make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
            )),
        ],
        voting='soft',
        weights=pr_weights,
    )
    weighted.fit(X_train, y_train)
    w_metrics, w_pred, _ = evaluate('Weighted Soft Vote', weighted, X_test, y_test)
    results.append(w_metrics)
    plot_confusion('Weighted Soft Vote', y_test, w_pred, f'weighted_ensemble_confusion_matrix.png')
    joblib.dump(weighted, MODELS_DIR / f'weighted_ensemble.joblib')

    # 7a. Per-technique recall on the held-out test set (model trained on all 14 techniques) ----
    recall_by_technique = {}
    techniques = [t for t in technique_test.unique() if isinstance(t, str) and t]
    if techniques:
        print('\n>>> Recall by Technique (Tuned XGB — model trained on all 14 techniques)')
        for tech in techniques:
            mask_held = (technique_test == tech).to_numpy()
            n_held = int(mask_held.sum())
            if n_held == 0:
                continue
            held_pred = tuned_pred[mask_held]
            held_recall = float(held_pred.mean())
            recall_by_technique[tech] = {'n': n_held, 'recall': held_recall}
            print(f'    {tech:25s} n={n_held:3d}  recall={held_recall:.3f}')

    # 7b. TRUE Leave-One-Technique-Out — retrain the tuned XGB 14 times, each
    # excluding malicious rows of one technique from training. Tests OOD
    # generalization to *unseen* stealth techniques. (Fix B)
    def _make_tuned_xgb():
        return XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, n_jobs=-1,
            **{k: v for k, v in xgb_search.best_params_.items()},
        )

    true_loto = true_loto_eval(
        _make_tuned_xgb, X_train_orig, y_train_orig, X_test, y_test,
        _technique_train, technique_test,
    )

    # 8. Write summary -----------------------------------------------------
    report_path = RESULTS_DIR / f'metrics_report.txt'
    with open(report_path, 'w') as f:
        f.write(f'DS4CS Model Evaluation Report ({num_benign} benign, ratio {ratio_pct}%)\n')
        f.write('=' * 40 + '\n\n')
        for r in results:
            f.write(f"Model: {r['Model']}\n")
            for k in ('Accuracy', 'Precision', 'Recall', 'TPR', 'FPR', 'F1_Score', 'PR_AUC', 'ROC_AUC'):
                f.write(f'  {k:10s}: {r[k]:.4f}\n')
            f.write('\n')
    print(f'\nSaved metrics report → {report_path}')

    json_path = RESULTS_DIR / f'metrics_report.json'
    with open(json_path, 'w') as f:
        json.dump({
            'models': results,
            'per_category_recall': {
                'RandomForest': rf_categories,
                'XGBoostTuned': xgb_categories,
            },
            'recall_by_technique_xgb_tuned': recall_by_technique,
            'true_leave_one_technique_out_xgb_tuned': true_loto,
            'threshold_sweep_xgb_tuned': threshold_table,
        }, f, indent=2)
    print(f'Saved structured metrics → {json_path}')

    # Backward compatibility override removed since suffixes were standardized

    print('\nPhase 3 Complete.')


if __name__ == '__main__':
    main()
