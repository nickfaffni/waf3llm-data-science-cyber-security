#!/usr/bin/env python3
"""
DS4CS — Jaccard Distance Structural Similarity Classifier
===========================================================
Parses HTML layout nodes (tags, classes, style properties, comments) into structural sets
and classifies pages using Jaccard Similarity against a database of known training configurations.
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Comment

warnings.filterwarnings('ignore')

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pick the fastest available parser
import importlib.util
PARSER = 'lxml' if importlib.util.find_spec('lxml') is not None else 'html.parser'
RANDOM_STATE = 42



# ============================================================
# Structural Token Extraction
# ============================================================
def extract_structural_tokens(html_content: str) -> set:
    """Extract set of tags, classes, style declarations, and comments as a layout fingerprint."""
    soup = BeautifulSoup(html_content or '', PARSER)
    tokens = set()
    
    # 1. Tags and Attributes
    for tag in soup.find_all(True):
        tokens.add(f"tag:{tag.name.lower()}")
        for attr in tag.attrs:
            tokens.add(f"attr:{attr.lower()}")
            
            # Extract classes
            if attr.lower() == 'class':
                classes = tag.get('class', [])
                if isinstance(classes, list):
                    for cls in classes:
                        tokens.add(f"class:{cls.lower()}")
                elif isinstance(classes, str):
                    tokens.add(f"class:{classes.lower()}")
                    
            # Extract inline style declarations
            elif attr.lower() == 'style':
                style_val = tag.get('style', '')
                if style_val:
                    for decl in style_val.split(';'):
                        if ':' in decl:
                            prop = decl.split(':')[0].strip().lower()
                            tokens.add(f"style-prop:{prop}")
                            
    # 2. Comments
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        tokens.add("comment:present")
        
    return tokens


# ============================================================
# Jaccard Helper
# ============================================================
def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


# ============================================================
# Jaccard Classifier
# ============================================================
class JaccardClassifier:
    def __init__(self):
        self.benign_db = []
        self.malicious_db = []

    def fit(self, X_train_html, y_train):
        print("Fitting Jaccard Database from training HTML contents...")
        self.benign_db = []
        self.malicious_db = []
        
        for idx, (html, label) in enumerate(zip(X_train_html, y_train)):
            if idx % 1000 == 0:
                print(f"  Fingerprinted {idx}/{len(y_train)} pages...")
            tokens = extract_structural_tokens(html)
            if label == 1:
                self.malicious_db.append(tokens)
            else:
                self.benign_db.append(tokens)
        print(f"Jaccard Database loaded: {len(self.benign_db)} benign, {len(self.malicious_db)} malicious templates.")

    def predict_score(self, html: str) -> float:
        """Score is S_malicious - S_benign. Positive means malicious, negative means benign."""
        tokens = extract_structural_tokens(html)
        
        # Max similarity to known benign layouts
        s_benign = max([jaccard_similarity(tokens, b) for b in self.benign_db], default=0.0)
        # Max similarity to known malicious layouts
        s_mal = max([jaccard_similarity(tokens, m) for m in self.malicious_db], default=0.0)
        
        return s_mal, s_benign

    def predict(self, X_html):
        preds = []
        for idx, html in enumerate(X_html):
            if idx % 500 == 0:
                print(f"  Scored {idx}/{len(X_html)} layouts...")
            s_mal, s_ben = self.predict_score(html)
            preds.append(1 if s_mal > s_ben else 0)
        return np.array(preds)


# ============================================================
# Main Evaluation
# ============================================================
def main():
    print("=== Phase 3: Jaccard Similarity Classifier ===")
    
    train_file = DATA_DIR / 'train_real_html.csv'
    test_file = DATA_DIR / 'test_real_html.csv'
    
    if not train_file.exists() or not test_file.exists():
        print("Error: train/test HTML datasets missing. Generate them first.")
        sys.exit(1)
        
    print("Loading datasets...")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    X_train_html = df_train['HTML_Content'].fillna('')
    y_train = df_train['Label'].astype(int).values
    
    X_test_html = df_test['HTML_Content'].fillna('')
    y_test = df_test['Label'].astype(int).values
    
    clf = JaccardClassifier()
    # Sample a representative subset for the DB to run fits and predictions quickly (since pairwise is O(N_test * N_train))
    # Using 1000 train samples and 500 test samples gives high statistical significance and finishes in 5-10s.
    np.random.seed(RANDOM_STATE)
    train_idx = np.random.choice(len(X_train_html), 1000, replace=False)
    test_idx = np.random.choice(len(X_test_html), 500, replace=False)
    
    clf.fit(X_train_html.iloc[train_idx].values, y_train[train_idx])
    
    print("\nRunning predictions on test subset...")
    y_pred = clf.predict(X_test_html.iloc[test_idx].values)
    y_true_sub = y_test[test_idx]
    
    # Calculate Jaccard metrics
    tp = ((y_pred == 1) & (y_true_sub == 1)).sum()
    fp = ((y_pred == 1) & (y_true_sub == 0)).sum()
    fn = ((y_pred == 0) & (y_true_sub == 1)).sum()
    tn = ((y_pred == 0) & (y_true_sub == 0)).sum()
    
    acc = (tp + tn) / len(y_true_sub)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    print("\n--- Jaccard Similarity Classifier Subset Performance ---")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    
    # Save Jaccard Classifier database for WAF use
    joblib.dump(clf, MODELS_DIR / 'jaccard_similarity_model.joblib')
    print(f"\nSaved Jaccard model → {MODELS_DIR / 'jaccard_similarity_model.joblib'}")
    
    # Write report
    report_path = RESULTS_DIR / 'jaccard_metrics_report.json'
    import json
    with open(report_path, 'w') as f:
        json.dump({
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1_Score': float(f1),
        }, f, indent=2)
    print(f"Saved Jaccard report → {report_path}")
    print("Jaccard Task Complete.")


if __name__ == '__main__':
    main()
