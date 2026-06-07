#!/usr/bin/env python3
"""
DS4CS — Inference entry point.

Loads the serialized semantic pipeline + chosen classifier and scores
a raw HTML string. Mirrors the feature space produced by Phase 2.

Usage:
    python predict.py path/to/page.html
    cat page.html | python predict.py -
    python predict.py page.html --model xgb_tuned   # xgb_tuned | xgb_baseline | rf | lr | or_ensemble | weighted_ensemble
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / 'models'

sys.path.insert(0, str(BASE_DIR / 'scripts' / 'feature_engineering'))
from extract_features import parse_html_features  # noqa: E402
from minhash_vectorizer import MinHashVectorizer  # noqa: E402, F401

# Same exclusion list as training so column ordering matches.
EXCLUDE_COLS = {'Label', 'Category', 'Technique', 'URL', 'HTML_Content', 'Source', 'Payload'}


def featurize(html: str) -> pd.DataFrame:
    pipeline = joblib.load(MODELS_DIR / 'semantic_pipeline_10000_r20.joblib')
    feats = parse_html_features(html)
    hidden_text = feats.pop('hidden_text', '')
    
    # Extract TF-IDF/LSA embeddings
    lsa = pipeline.transform([hidden_text])
    n_lsa = lsa.shape[1]
    row = {k: feats[k] for k in feats if k != 'hidden_text'}
    for i in range(n_lsa):
        row[f'lsa_embed_{i}'] = float(lsa[0, i])
        
    # Extract SBERT embeddings if enabled
    sbert_config_path = MODELS_DIR / 'sbert_config_10000_r20.joblib'
    if sbert_config_path.exists():
        from transformer_embeddings import TransformerEmbedder
        embedder = TransformerEmbedder()
        sbert_vec = embedder.encode([hidden_text], show_progress=False)
        for i in range(sbert_vec.shape[1]):
            row[f'sbert_embed_{i}'] = float(sbert_vec[0, i])

    df = pd.DataFrame([row])
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != 'anomaly_score']
    df_feat = df[feature_cols]
    
    iso_path = MODELS_DIR / 'isolation_forest.joblib'
    if iso_path.exists():
        iso = joblib.load(iso_path)
        df_feat['anomaly_score'] = -iso.decision_function(df_feat)
    return df_feat


def score(html: str, model_name: str):
    X = featurize(html)
    model_paths = {
        'xgb_tuned': 'xgb_tuned_10000_r20.joblib',
        'xgb_baseline': 'xgb_baseline_10000_r20.joblib',
        'rf': 'rf_model_10000_r20.joblib',
        'lr': 'lr_model_10000_r20.joblib',
        'or_ensemble': 'or_ensemble_10000_r20.joblib',
        'weighted_ensemble': 'weighted_ensemble_10000_r20.joblib',
    }
    if model_name not in model_paths:
        raise SystemExit(f'Unknown model {model_name!r}. Available: {list(model_paths)}')
    model = joblib.load(MODELS_DIR / model_paths[model_name])
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])
    return {'malicious_prob': proba, 'prediction': pred, 'features': X.iloc[0].to_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', help='HTML file path, or "-" to read stdin')
    ap.add_argument('--model', default='xgb_tuned')
    ap.add_argument('--verbose', action='store_true', help='also print extracted features')
    args = ap.parse_args()

    html = sys.stdin.read() if args.path == '-' else Path(args.path).read_text(encoding='utf-8', errors='replace')
    result = score(html, args.model)

    label = 'MALICIOUS' if result['prediction'] == 1 else 'benign'
    print(f'{label}  (P(malicious) = {result["malicious_prob"]:.4f})')
    if args.verbose:
        for k, v in result['features'].items():
            if not k.startswith('sbert_embed_') and not k.startswith('lsa_embed_'):
                print(f'  {k:30s} {v}')


if __name__ == '__main__':
    main()
