#!/usr/bin/env python3
"""
DS4CS — Out-of-Distribution Evaluation
=======================================
Scores the trained classifiers on two held-out sets the model has never seen:

1. `data/ood/ood_adversarial.csv` — 150 malicious samples injected with 8
   stealth techniques that are NOT in the training-time injector vocabulary
   (e.g. class-based <style> hide, text-indent offscreen, color camouflage,
   transform translate, CSS pseudo-element content, aria-hidden, HTML-entity-
   encoded visible text, <template> tags). Tests generalization to unseen
   stealth styles.

2. `data/ood/natural_benign/*.html` — hand-crafted realistic web markup
   (Bootstrap modals, CSRF tokens, ARIA-hidden screen-reader content, lazy-
   loaded images, React SPA shells, etc.). These pages legitimately use the
   same DOM patterns the classifier flags. Tests false-positive rate on
   realistic web traffic — the core "Readability Trap" risk the project is
   built around.

Outputs:
    results/ood_evaluation.json     — structured metrics
    results/ood_evaluation.txt      — human-readable summary
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
OOD_DIR = DATA_DIR / 'ood'
NATURAL_BENIGN_DIR = OOD_DIR / 'natural_benign'

sys.path.insert(0, str(BASE_DIR / 'scripts' / 'feature_engineering'))
sys.path.insert(0, str(BASE_DIR / 'scripts' / 'model_training'))
from extract_features import parse_html_features, extract_dom_graph_features  # noqa: E402
# Importing HardOrEnsemble registers the class so joblib can unpickle or_ensemble.joblib
from train_evaluate import HardOrEnsemble  # noqa: E402, F401
from minhash_vectorizer import MinHashVectorizer  # noqa: E402, F401

csv.field_size_limit(sys.maxsize)

EXCLUDE_COLS = {'Label', 'Category', 'Technique', 'URL', 'HTML_Content', 'Source', 'Payload'}


def _featurize_one(html: str, semantic_pipeline) -> dict:
    feats = parse_html_features(html or '')
    hidden_text = feats.pop('hidden_text', '')
    lsa = semantic_pipeline.transform([hidden_text])
    row = {k: feats[k] for k in feats if k != 'hidden_text'}
    for i in range(lsa.shape[1]):
        row[f'lsa_embed_{i}'] = float(lsa[0, i])
    return row, hidden_text


def featurize_corpus(pairs, semantic_pipeline, iso_forest=None, sbert_embedder=None):
    """pairs: list of (html, metadata_dict). Returns (feature_df, metadata_list)."""
    rows, metas, hidden_texts = [], [], []
    for html, meta in pairs:
        row, ht = _featurize_one(html, semantic_pipeline)
        rows.append(row)
        metas.append(meta)
        hidden_texts.append(ht)
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], 0).fillna(0)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != 'anomaly_score']
    df_feat = df[feature_cols].copy()

    # Transformer embeddings
    if sbert_embedder is not None:
        from transformer_embeddings import EMBED_DIM
        sbert_vecs = sbert_embedder.encode(hidden_texts, show_progress=False)
        sbert_cols = [f'sbert_embed_{i}' for i in range(EMBED_DIM)]
        sbert_df = pd.DataFrame(sbert_vecs, columns=sbert_cols)
        df_feat = pd.concat([df_feat.reset_index(drop=True), sbert_df], axis=1)

    # Anomaly score
    if iso_forest is not None:
        iso_cols = [c for c in df_feat.columns if c not in EXCLUDE_COLS and c != 'anomaly_score']
        df_feat['anomaly_score'] = -iso_forest.decision_function(df_feat[iso_cols])
    return df_feat, metas


def load_ood_adversarial():
    path = OOD_DIR / 'ood_adversarial.csv'
    if not path.exists():
        raise SystemExit(f'Missing {path}. Run generate_ood_set.py first.')
    pairs = []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pairs.append((row['HTML_Content'], {'technique': row['Technique'], 'label': 1}))
    print(f'  OOD adversarial: {len(pairs)} samples')
    return pairs


def load_natural_benign():
    pairs = []
    # Hand-crafted realistic SPA / form / docs / etc.
    if NATURAL_BENIGN_DIR.exists():
        for p in sorted(NATURAL_BENIGN_DIR.glob('*.html')):
            pairs.append((p.read_text(encoding='utf-8'), {'source': p.name, 'label': 0}))
    # The existing example fixtures (5 hand-picked benign C4 pages)
    examples_dir = DATA_DIR / 'examples'
    if examples_dir.exists():
        for p in sorted(examples_dir.glob('benign_*.html')):
            pairs.append((p.read_text(encoding='utf-8'), {'source': f'examples/{p.name}', 'label': 0}))
    print(f'  Natural benign: {len(pairs)} samples')
    return pairs


def score_model(model, X, y_true):
    y_pred = model.predict(X)
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = y_pred.astype(float)
    y_true_arr = np.asarray(y_true)
    out = {
        'n': int(len(y_true_arr)),
        'n_positive': int((y_true_arr == 1).sum()),
        'n_negative': int((y_true_arr == 0).sum()),
        'flagged': int((y_pred == 1).sum()),
        'true_positive': int(((y_pred == 1) & (y_true_arr == 1)).sum()),
        'false_positive': int(((y_pred == 1) & (y_true_arr == 0)).sum()),
        'recall': float(((y_pred == 1) & (y_true_arr == 1)).sum() / max((y_true_arr == 1).sum(), 1)),
        'fpr': float(((y_pred == 1) & (y_true_arr == 0)).sum() / max((y_true_arr == 0).sum(), 1)),
        'mean_score': float(np.mean(y_score)) if len(y_score) else 0.0,
    }
    return out, y_pred, y_score


def per_group_recall(y_pred, metas, key):
    """For each unique value of metas[i][key], compute (n, recall_or_fpr)."""
    groups = {}
    for pred, meta in zip(y_pred, metas):
        g = meta.get(key, '_unknown')
        groups.setdefault(g, []).append(int(pred))
    return {
        g: {'n': len(preds), 'flagged_rate': float(np.mean(preds))}
        for g, preds in sorted(groups.items())
    }


def main():
    parser = argparse.ArgumentParser(description='DS4CS Out-of-Distribution Evaluation')
    parser.add_argument('--num-benign', type=int, default=3000,
                        help='Number of benign HTML pages used in generation')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='Malicious ratio used in dataset generation')
    args = parser.parse_args()

    num_benign = args.num_benign
    ratio_pct = int(args.ratio * 100)

    pipeline_path = MODELS_DIR / f'semantic_pipeline.joblib'
    if not pipeline_path.exists():
        raise SystemExit(f'Missing {pipeline_path}. Run extract_features.py first.')
    pipeline = joblib.load(pipeline_path)

    models_to_evaluate = {
        'xgb_baseline': f'xgb_baseline.joblib',
        'xgb_tuned': f'xgb_tuned.joblib',
        'rf': f'rf_model.joblib',
        'or_ensemble': f'or_ensemble.joblib',
    }

    print(f'\n=== OOD Evaluation (Benign: {num_benign}, Ratio: {ratio_pct}%) ===')
    print('Loading held-out sets...')
    adv_pairs = load_ood_adversarial()
    nat_pairs = load_natural_benign()

    # Load Isolation Forest if exists
    iso_path = MODELS_DIR / f'isolation_forest.joblib'
    iso = joblib.load(iso_path) if iso_path.exists() else None

    # Load Transformer Embedder if config exists
    sbert_config_path = MODELS_DIR / f'sbert_config.joblib'
    sbert_embedder = None
    if sbert_config_path.exists():
        from transformer_embeddings import TransformerEmbedder
        sbert_embedder = TransformerEmbedder()
        print('  Loaded Transformer Embedder for OOD featurization.')

    print('\nFeaturizing OOD adversarial...')
    X_adv, meta_adv = featurize_corpus(adv_pairs, pipeline, iso, sbert_embedder)
    print('Featurizing natural-benign...')
    X_nat, meta_nat = featurize_corpus(nat_pairs, pipeline, iso, sbert_embedder)

    y_adv = np.array([m['label'] for m in meta_adv])
    y_nat = np.array([m['label'] for m in meta_nat])

    full_report = {'adversarial': {}, 'natural_benign': {}}
    for name, fname in models_to_evaluate.items():
        model_path = MODELS_DIR / fname
        if not model_path.exists():
            print(f'  (skip {name}: {model_path} missing)')
            continue
        model = joblib.load(model_path)

        print(f'\n--- {name} on OOD adversarial ---')
        adv_metrics, adv_pred, _ = score_model(model, X_adv, y_adv)
        adv_by_tech = per_group_recall(adv_pred, meta_adv, 'technique')
        adv_metrics['recall_by_ood_technique'] = adv_by_tech
        for k, v in adv_metrics.items():
            if k != 'recall_by_ood_technique':
                print(f'    {k:20s} {v}')
        print('    recall by OOD technique:')
        for tech, info in adv_by_tech.items():
            print(f'      {tech:30s} n={info["n"]:3d}  recall={info["flagged_rate"]:.3f}')
        full_report['adversarial'][name] = adv_metrics

        print(f'\n--- {name} on natural-benign (false-positive test) ---')
        nat_metrics, nat_pred, _ = score_model(model, X_nat, y_nat)
        nat_by_source = per_group_recall(nat_pred, meta_nat, 'source')
        nat_metrics['flagged_by_source'] = nat_by_source
        for k, v in nat_metrics.items():
            if k != 'flagged_by_source':
                print(f'    {k:20s} {v}')
        print('    flagged by source (1 = false alarm):')
        for source, info in nat_by_source.items():
            tick = ' *** FALSE ALARM' if info['flagged_rate'] > 0 else ''
            print(f'      {source:40s}  flagged={info["flagged_rate"]:.2f}{tick}')
        full_report['natural_benign'][name] = nat_metrics

    json_path = RESULTS_DIR / f'ood_evaluation.json'
    with open(json_path, 'w') as f:
        json.dump(full_report, f, indent=2)

    txt_path = RESULTS_DIR / f'ood_evaluation.txt'
    with open(txt_path, 'w') as f:
        f.write(f'DS4CS OOD Evaluation Report ({num_benign} benign, ratio {ratio_pct}%)\n')
        f.write('=' * 40 + '\n\n')
        f.write('Models trained on the 14 in-distribution techniques are evaluated\n')
        f.write('on (a) 150 malicious samples using 8 unseen stealth techniques and\n')
        f.write('(b) hand-crafted realistic benign HTML.\n\n')
        for split in ('adversarial', 'natural_benign'):
            f.write(f'--- {split.upper()} ---\n')
            for name, m in full_report[split].items():
                f.write(f'  {name}\n')
                f.write(f'    recall = {m["recall"]:.4f}  fpr = {m["fpr"]:.4f}  '
                        f'flagged {m["flagged"]}/{m["n"]}\n')
            f.write('\n')
    print(f'\nSaved {json_path} and {txt_path}')

    # Backward compatibility override
    if num_benign == 3000 and ratio_pct == 20:
        import shutil
        shutil.copyfile(json_path, RESULTS_DIR / 'ood_evaluation.json')
        shutil.copyfile(txt_path, RESULTS_DIR / 'ood_evaluation.txt')
        print("Saved backward-compatible OOD evaluation reports.")


if __name__ == '__main__':
    main()
