#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'

sys.path.insert(0, str(BASE_DIR / 'scripts' / 'feature_engineering'))
from extract_features import parse_html_features
from minhash_vectorizer import MinHashVectorizer  # Needed for unpickling

def ingest(html_path, label, is_benign=False):
    html = Path(html_path).read_text(encoding='utf-8', errors='replace')
    
    # 1. Structural features
    feats = parse_html_features(html)
    hidden_text = feats.pop('hidden_text', '')
    
    # 2. LSA embeddings
    pipeline = joblib.load(MODELS_DIR / 'semantic_pipeline.joblib')
    lsa = pipeline.transform([hidden_text])
    for i in range(lsa.shape[1]):
        feats[f'lsa_embed_{i}'] = float(lsa[0, i])
        
    # 3. SBERT embeddings
    sbert_config_path = MODELS_DIR / 'sbert_config.joblib'
    if sbert_config_path.exists():
        from transformer_embeddings import TransformerEmbedder
        embedder = TransformerEmbedder()
        sbert_vec = embedder.encode([hidden_text], show_progress=False)
        for i in range(sbert_vec.shape[1]):
            feats[f'sbert_embed_{i}'] = float(sbert_vec[0, i])
            
    # 4. Add metadata
    feats['Label'] = label
    if is_benign:
        feats['Category'] = 'Benign'
        feats['Technique'] = ''
    else:
        feats['Category'] = 'ActiveLearning'
        feats['Technique'] = 'OOD_MassiveDOM'
    feats['URL'] = Path(html_path).name
    
    # 5. Get CSV columns to ensure order
    train_csv = DATA_DIR / 'features_train.csv'
    df_existing = pd.read_csv(train_csv, nrows=0)
    columns = list(df_existing.columns)
    
    # Fill any missing columns with 0
    row = {}
    for col in columns:
        if col in feats:
            row[col] = feats[col]
        else:
            row[col] = 0
            
    df_new = pd.DataFrame([row])
    
    # Append to CSV
    df_new.to_csv(train_csv, mode='a', header=False, index=False)
    print(f"Successfully ingested {Path(html_path).name} into features_train.csv with Label={label}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to HTML file')
    parser.add_argument('--label', type=int, required=True, help='0 for benign, 1 for malicious')
    args = parser.parse_args()
    ingest(args.path, args.label, is_benign=(args.label == 0))
