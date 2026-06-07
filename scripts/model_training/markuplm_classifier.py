#!/usr/bin/env python3
"""
DS4CS — MarkupLM HTML-Native Classifier
==========================================
Uses microsoft/markuplm-base to classify HTML pages by natively embedding
the XPath structure and relationships of HTML tags alongside the text.

Usage:
    python markuplm_classifier.py [--num-benign 10000] [--ratio 0.1] [--epochs 3]
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    MarkupLMProcessor,
    MarkupLMForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings('ignore')

csv.field_size_limit(sys.maxsize)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'microsoft/markuplm-base'
MAX_LEN = 512

# ============================================================
# Dataset
# ============================================================
class HTMLMarkupDataset(Dataset):
    def __init__(self, html_texts, labels, processor, max_len=MAX_LEN):
        self.html_texts = html_texts
        self.labels = labels
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.html_texts)

    def __getitem__(self, idx):
        html = self.html_texts[idx]
        label = self.labels[idx]

        # MarkupLM natively parses HTML and extracts XPaths
        try:
            encoding = self.processor(
                html,
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )
        except Exception:
            # Fallback for completely malformed HTML strings that crash bs4
            encoding = self.processor(
                "<html><body></body></html>",
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item


# ============================================================
# Training Loop
# ============================================================
def train_model(model, train_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)

        loss = outputs.loss
        total_loss += loss.item()

        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch['labels']).sum().item()
        total += len(batch['labels'])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if batch_idx % 50 == 0:
            print(f'    Batch {batch_idx}/{len(train_loader)} — loss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    acc = correct / total
    print(f'  Epoch {epoch}: avg_loss={avg_loss:.4f}, train_acc={acc:.4f}')
    return avg_loss


def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop('labels')
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            preds = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='DS4CS MarkupLM Classifier')
    parser.add_argument('--num-benign', type=int, default=10000)
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-train', type=int, default=2000,
                        help='Max training samples (subsample for speed on CPU)')
    parser.add_argument('--max-test', type=int, default=500,
                        help='Max test samples')
    args = parser.parse_args()

    num_benign = args.num_benign
    ratio_pct = int(args.ratio * 100)

    print(f'=== MarkupLM HTML-Native Classifier ===')
    print(f'  Model: {MODEL_NAME}')
    print(f'  Dataset: {num_benign} benign, ratio {ratio_pct}%')

    train_file = DATA_DIR / f'train_real_html_{num_benign}_r{ratio_pct}.csv'
    test_file = DATA_DIR / f'test_real_html_{num_benign}_r{ratio_pct}.csv'

    if not train_file.exists():
        print(f'Error: {train_file} not found.')
        sys.exit(1)

    print('Loading datasets...')
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # Subsample for tractable training on CPU
    if len(df_train) > args.max_train:
        mal = df_train[df_train['Label'] == 1]
        ben = df_train[df_train['Label'] == 0]
        mal_n = min(len(mal), args.max_train // 5)
        ben_n = args.max_train - mal_n
        df_train = pd.concat([
            mal.sample(n=mal_n, random_state=42),
            ben.sample(n=min(ben_n, len(ben)), random_state=42)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f'  Subsampled training to {len(df_train)} (mal={mal_n}, ben={len(df_train)-mal_n})')

    if len(df_test) > args.max_test:
        df_test = df_test.sample(n=args.max_test, random_state=42).reset_index(drop=True)
        print(f'  Subsampled test to {len(df_test)}')

    train_htmls = df_train['HTML_Content'].fillna('').tolist()
    train_labels = df_train['Label'].astype(int).tolist()
    test_htmls = df_test['HTML_Content'].fillna('').tolist()
    test_labels = df_test['Label'].astype(int).tolist()

    print(f'  Train: {len(train_htmls)} samples, Test: {len(test_htmls)} samples')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')

    print(f'  Loading MarkupLM processor and model...')
    processor = MarkupLMProcessor.from_pretrained(MODEL_NAME)
    processor.parse_html = True
    model = MarkupLMForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    print('  Creating layout datasets...')
    train_dataset = HTMLMarkupDataset(train_htmls, train_labels, processor)
    test_dataset = HTMLMarkupDataset(test_htmls, test_labels, processor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    print(f'\n--- Training for {args.epochs} epochs ---')
    for epoch in range(1, args.epochs + 1):
        train_model(model, train_loader, optimizer, scheduler, device, epoch)

        preds, labels, probs = evaluate_model(model, test_loader, device)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.0
        print(f'  Eval — Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}')

    model_save_path = MODELS_DIR / f'markuplm_{num_benign}_r{ratio_pct}'
    model.save_pretrained(model_save_path)
    processor.save_pretrained(model_save_path)
    print(f'\nSaved MarkupLM model → {model_save_path}')

    preds, labels, probs = evaluate_model(model, test_loader, device)
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)

    report = {
        'Model': 'MarkupLM',
        'Accuracy': float(accuracy_score(labels, preds)),
        'Precision': float(precision_score(labels, preds, zero_division=0)),
        'Recall': float(recall_score(labels, preds, zero_division=0)),
        'TPR': float(tpr),
        'FPR': float(fpr),
        'F1_Score': float(f1_score(labels, preds, zero_division=0)),
        'ROC_AUC': float(roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0),
        'Epochs': args.epochs,
        'Train_Samples': len(train_htmls),
        'Test_Samples': len(test_htmls),
    }

    report_path = RESULTS_DIR / f'markuplm_metrics_{num_benign}_r{ratio_pct}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Saved MarkupLM report → {report_path}')

    print('\n--- MarkupLM Final Results ---')
    for k, v in report.items():
        print(f'  {k}: {v}')

    print('\nMarkupLM Classifier Complete.')


if __name__ == '__main__':
    main()
