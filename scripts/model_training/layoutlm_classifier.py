#!/usr/bin/env python3
"""
DS4CS — LayoutLM Visual-Layout Classifier
==========================================
Uses microsoft/layoutlmv3-base to classify HTML pages by combining text content
with estimated 2D positional layout information.

Since we don't have a real browser renderer, we estimate bounding boxes from DOM
structure:
    - Block elements → vertical stacking (y increases with DOM order)
    - Inline elements → horizontal flow
    - Nesting depth → indentation (x-offset)
    - Hidden elements → special "off-canvas" coordinates (x=999, y=999)

The LayoutLM model is fine-tuned on the training set with a sequence classification
head and outputs a malicious probability that can be used as a standalone classifier
or as an additional feature for the XGBoost ensemble.

Usage:
    python layoutlm_classifier.py [--num-benign 10000] [--ratio 0.2] [--epochs 3]
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
    LayoutLMv3TokenizerFast,
    LayoutLMv3ForSequenceClassification,
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

sys.path.insert(0, str(BASE_DIR / 'scripts' / 'feature_engineering'))
from extract_features import PARSER  # noqa: E402

MODEL_NAME = 'microsoft/layoutlmv3-base'
MAX_LEN = 512
BLOCK_TAGS = frozenset({
    'div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'section', 'article',
    'header', 'footer', 'nav', 'main', 'aside', 'ul', 'ol', 'li', 'table',
    'tr', 'td', 'th', 'form', 'fieldset', 'blockquote', 'pre', 'hr', 'br',
    'dl', 'dt', 'dd', 'figure', 'figcaption', 'details', 'summary',
})
HIDDEN_TAGS = frozenset({'script', 'noscript', 'style', 'template'})


# ============================================================
# Layout Estimation — DOM → Bounding Boxes
# ============================================================
def estimate_layout_boxes(html_content: str, max_words: int = 450):
    """Extract words with estimated 2D bounding boxes from HTML.

    Returns:
        words: list of str
        boxes: list of [x0, y0, x1, y1] normalised to 0-1000 range
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content or '', PARSER)
    words = []
    boxes = []

    y_cursor = 0
    x_cursor = 0
    line_height = 20  # pixels per line

    def _clamp(val, lo=0, hi=1000):
        return max(lo, min(val, hi))

    def _walk(tag, depth=0, is_hidden=False):
        nonlocal y_cursor, x_cursor

        if y_cursor > 950 or len(words) >= max_words:
            return

        if not hasattr(tag, 'children'):
            return

        for child in tag.children:
            if len(words) >= max_words:
                return

            if isinstance(child, str):
                text = child.strip()
                if not text:
                    continue
                for w in text.split()[:10]:  # limit words per text node
                    if len(words) >= max_words:
                        return
                    w_clean = w[:30]  # truncate very long "words"
                    x_indent = min(depth * 30, 200)

                    if is_hidden:
                        # Hidden elements get off-canvas coordinates (within range)
                        box = [900, 900, 999, 999]
                    else:
                        x0 = _clamp(x_indent + x_cursor)
                        x1 = _clamp(x0 + len(w_clean) * 8)
                        y0 = _clamp(y_cursor)
                        y1 = _clamp(y0 + line_height)
                        # Ensure x1 > x0 and y1 > y0
                        if x1 <= x0:
                            x1 = _clamp(x0 + 1)
                        if y1 <= y0:
                            y1 = _clamp(y0 + 1)
                        box = [x0, y0, x1, y1]
                        x_cursor = min(x1 + 5, 990)

                    words.append(w_clean)
                    boxes.append(box)
                continue

            if not hasattr(child, 'name') or child.name is None:
                continue

            tag_name = child.name.lower()

            # Determine if this child is hidden
            child_hidden = is_hidden or tag_name in HIDDEN_TAGS
            if not child_hidden and child.has_attr('type') and child.get('type') == 'hidden':
                child_hidden = True
            if not child_hidden and child.has_attr('style'):
                style = child.get('style', '').lower()
                if any(h in style for h in ['display:none', 'visibility:hidden', 'opacity:0']):
                    child_hidden = True

            # Block elements advance the y cursor
            if tag_name in BLOCK_TAGS:
                y_cursor = min(y_cursor + line_height, 980)
                x_cursor = 0

            _walk(child, depth + 1, child_hidden)

            if tag_name in BLOCK_TAGS:
                y_cursor = min(y_cursor + line_height // 2, 980)
                x_cursor = 0

    _walk(soup, depth=0)

    if not words:
        words = ['[empty]']
        boxes = [[0, 0, 100, 20]]

    return words, boxes


# ============================================================
# Dataset
# ============================================================
class HTMLLayoutDataset(Dataset):
    def __init__(self, html_texts, labels, tokenizer, max_len=MAX_LEN):
        self.html_texts = html_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.html_texts)

    def __getitem__(self, idx):
        html = self.html_texts[idx]
        label = self.labels[idx]

        words, boxes = estimate_layout_boxes(html)

        try:
            encoding = self.tokenizer(
                words,
                boxes=boxes,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
        except Exception:
            # Fallback for edge-case tokenization failures
            encoding = self.tokenizer(
                ['[empty]'],
                boxes=[[0, 0, 100, 20]],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'bbox': encoding['bbox'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# Training Loop
# ============================================================
def train_model(model, train_loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        bbox = batch['bbox'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            labels=labels,
        )

        loss = outputs.loss
        total_loss += loss.item()

        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bbox = batch['bbox'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
            )

            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            preds = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='DS4CS LayoutLM Classifier')
    parser.add_argument('--num-benign', type=int, default=10000)
    parser.add_argument('--ratio', type=float, default=0.2)
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

    print(f'=== LayoutLMv3 Visual-Layout Classifier ===')
    print(f'  Model: {MODEL_NAME}')
    print(f'  Dataset: {num_benign} benign, ratio {ratio_pct}%')

    # Load raw HTML data
    train_file = DATA_DIR / f'train_real_html.csv'
    test_file = DATA_DIR / f'test_real_html.csv'

    if not train_file.exists():
        print(f'Error: {train_file} not found.')
        sys.exit(1)

    print('Loading datasets...')
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # Subsample for tractable training on CPU
    if len(df_train) > args.max_train:
        # Stratified subsample
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

    # Load tokenizer and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')

    print(f'  Loading LayoutLMv3 tokenizer and model...')
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(MODEL_NAME, apply_ocr=False)
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    # Create datasets
    print('  Creating layout datasets...')
    train_dataset = HTMLLayoutDataset(train_htmls, train_labels, tokenizer)
    test_dataset = HTMLLayoutDataset(test_htmls, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    # Training
    print(f'\n--- Training for {args.epochs} epochs ---')
    for epoch in range(1, args.epochs + 1):
        train_model(model, train_loader, optimizer, scheduler, device, epoch)

        # Evaluate after each epoch
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

    # Save model
    model_save_path = MODELS_DIR / f'layoutlm'
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f'\nSaved LayoutLMv3 model → {model_save_path}')

    # Final metrics report
    preds, labels, probs = evaluate_model(model, test_loader, device)
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)

    report = {
        'Model': 'LayoutLMv3',
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

    report_path = RESULTS_DIR / f'layoutlm_metrics.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Saved LayoutLMv3 report → {report_path}')

    print('\n--- LayoutLMv3 Final Results ---')
    for k, v in report.items():
        print(f'  {k}: {v}')

    print('\nLayoutLM Classifier Complete.')


if __name__ == '__main__':
    main()
