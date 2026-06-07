#!/usr/bin/env python3
"""
DS4CS — Text-to-Image CNN Classifier
=====================================
Converts HTML text streams into 2D character-intensity pixel maps (images)
and trains a PyTorch CNN to detect indirect prompt injections.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Paths & Config
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR / 'scripts' / 'feature_engineering'))
from extract_features import parse_html_features  # noqa: E402

RANDOM_STATE = 42
GRID_SIZE = 64  # 64 x 64 = 4096 characters
BATCH_SIZE = 64
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Text-to-Image Helper
# ============================================================
def text_to_image(text: str) -> np.ndarray:
    """Map first 4096 chars of text to a 64x64 float grid normalized to [0, 1]."""
    arr = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
    # Get ascii values for the first 4096 characters
    chars = [ord(c) for c in text[: GRID_SIZE * GRID_SIZE]]
    arr[: len(chars)] = chars
    # Normalize to [0, 1]
    arr = arr / 255.0
    return arr.reshape(1, GRID_SIZE, GRID_SIZE)


def prepare_dataset(filepath: Path) -> TensorDataset:
    print(f"Loading and processing {filepath.name}...")
    df = pd.read_csv(filepath)
    images = []
    labels = []
    for idx, row in enumerate(df.itertuples(index=False)):
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(df)} HTML layouts...")
        html = str(getattr(row, 'HTML_Content', '') or '')
        feats = parse_html_features(html)
        hidden_text = feats.get('hidden_text', '')
        
        img = text_to_image(hidden_text)
        images.append(img)
        labels.append(int(getattr(row, 'Label', 0)))
        
    X = torch.tensor(np.stack(images), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(X, y)


# ============================================================
# CNN Model Definition
# ============================================================
class HTMLCNN(nn.Module):
    def __init__(self):
        super(HTMLCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# Main Trainer
# ============================================================
def main():
    print(f"=== CNN Training on {DEVICE} ===")
    
    train_file = DATA_DIR / 'train_real_html_10000_r20.csv'
    test_file = DATA_DIR / 'test_real_html_10000_r20.csv'
    
    if not train_file.exists() or not test_file.exists():
        print("Error: train/test files not found. Run Phase 1 generation first.")
        sys.exit(1)
        
    # Process dataset into visual matrices
    train_ds = prepare_dataset(train_file)
    test_ds = prepare_dataset(test_file)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = HTMLCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train Loop
    print("\nStarting CNN Model training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"  Epoch {epoch}/{EPOCHS} — Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
        
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    
    acc = (tp + tn) / len(all_labels)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    print("\n--- CNN Classifier Test Evaluation ---")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    
    # Save Model state
    model_path = MODELS_DIR / 'cnn_model_10000_r20.pt'
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved PyTorch CNN weights to {model_path}")
    
    # Write report
    report_path = RESULTS_DIR / 'cnn_metrics_report.json'
    import json
    with open(report_path, 'w') as f:
        json.dump({
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1_Score': float(f1),
        }, f, indent=2)
    print(f"Saved CNN report → {report_path}")
    
    print("CNN Training Complete.")


if __name__ == '__main__':
    main()
