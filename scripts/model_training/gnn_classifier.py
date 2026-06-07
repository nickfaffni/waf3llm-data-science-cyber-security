#!/usr/bin/env python3
"""
DS4CS — GNN DOM-Tree Classifier
==========================================
Parses HTML into a PyTorch Geometric Graph where Nodes = Tags and Edges = Nesting.
"""

import argparse
import sys
import json
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Small vocabulary for typical tags. Everything else maps to UNKNOWN.
TAG_VOCAB = [
    'div', 'span', 'script', 'a', 'p', 'img', 'html', 'body', 'head', 'style', 
    'iframe', 'form', 'input', 'button', 'ul', 'li', 'noscript', 'meta', 'link'
]
TAG_TO_ID = {tag: i for i, tag in enumerate(TAG_VOCAB)}
UNKNOWN_ID = len(TAG_VOCAB)


def html_to_graph(html_content):
    """Parses HTML into PyTorch Geometric Data format."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content or "", "html.parser")
    
    nodes = []
    edges = []
    
    def traverse(element, parent_id=None):
        if not hasattr(element, 'name') or element.name is None:
            return
            
        node_id = len(nodes)
        tag_name = str(element.name).lower()
        tag_idx = TAG_TO_ID.get(tag_name, UNKNOWN_ID)
        
        nodes.append([tag_idx])
        
        if parent_id is not None:
            edges.append([parent_id, node_id])
            edges.append([node_id, parent_id])
            
        for child in element.children:
            traverse(child, node_id)
            
    traverse(soup)
    
    if not nodes:
        nodes = [[UNKNOWN_ID]]
        edges = []
        
    x = torch.tensor(nodes, dtype=torch.long)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    return Data(x=x, edge_index=edge_index)


class HTMLGraphDataset(Dataset):
    def __init__(self, htmls, labels):
        super().__init__(None, None, None)
        self.htmls = htmls
        self.labels = labels
        self.data_list = []
        print(f"    Parsing {len(htmls)} HTML documents into Graphs...")
        for h, l in zip(htmls, labels):
            data = html_to_graph(h)
            data.y = torch.tensor([l], dtype=torch.float)
            self.data_list.append(data)
            
    def len(self):
        return len(self.data_list)
        
    def get(self, idx):
        return self.data_list[idx]


class GNNClassifier(nn.Module):
    def __init__(self, num_tags, hidden_channels):
        super().__init__()
        self.embedding = nn.Embedding(num_tags + 1, hidden_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.embedding(x.squeeze(-1))
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        probs = torch.sigmoid(out).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(batch.y.cpu().numpy())
        
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description='DS4CS GNN Classifier')
    parser.add_argument('--num-benign', type=int, default=10000)
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--hidden', type=int, default=64)
    args = parser.parse_args()

    print(f'=== GNN DOM-Tree Classifier ===')
    
    train_file = DATA_DIR / f'train_real_html_{args.num_benign}_r{int(args.ratio*100)}.csv'
    test_file = DATA_DIR / f'test_real_html_{args.num_benign}_r{int(args.ratio*100)}.csv'

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    train_dataset = HTMLGraphDataset(df_train['HTML_Content'].fillna('').tolist(), df_train['Label'].tolist())
    test_dataset = HTMLGraphDataset(df_test['HTML_Content'].fillna('').tolist(), df_test['Label'].tolist())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNClassifier(num_tags=len(TAG_VOCAB), hidden_channels=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        preds, labels, probs = evaluate(model, test_loader, device)
        acc = accuracy_score(labels, preds)
        rec = recall_score(labels, preds, zero_division=0)
        auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, Test Recall: {rec:.4f}, Test AUC: {auc:.4f}')

    model_save_path = MODELS_DIR / f'gnn_{args.num_benign}_r{int(args.ratio*100)}.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f'Saved GNN model → {model_save_path}')

    preds, labels, probs = evaluate(model, test_loader, device)
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)

    report = {
        'Model': 'GNN',
        'Accuracy': float(accuracy_score(labels, preds)),
        'Precision': float(precision_score(labels, preds, zero_division=0)),
        'Recall': float(recall_score(labels, preds, zero_division=0)),
        'TPR': float(tpr),
        'FPR': float(fpr),
        'F1_Score': float(f1_score(labels, preds, zero_division=0)),
        'ROC_AUC': float(roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0),
    }

    report_path = RESULTS_DIR / f'gnn_metrics_{args.num_benign}_r{int(args.ratio*100)}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print('\n--- GNN Final Results ---')
    for k, v in report.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
