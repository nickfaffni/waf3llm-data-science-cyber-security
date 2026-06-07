#!/usr/bin/env python3
"""
DS4CS — Phase 4: Generate Essential Figures
===========================================
Generates publication-quality, high-contrast visual figures for academic presentation:
1. ROC Curves (comparing RF, XGB Baseline, XGB Tuned, LR, and Weighted Soft Vote).
2. Feature Importance Comparative Bar Chart (showing RF vs. Tuned XGB top features).
3. Data Distribution Statistical Analysis (2x2 Boxplots/Violinplots of structural anomalies).

All figures are saved to the `figures/` directory.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# ============================================================
# Paths & Settings
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
FIGURES_DIR = BASE_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Aesthetic Styling (Dark Mode Theme)
plt.style.use('dark_background')
sns.set_theme(style="dark", palette="muted")

# Override dark theme background colors for extreme premium look
plt.rcParams.update({
    'grid.color': '#1f2937',
    'axes.facecolor': '#090f1d',
    'figure.facecolor': '#050811',
    'axes.edgecolor': '#1f2937',
    'text.color': '#e5e7eb',
    'axes.labelcolor': '#9ca3af',
    'xtick.color': '#9ca3af',
    'ytick.color': '#9ca3af',
    'font.family': 'sans-serif',
})

EXCLUDE_COLS = {'Label', 'Category', 'Technique', 'URL', 'HTML_Content', 'Source', 'Payload'}

# ============================================================
# Load Datasets & Models
# ============================================================
print("Loading datasets...")
train_df = pd.read_csv(DATA_DIR / 'features_train_10000_r20.csv')
test_df = pd.read_csv(DATA_DIR / 'features_test_10000_r20.csv')

feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]
X_test = test_df[feature_cols].fillna(0)
y_test = test_df['Label']

print("Loading models...")
models = {
    'Random Forest': joblib.load(MODELS_DIR / 'rf_model_10000_r20.joblib'),
    'XGBoost (Baseline)': joblib.load(MODELS_DIR / 'xgb_baseline_10000_r20.joblib'),
    'XGBoost (Tuned)': joblib.load(MODELS_DIR / 'xgb_tuned_10000_r20.joblib'),
    'Logistic Regression': joblib.load(MODELS_DIR / 'lr_model_10000_r20.joblib'),
    'Weighted Soft Vote': joblib.load(MODELS_DIR / 'weighted_ensemble_10000_r20.joblib'),
}

# ============================================================
# 1. Generate & Plot ROC Curves
# ============================================================
print("\n--- Generating ROC Curves ---")
plt.figure(figsize=(10, 8), dpi=300)

# Colors matching the presentation template
colors = {
    'Random Forest': '#10b981',       # Emerald Green
    'XGBoost (Baseline)': '#a855f7',  # Purple
    'XGBoost (Tuned)': '#06b6d4',     # Cyan
    'Logistic Regression': '#f43f5e', # Rose Red
    'Weighted Soft Vote': '#eab308',  # Amber Yellow
}

for name, model in models.items():
    print(f"Evaluating ROC for {name}...")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback for general estimators
        probs = model.decision_function(X_test) if hasattr(model, "decision_function") else model.predict(X_test)
        
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color=colors[name], lw=2.5, label=f'{name} (AUC = {roc_auc:.4f})')

# Random Guess Reference Line
plt.plot([0, 1], [0, 1], color='#4b5563', lw=1.5, linestyle='--', label='Random Guess')

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate (FPR)', fontsize=11, labelpad=10)
plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=11, labelpad=10)
plt.title('Receiver Operating Characteristic (ROC) Curves\n[URL-Grouped Clean Split Evaluation]', fontsize=13, fontweight='bold', pad=15)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower right', frameon=True, facecolor='#090f1d', edgecolor=(1.0, 1.0, 1.0, 0.08), fontsize=10)

roc_path = FIGURES_DIR / 'roc_curves.png'
plt.savefig(roc_path, bbox_inches='tight', facecolor='#050811')
plt.close()
print(f"Saved ROC curve figure to: {roc_path}")

# ============================================================
# 2. Generate Feature Importance Map
# ============================================================
print("\n--- Generating Feature Importance Standings ---")
rf_model = models['Random Forest']
xgb_model = models['XGBoost (Tuned)']

rf_importances = rf_model.feature_importances_
xgb_importances = xgb_model.feature_importances_

# Map to feature names
rf_series = pd.Series(rf_importances, index=feature_cols).sort_values(ascending=False)
xgb_series = pd.Series(xgb_importances, index=feature_cols).sort_values(ascending=False)

# Select the union of the top 8 features for comparative visualization
top_features = list(pd.concat([rf_series.head(8), xgb_series.head(8)]).index.unique())

df_imp = pd.DataFrame({
    'Feature': top_features * 2,
    'Importance': list(rf_series[top_features]) + list(xgb_series[top_features]),
    'Classifier': ['Random Forest'] * len(top_features) + ['Tuned XGBoost'] * len(top_features)
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(11, 7), dpi=300)
ax = sns.barplot(
    data=df_imp, 
    y='Feature', 
    x='Importance', 
    hue='Classifier', 
    palette={'Random Forest': '#10b981', 'Tuned XGBoost': '#06b6d4'},
    edgecolor=(1.0, 1.0, 1.0, 0.08)
)

# Customizing plot details
plt.title('Top Feature Importance Standings\n[Random Forest vs. Tuned XGBoost Weights]', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Relative Feature Importance Weight', fontsize=11, labelpad=10)
plt.ylabel('Engineered DOM Feature name', fontsize=11, labelpad=10)
plt.grid(True, axis='x', linestyle=':', alpha=0.6)
plt.legend(loc='lower right', frameon=True, facecolor='#090f1d', edgecolor=(1.0, 1.0, 1.0, 0.08), fontsize=10)

# Add value annotations
for p in ax.patches:
    width = p.get_width()
    if width > 0.002: # Annotate significant bars
        ax.text(
            width + 0.0005, 
            p.get_y() + p.get_height() / 2, 
            f'{width:.4f}', 
            ha='left', 
            va='center', 
            fontsize=8, 
            color='#9ca3af'
        )

imp_path = FIGURES_DIR / 'feature_importance_heatmap.png'
plt.savefig(imp_path, bbox_inches='tight', facecolor='#050811')
plt.close()
print(f"Saved Feature Importance standings to: {imp_path}")

# ============================================================
# 3. Generate Data Distribution Statistical Analysis
# ============================================================
print("\n--- Generating Data Distribution Statistical Charts ---")

# Set up a 2x2 grid of subplots for statistical distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300, facecolor='#050811')
fig.suptitle('DOM Structural Anomaly Distribution Analysis\n[Benign C4 Web Pages vs. Malicious IPI Clones]', fontsize=15, fontweight='bold', y=0.98)

dist_features = [
    ('hidden_to_visible_ratio', 'Hidden-to-Visible Text Ratio', True),  # Use Log Scale
    ('hidden_element_count', 'Hidden Elements Count', False),
    ('css_trick_count', 'CSS Camouflage Styles Count', False),
    ('script_count', 'Script Containers Count', False)
]

labels_map = {0: 'Benign (C4 Page)', 1: 'Malicious (IPI Clone)'}
plot_df = train_df.copy()
plot_df['Class Label'] = plot_df['Label'].map(labels_map)

# Color palette: emerald green for benign, rose red for malicious
colors_dist = {'Benign (C4 Page)': '#10b981', 'Malicious (IPI Clone)': '#f43f5e'}

for idx, (col, title, use_log) in enumerate(dist_features):
    ax = axes[idx // 2, idx % 2]
    
    # Generate Boxenplot or Violinplot for clean statistical comparison
    sns.violinplot(
        data=plot_df,
        x='Class Label',
        y=col,
        ax=ax,
        hue='Class Label',
        palette=colors_dist,
        dodge=False,
        inner='quartile',
        linewidth=1.2,
        density_norm='width'
    )
    
    ax.set_title(title, fontsize=12, fontweight='semibold', pad=10)
    ax.set_xlabel('')
    ax.set_ylabel('Feature Value', fontsize=10)
    
    if use_log:
        ax.set_yscale('log')
        ax.set_ylabel('Feature Value (Log Scale)', fontsize=10)
        
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.legend().remove() # Hue legend is redundant due to X ticks

plt.subplots_adjust(hspace=0.25, wspace=0.22)
dist_path = FIGURES_DIR / 'data_distribution_analysis.png'
plt.savefig(dist_path, bbox_inches='tight', facecolor='#050811')
plt.close()
print(f"Saved Data Distribution charts to: {dist_path}")

print("\nAll essential figures generated successfully inside the 'figures/' folder!")
