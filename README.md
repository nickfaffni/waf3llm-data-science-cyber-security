<div align="center">
  
# WAF3LLM 🛡️🕸️
**Detection of Web-Based Indirect Prompt Injection Attacks**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-red.svg)](https://xgboost.readthedocs.io/)

*A proactive "shift-left" defense mechanism for detecting Indirect Prompt Injection (IPI) attacks targeting LLM-powered web scraping agents.*

</div>

---

## 🎯 The Problem: Indirect Prompt Injections (IPI)

As Large Language Models (LLMs) are increasingly integrated into web-browsing agents (e.g., Perplexity, OpenAI Search, custom enterprise scrapers), they become vulnerable to **Indirect Prompt Injections**. Malicious actors embed hidden text in legitimate-looking websites using CSS tricks (`display:none`, `opacity:0`), invisible inputs, or HTML comments. When the LLM agent scrapes the page, it unknowingly ingests the payload, leading to data exfiltration, context hijacking, or malicious roleplay.

## 🛡️ The Solution: WAF3LLM

**WAF3LLM** acts as a Web Application Firewall specifically for LLM Agents. Instead of trying to filter the LLM's output *after* the prompt is injected, WAF3LLM statically analyzes the HTML source **before** it reaches the agent.

### Core Novelty
By simulating both the **human-visible DOM** and the **LLM-readable DOM**, WAF3LLM detects the structural and semantic discrepancies that characterize real-world stealth injections. If the hidden content deviates significantly from the visible content, the WAF blocks the scraping operation.

---

## 🧠 Architecture & Methodology

WAF3LLM uses a dual-modality approach, combining **Structural DOM Features** with **Semantic Embeddings**:

1. **Structural Feature Engineering**: 
   - Internal `<style>` blocks parsed with a declaration-list parser (detecting `font-size:0`, off-screen positioning, `clip-path` tricks).
   - Event handlers (inline `on*` attributes, `javascript:` hrefs).
   - DOM branching factors, hidden-to-visible text ratios, and depth variances.
2. **Semantic Text Embeddings**:
   - **LSA (Latent Semantic Analysis)**: TF-IDF bigrams + TruncatedSVD for lightweight semantic grouping.
   - **Sentence-BERT (SBERT)**: Advanced transformer-based representations of the hidden payload text.
3. **Machine Learning Ensembles**:
   - Tuned **XGBoost** and **Random Forest** models operating in a Hard-OR ensemble to maximize recall.

### 🔄 Active Learning Pipeline
OOD (Out-Of-Distribution) adversarial evasions—such as massive benign Real Estate websites loaded with naturally hidden CSS—can sometimes cause false negatives. WAF3LLM features a closed-loop **Active Learning Ingester** (`ingest_active_learning.py`). False negatives are pushed back into the feature matrix and automatically rebalanced via SMOTE oversampling, instantly patching zero-day structural bypasses.

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/nickfaffni/waf3llm-data-science-cyber-security.git
cd waf3llm-data-science-cyber-security
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Inference (Using the WAF)

To score a single HTML page and determine if it contains an Indirect Prompt Injection:

```bash
python scripts/inference/predict.py data/examples/example_injected_page.html --model xgb_tuned
```
**Output:**
```text
MALICIOUS  (P(malicious) = 0.9257)
```

### 3. Pipeline Execution

To recreate the datasets, extract features, and train the models from scratch:

```bash
# 1. Generate dataset (streams real HTML from C4/Common Crawl via HuggingFace)
python scripts/data_generation/generate_dataset_real_html.py

# 2. Extract structural and semantic features
python scripts/feature_engineering/extract_features.py

# 3. Train + evaluate ML models (baseline & tuned)
python scripts/model_training/train_evaluate.py

# 4. Generate the out-of-distribution (OOD) adversarial set
python scripts/data_generation/generate_ood_set.py

# 5. Evaluate trained models on OOD samples
python scripts/evaluation/evaluate_ood.py
```

### 4. Continuous Active Learning

If an injected page bypasses the WAF (False Negative):

```bash
# Ingest the misclassified HTML and label it as malicious (1)
python scripts/data_generation/ingest_active_learning.py path/to/bypassed_page.html --label 1

# Retrain the WAF models to close the vulnerability
python scripts/model_training/train_evaluate.py
```

---

## 📊 Evaluation & Performance

WAF3LLM was trained on 10,000 real-world DOM trees from the C4/Common Crawl dataset, injected with 30 distinct payloads masked by 14 stealth HTML/CSS techniques.

- **Detection Rate**: **78.8%** recall (Tuned XGBoost) / **81.0%** recall (Hard-OR Ensemble) against high-obfuscation injections, with **92.3% precision**.
- **Natural-Benign FPR**: Maintained at an operational ceiling of **0.00%** (Random Forest) and **1.65%** (Tuned XGBoost) on complex, noisy single-page applications.
- **Leave-One-Technique-Out (LOTO)**: Demonstrated robust Zero-Day generalizability by maintaining **74.8% mean recall** against stealth techniques explicitly withheld from the training set (e.g. capturing unseen `anchor_js` hidden links at **86.8% recall**).

---

## 📄 References & Bibliography

This research builds upon the latest findings in LLM Agent Security:
- **MUZZLE**: Adaptive Agentic Red-Teaming of Web Agents Against IPI Attacks (2026)
- **WebSentinel**: Detecting and Localizing Prompt Injection Attacks for Web Agents (2026)
- **Dual-Modality Multi-Stage Adversarial Safety Training** (2026)

*Developed for the Data Science for Cyber-Security 2026 Research Initiative.*
