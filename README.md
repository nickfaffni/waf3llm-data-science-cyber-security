# DS4CS — Detection of Web-Based Indirect Prompt Injection Attacks

> **Course:** Data Science for Cyber-Security 2026  
> **Student:** Nick Gaffni (nikitaa@post.bgu.ac.il)  
> **Status:** Phases 1–3 Complete — Phase 4 (Presentation Materials) In Progress

---

## 🎯 Project Overview

This project implements a **proactive "shift-left" defense mechanism** for detecting **Indirect Prompt Injection (IPI)** attacks targeting LLM-powered web agents. Rather than filtering outputs post-inference, this approach stops attacks at the **web-scraping layer** by analyzing the discrepancy between visible text and hidden DOM elements (e.g., `display:none`, CSS tricks, HTML comments) to catch real-world stealth injection techniques.

### Core Novelty

Static analysis of a website's HTML to compare what a **human sees** (visible text) vs. what the **LLM scraper ingests** (hidden DOM elements, metadata, comments) — enabling detection *before* the LLM ever processes the injected content.

### Learning Task

**Supervised Binary Classification**: Benign (0) vs. Malicious (1) web pages.

---

## 📁 Project Structure

```
DS4CS/
├── README.md
├── TODO.md                              # Phase 4 submission checklist
├── requirements.txt                     # Pinned dependencies
├── .gitignore
│
├── scripts/                             # All Python source
│   ├── data_generation/
│   │   ├── generate_dataset_real_html.py  # C4 stream + 14 stealth injectors, URL-group split before injection
│   │   └── generate_ood_set.py            # 150 adversarial samples using 8 stealth techniques NOT in training
│   ├── feature_engineering/
│   │   └── extract_features.py            # DOM parsing (incl. <style> blocks + event handlers) + LSA Pipeline
│   ├── model_training/
│   │   ├── train_evaluate.py              # RF, XGB (baseline+tuned), LR, Hard-OR + Weighted Soft-Vote; true LOTO + PR-curve sweep
│   │   ├── cnn_text_to_image.py           # PyTorch Conv2D HTML ASCII-intensity representation model
│   │   ├── jaccard_similarity.py          # DOM structural template similarity classifier
│   │   └── waf3llm.py                     # WAF3LLM multi-layer operational defense & threshold sweeping
│   ├── evaluation/
│   │   ├── evaluate_ood.py                # Scores all trained models on OOD adversarial + natural-benign fixtures
│   │   ├── explain_models.py              # TreeSHAP feature explanations and plotting
│   │   ├── generate_essential_figures.py  # Generates project progression figures
│   │   └── unsupervised_clustering.py     # K-Means/DBSCAN layout clustering & t-SNE projection
│   └── inference/
│       └── predict.py                     # End-to-end scoring for a single HTML file
│
├── tests/
│   └── test_features.py                 # 14-technique + CSS bypass regression tests (20 tests)
│
├── data/                                # Datasets (gitignored — regenerate with the data-gen script)
│   ├── features_train_10000_r20.csv       # 80% training split features
│   ├── features_test_10000_r20.csv        # 20% test split features
│   ├── sample_real_html_10000_r20.csv     # Combined dataset (10,000 rows)
│   ├── benign_real_html_8000.csv          # Raw benign C4 sample (8,000 rows)
│   ├── examples/                          # 10 hand-picked HTML fixtures for predict.py demos
│   └── ood/                               # Out-of-distribution evaluation sets
│       ├── ood_adversarial.csv              # 150 malicious samples, 8 UNSEEN stealth techniques
│       └── natural_benign/*.html            # 10 realistic SPA / form / docs / blog fixtures (FPR test)
│
├── models/                              # Trained artefacts (gitignored — joblib pipelines)
├── results/                             # Metrics + confusion matrices (gitignored)
│
├── docs/                                # Course documents + project narrative
│   ├── Master Project Roadmap_ DS4CS.docx
│   ├── 0_Instructions_For_Students_*.pdf
│   ├── ML_Progression_Report.md
│   ├── Attack_Flow_Diagrams.md
│   ├── Slide_Deck_Template.md
│   ├── Kahoot_Quiz.md
│   └── ipi_taxonomy.md                  # Comprehensive taxonomy of 30 payloads and 14 stealth techniques
│
├── figures/                             # Architecture / attack-flow PNGs for the slide deck
├── literature_review/                   # Research papers + IPI_Research_Bibliography.bib
└── lectures/                            # Course lecture recordings (3 MP4s)
```

---

## 🔬 Methodology

### Data Pipeline

1. **Benign Source**: 10,000 real HTML web pages from **C4/Common Crawl** (`bs-modeling-metadata/c4-en-html-with-metadata` on HuggingFace) — genuine web pages with complex DOM structures (avg 50-700+ divs, scripts, CSS, comments)
2. **Attack Payloads**: 30 payloads across **10 IPI categories**:
   - Direct Override, Roleplay Injection, Context Injection
   - Encoding Obfuscation, Camouflage Comment, Hidden Attribute
   - Script Tag Obfuscation, CSS Injection, JavaScript Event
   - Data Exfiltration Attempt
3. **Stealth Injection**: Each benign page is cloned and injected with a random payload using one of **14 stealth techniques**:
   - `display:none` divs, `visibility:hidden`, `opacity:0`, off-screen positioning
   - HTML comments, hidden inputs, meta tags, `<noscript>`, `<script type="text/plain">`
   - Zero-size divs, data URI images, SVG/body `onload`, JavaScript anchors
4. **Final Dataset**: 10,000 samples (8,000 benign + 2,000 malicious = 80/20 class ratio), split 80/20 train/test

### Feature Engineering (Phase 2)

| Feature Type | Description |
|:---|:---|
| **Structural** | `hidden_element_count`, `css_trick_count`, `comment_count`, `hidden_input_count`, `script_count`, `event_handler_count`, `hidden_text_length`, `visible_text_length`, `hidden_to_visible_ratio`, `has_visible_text` |
| **CSS Parsing** | Internal `<style>` blocks parsed with declaration-list parser; handles `/*comments*/`, `!important`, zero-size boxes, `font-size:0`, off-screen positioning, `clip-path` tricks |
| **Event Handlers** | Inline `on*` attributes and `javascript:` href values captured into the hidden stream |
| **Semantic** | TF-IDF bigrams + TruncatedSVD (50 LSA components), wrapped in a serialized `sklearn.Pipeline` for inference |
| **DOM Parsing** | Two streams — *visible* (what the human sees) vs. *hidden* (where the LLM payload lives) |

### Models (Phase 3)

- **Random Forest** — baseline with `class_weight='balanced'`
- **XGBoost** — baseline + tuned (RandomizedSearchCV, both with `scale_pos_weight`)
- **Logistic Regression** — linear baseline (StandardScaler + balanced weights)
- **Hard-OR Ensemble** — recall-maximizing operational choice for the WAF
- **Weighted Soft-Vote Ensemble** — weights ∝ validation PR-AUC
- Metrics: Accuracy, Precision, Recall, F1, **PR-AUC**, **ROC-AUC**, per-category recall, recall-by-technique, true leave-one-technique-out (retrain 14×), threshold sweep

### Out-of-Distribution Evaluation (Iteration 4)

- **True LOTO**: retrain tuned XGB 14× excluding one technique at a time → mean recall on unseen techniques: **52.3%**
- **OOD adversarial set**: 150 samples using 8 stealth techniques never injected during training (class-based `<style>` hide, text-indent offscreen, color camouflage, transform translate, CSS pseudo-element content, `aria-hidden`, HTML-entity-encoded visible text, `<template>` tag) → tuned XGB recall **55.3%**
- **Natural-benign FPR**: 10 hand-crafted realistic pages (Bootstrap modal, CSRF form, dropdown nav, tab panels, lazy-load gallery, accessibility skip-link, React SPA shell, e-commerce product, docs sidebar, blog) → Random Forest **0% FPR**; tuned XGB **60% FPR** (the operational deployment ceiling — see [docs/ML_Progression_Report.md](docs/ML_Progression_Report.md) §4c)

---

## 🚀 Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (pinned in requirements.txt)
pip install -r requirements.txt

# 1. Generate dataset (streams real HTML from C4/Common Crawl via HuggingFace)
python scripts/data_generation/generate_dataset_real_html.py

# 2. Extract features (structural + LSA, saves models/semantic_pipeline.joblib)
python scripts/feature_engineering/extract_features.py

# 3. Train + evaluate ML models (baseline & tuned)
python scripts/model_training/train_evaluate.py

# 4. Run unsupervised clustering & visualization
python scripts/evaluation/unsupervised_clustering.py

# 5. Train text-to-image PyTorch CNN layout model
python scripts/model_training/cnn_text_to_image.py

# 6. Run Explainable AI (SHAP feature importance plots)
python scripts/evaluation/explain_models.py

# 7. Execute WAF3LLM multi-layer operational pipeline & threshold sweep
python scripts/model_training/waf3llm.py

# 8. Generate the out-of-distribution adversarial set (8 unseen stealth techniques)
python scripts/data_generation/generate_ood_set.py

# 9. Evaluate trained models on OOD adversarial + natural-benign fixtures
python scripts/evaluation/evaluate_ood.py

# 10. Score a single HTML file at inference time
python scripts/inference/predict.py path/to/page.html --model xgb_tuned

# 11. Run the unit test suite
python -m pytest tests/ -v
```

---

## 📊 Current Progress

| Phase | Status | Description |
|:---:|:---:|:---|
| **1.1** | ✅ | Sourced 10,000 real HTML web pages from C4/Common Crawl |
| **1.2** | ✅ | Defined 30 IPI payloads across 10 attack categories (Documented in `ipi_taxonomy.md`) |
| **1.3** | ✅ | Developed injector script with 14 stealth techniques |
| **1.4** | ✅ | Compiled `sample_real_html_10000_r20.csv` (10,000 samples, 20% malicious) |
| **2.1** | ✅ | DOM parsing (visible vs. hidden text streams) |
| **2.2** | ✅ | Structural feature extraction (incl. internal `<style>` parsing + event handlers) |
| **2.3** | ✅ | Semantic feature extraction (TF-IDF + LSA, serialized as joblib Pipeline) |
| **3.1** | ✅ | URL-group split *before* injection (no cross-split leakage) |
| **3.2** | ✅ | Train RF, XGBoost (baseline & tuned), LR, Hard-OR + Weighted Soft-Vote ensembles |
| **3.3** | ✅ | Evaluate (Acc/Prec/Rec/F1/PR-AUC/ROC-AUC + per-category + recall-by-technique + threshold sweep) |
| **3.4** | ✅ | True LOTO retraining (14×) + OOD adversarial (8 unseen techniques) + natural-benign FPR test |
| **4.1** | ⬜ | Slide deck (English, up to 25 slides) |
| **4.2** | ⬜ | Video recording (Hebrew, 15 min) |
| **4.3** | ⬜ | Kahoot quiz (5 questions) |
| **4.4** | ⬜ | Final upload to Moodle |

---

## 📄 Key References

- **MUZZLE** — Adaptive Agentic Red-Teaming of Web Agents Against IPI Attacks (2026)
- **WebSentinel** — Detecting and Localizing Prompt Injection Attacks for Web Agents (2026)
- **Dual-Modality Multi-Stage Adversarial Safety Training** — Robustifying Multimodal Web Agents (2026)

See `literature_review/IPI_Research_Bibliography.bib` for the full bibliography.

---

## 📋 Submission Requirements

- **Presentation Slot:** Lecture 4, 13:35
- **Deadline:** 3 days before Lecture 4 at 09:00 AM
- **Format:** Pre-recorded video (Group 3 — exempt from live presentation)
- **Deliverables:** PPTX, MP4 Video, Python Source Code, CSV Dataset, Kahoot Link
