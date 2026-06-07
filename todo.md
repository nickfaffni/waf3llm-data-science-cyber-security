# DS4CS — Research Roadmap & TODO List

This document tracks our research exploration, new model integrations, and performance improvement goals for the **Indirect Prompt Injection (IPI)** detection system.

---

## 🎯 Target Performance Benchmarks
- **Lecturer Reference Benchmark (from other topics):**
  - **TPR (Recall):** $\ge 0.90$
  - **FPR:** $\le 0.098$
  - **AUC:** $\ge 0.939$

---

## 📋 Tasks & Feature Explorations

### 1. Expanded Model Suite & Comparison
Compare our baseline tree models against a wider range of classifiers to find the best decision boundary:
- [x] **Random Forest** (Implemented & scaled)
- [x] **Logistic Regression** (Implemented & scaled)
- [x] **Naive Bayes Classifier** (e.g., Multinomial / Gaussian NB) (Implemented)
- [x] **Support Vector Machines (SVM)** (with linear / RBF kernels) (Implemented)
- [x] **K-Nearest Neighbors (KNN)** (Implemented)
- [x] **Artificial Neural Networks (ANN)** (shallow multi-layer perceptron) (Implemented)
- [x] **Deep Neural Networks (DNN)** (multi-layer dense network with dropout and batch norm) (Implemented)
- [x] **Convolutional Neural Networks (CNN)** (for 2D image / grid representations) (Implemented)

### 2. Feature Representations & Image Mapping
Explore alternative ways of representing HTML content:
- [x] **Text-to-Image Representation Learning**: Convert HTML layouts or raw code structures into 2D grid/image representations (e.g., pixel matrices based on tag frequency or DOM depth) and train a CNN to detect spatial layouts of obfuscation. (Implemented)

### 3. Unsupervised Learning & Similarity Metrics
Implement unsupervised methods to identify anomalies in injection structures:
- [x] **Jaccard Distance Similarity Classifier**: Build a knowledge-based dataset reference and classify pages based on their structural similarity (Jaccard distance of DOM structure/tokens) to known benign templates. (Implemented)
- [x] **Unsupervised Clustering & Visualization**: Use K-Means or DBSCAN on LSA features to map the layout space and visualize clusters (e.g., using t-SNE or UMAP). (Implemented)

### 4. Advanced Architectures & Explainability
- [x] **WAF3LLM ("WAF 3 Layer LM") Defense Architecture**: Design a multi-layered security system consisting of:
  1. *Layer 1*: Lightweight similarity/signature check (Jaccard similarity).
  2. *Layer 2*: Structural & semantic classification (Tuned XGBoost / RF).
  3. *Layer 3*: Downstream token-attention or LLM-side sanitization. (Implemented as WAF3LLM)
- [x] **Explainable AI (XAI)**: Integrate SHAP (SHapley Additive exPlanations) or LIME to explain model predictions, identifying exactly which DOM anomalies (e.g., specific CSS properties or hidden ratios) trigger flags. (Implemented)
- [x] **FPR/TPR Improvement**: Focus on tuning prediction thresholds and feature weights to hit the target benchmark ($\text{TPR} \ge 0.9$, $\text{FPR} \le 0.098$). (Tuned optimal operating point evaluated)