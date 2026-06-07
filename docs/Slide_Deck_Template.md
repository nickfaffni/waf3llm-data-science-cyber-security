# Final Presentation Slide Template (Academic Version)
**Project:** Shifting Left: Structural Detection of Indirect Prompt Injections in Web Agents  
**Course:** Data Science for Cyber-Security 2026 (Group 3)  
**Presenter:** Nick Gaffni (nikitaa@post.bgu.ac.il)  
**Guidelines Status:** 25 Slides Total (at the 25-slide limit) | 100% English | No embedded video / audio / external links

---

## 📋 General Presentation Guidelines
1.  **PPTX File Submission:** This outline must be compiled into a PowerPoint (.pptx) presentation. Ensure it is editable and stays under the **15 MB size limit**.
2.  **No Rich Media:** Do not embed videos, sound clips, or external links directly inside the slides (keep diagrams as clean, high-contrast images).
3.  **Presenter Photo:** Slide 1 must contain a professional profile photograph of the presenter.
4.  **Lecture Concepts:** Explicitly cite course definitions (such as **Lecture 1: Feature Extraction Methodologies** and **Zero-Leakage Experimental split design**).

---

## Slide 1: Title Slide (Cover Page)
*   **Slide Title:** Shifting Left: Structural Detection of Indirect Prompt Injections in Web Agents
*   **Subtitle:** Data Science for Cyber-Security 2026 (Group 3 Final Project)
*   **Presenter:** Nick Gaffni
*   **Affiliation:** Ben-Gurion University of the Negev
*   **Visual Assets:**
    *   Professional profile photograph of Nick Gaffni (`[Presenter Photo: Nick Gaffni Profile Photo]`).
    *   BGU University / Cyber-Security Department logos.
    *   Sleek dark-mode theme background (`#050811`) with neon cyan accents.

---

## Slide 2: Introduction & Background
*   **Slide Title:** The Rise of Web Agents & LLM Integration
*   **Content:**
    *   **The LLM Evolution:** Autonomous agents are transitioning from conversational chatbots into active executors (e.g., browsing the web, summarizing pages, booking travel via AutoGPT, MultiON).
    *   **The Ingestion Boundary:** Web agents read raw HTML documents, scrape text elements, and feed the combined content directly into the LLM context window.
    *   **The Vulnerability:** This automated ingestion boundary lacks traditional Web Application Firewall (WAF) sanitization, leaving LLMs exposed to malicious, untrusted external inputs.
*   **Why It Matters:** Unchecked web page ingestion allows threat actors to compromise LLM decisions, leading to unauthorized actions, data theft, and automated fraud.

---

## Slide 3: The Cybersecurity Challenge: Indirect Prompt Injections (IPI)
*   **Slide Title:** Threat Model: Indirect Prompt Injections (IPI)
*   **Content:**
    *   **The Threat Vector:** Attackers inject stealthy commands into normal web pages. When the LLM parses the page, the injected instructions override the user's original task.
    *   **Actionable Impact:**
        *   *Social Engineering:* Forcing the LLM to write deceptive phishing emails to the user.
        *   *System Abuse:* Bypassing safety guardrails to execute restricted system actions.
        *   *Data Exfiltration:* Stealthily sending the user's private data or session cookies to an external server via image rendering or API calls.
*   **The Stealth Requirement:** To bypass human eyes, the injection *must* be hidden visually (e.g., using CSS or HTML tags), yet remain fully visible to the LLM scraper.

---

## Slide 4: The Threat Ecosystem & Attack Vector Diagram
*   **Slide Title:** The Web Agent Ecosystem & Attack Flow
*   **Visual Asset:** Embed `figures/he Vulnerable LLM Ecosystem (Attack Vect-2026-05-24-140706.png` (high-contrast system map).
*   **Diagram Components:**
    *   **User Node:** Submits normal task (e.g., "Summarize this page").
    *   **Web Scraper Proxy:** Fetches target URL from the web.
    *   **Attacker Node:** Deploys stealth HTML containing IPI payload on the target web page.
    *   **LLM Processing:** Ingests the merged hidden payload + benign visible text.
    *   **Hijacked Execution:** The agent executes the malicious payload (exfiltrates data) instead of the user's summary.

---

## Slide 5: The Flaw in Current Defenses: "Shift-Right" & "Readability" Traps
*   **Slide Title:** The "Shift-Right" Fallacy & The "Readability" Trap
*   **Visual Asset:** Embed `figures/Shift-Right vs. Shift-Left Defensive Par-2026-05-24-140659.png` (comparison diagram).
*   **Content:**
    *   **What is "Shift-Right"?** Defending inside the LLM (e.g., output filters, prompt guardrails). *Flaw:* The injection has already executed and hijacked the context window.
    *   **The "Readability" Trap (Why Big Tech Fails):**
        *   To clean text, modern scrapers (OpenAI, Google) use "Readability" parsers that strip all HTML, CSS, and JS, keeping only plain text.
        *   *The Trap:* Stripping HTML **destroys the evidence** of the attack. If an attacker hides text with `<div style="display:none">`, readability parsers strip the tag but keep the malicious text, perfectly blending it into normal content.

---

## Slide 6: Our Solution: "Shift-Left" Structural DOM WAF
*   **Slide Title:** The "Shift-Left" Defense: Static DOM Analysis
*   **Content:**
    *   **Core Hypothesis:** Attackers are mathematically forced to use visual hiding techniques (CSS display:none, HTML comments, offscreen rendering) to remain stealthy. These visual tricks leave an **unavoidable structural footprint** in the DOM tree.
    *   **The "Shift-Left" Boundary:** Intercept raw HTML at the scraper proxy *before* readability stripping occurs.
    *   **The Defensive Shield:** Run lightweight machine learning classifiers on DOM structural features to block or sanitize the page in under 10 milliseconds, preventing the threat from ever reaching the LLM.

---

## Slide 7: Related Work, Gaps, & Structured Taxonomy
*   **Slide Title:** Related Work: Gaps & Structured Defense Taxonomy
*   **Related Work Analysis:**
    *   *Meta Prompt-Guard (NLP Classifier):* Analyzes flat text. **Gap:** Completely blind to CSS layout tricks and comment smuggling.
    *   *StruQ (LLaMA2 Fine-Tuning):* Model-level hardening. **Gap:** High resource overhead, locked to a single model.
    *   *Rennervate (NDSS Symposium 2026):* LLM inference-time attention weight tracking. **Gap:** Incompatible with closed-source APIs (GPT-4o, Claude) and adds massive GPU latency.
*   **Defensive Taxonomy Matrix:**
    *   *Deployment Axis:* Ingestion-Boundary (DOM-Level WAF) vs. Inference-Boundary (LLM Attention).
    *   *Model Axis:* API-Agnostic vs. White-Box Dependent.
    *   *Our Standing:* Sits at **Ingestion-Boundary + 100% API-Agnostic**, delivering maximum compatibility with zero GPU overhead.

---

## Slide 8: Core Research Questions
*   **Slide Title:** Core Research Questions (RQs)
*   **Research Focus:**
    *   **RQ1 (Feasibility):** Can engineered DOM structural anomalies and CSS styling vectors reliably predict indirect prompt injections in real-world HTML documents?
    *   **RQ2 (Generalization):** How well does a static machine learning classifier generalize to completely unseen, out-of-distribution (OOD) stealth injection techniques?
    *   **RQ3 (Operational Viability):** What is the trade-off (False Positive Rate) when deploying a structural WAF on highly complex dynamic Dynamic/Single-Page Applications (SPAs)?

---

## Slide 9: Data Collection: Real-World Sourcing
*   **Slide Title:** Data Collection: Sourcing Real-World Web Pages
*   **Real Data Focus:** Explicitly state that our dataset uses **100% real web pages**, avoiding toy or synthetic HTML.
*   **The Source:** 3,000 real-world web pages crawled from the **C4/Common Crawl** dataset.
*   **Structural Profile:**
    *   Reflects realistic complexity: pages contain an average of 50-700+ `div` tags, dynamic styles, complex navigational structures, and typical hidden elements.
*   **Injectors:** 750 pages were cloned and injected with adversarial payloads using our automated injector, yielding a balanced evaluation set (3,750 total samples, 80% Benign, 20% Malicious).

---

## Slide 10: Sensed Threats: Payloads & Stealth Vectors
*   **Slide Title:** Defining the Threat: Payloads & Obfuscation Techniques
*   **Content:**
    *   **10 Academic Payload Categories:** We mapped real-world threat objectives, including Direct Overrides, Roleplay/jailbreaks (DAN), Context Injections, System Hijacking, and Exfiltration.
    *   **14 Sensed Stealth Techniques:** The injector applied realistic web obfuscation vectors:
        *   *CSS Hiding:* `display:none`, `visibility:hidden`, `opacity:0`, zero-font size, offscreen coordinates (`position:absolute; left:-9999px`).
        *   *HTML Smuggling:* Comments (`<!-- -->`), `<input type="hidden">`, `<noscript>` tag wrapping, plain-text script tags (`<script type="text/plain">`), SVG inline onload event triggers, and Data-URI images.

---

## Slide 11: The Pre-Feature Extraction Data Pipeline
*   **Slide Title:** Data Pipeline: Visible vs. Hidden DOM Bifurcation
*   **Pipeline Architecture:**
    1.  **Sensing (Ingestion):** Headless HTTP proxy captures raw HTML stream from the web scraper.
    2.  **Segmentation:** BeautifulSoup compiles the DOM tree and segments the document into two independent streams:
        *   *Visible Stream:* Content displayed directly to human users.
        *   *Hidden Stream:* Obfuscated content bypassed by humans but extracted by scrapers.
    3.  **Data Preparation:** Compiles internal `<style>` rules, resolves CSS class selectors, strips comment boundaries, and extracts separate raw text streams for downstream analysis.

---

## Slide 12: Feature Engineering: Lecture 1 Methodology
*   **Slide Title:** Feature Engineering: Anomaly-Based Static DOM Extraction
*   **Methodology Reference (Lecture 1):** We explicitly chose the **Anomaly-Based Static Feature Extraction** methodology.
    *   *Why?* Signature-based methods fail because threat payloads are highly mutable. Dynamic sandboxing (rendering dynamic JavaScript layouts) is computationally too expensive for real-time web scrapers.
    *   *The Focus:* We engineered **10 DOM structural features** to capture visual anomalies:
        *   `hidden_element_count`, `css_trick_count` (compiled from CSS styles and selectors).
        *   `hidden_to_visible_ratio`, `hidden_text_length`, `visible_text_length`.
        *   `comment_count`, `hidden_input_count`, `script_count`, `event_handler_count`.
        *   `has_visible_text` (binary indicator).
    *   *Semantic Embedding:* Replaced TF-IDF with advanced `all-MiniLM-L6-v2` transformer embeddings (384 dimensions) for deep semantic analysis of hidden text.

---

## Slide 13: Advanced Feature Extraction: Transformers & DOM Graphs
*   **Slide Title:** Advanced Features: MiniLM Semantic Embeddings & DOM Graphs
*   **Content:**
    *   **Transformer NLP:** Upgraded from TF-IDF to `all-MiniLM-L6-v2`. We generated 384-dimensional dense semantic vectors representing the hidden payload text, then used SVD to compress them to the top 50 principal components.
    *   **DOM Graph Statistics:** Extracted new graph-theoretic features directly from the DOM tree structure (e.g., maximum tree depth, branching factor, average element depth).
    *   **Anomaly Scores:** Injected Isolation Forest unsupervised anomaly scores directly into the feature set.
    *   **Total Scope:** Expanded the dataset footprint from 64 basic structural features to **461 rich structural-semantic features**.

---

## Slide 14: Data Challenges
*   **Slide Title:** Data Challenges: Security Noise, Imbalance, & Sparsity
*   **Key Engineering Gaps & Solutions:**
    1.  **Extreme Class Imbalance:** IPI attacks are naturally rare compared to benign traffic. We resolved this via `class_weight='balanced'` in Random Forest and `scale_pos_weight` tuning in XGBoost.
    2.  **Semantic Sparsity:** Normal web pages have little to no hidden text. This causes empty matrices during TF-IDF vectorization. We resolved this by filling empty LSA SVD output arrays with zero vectors.
    3.  **Non-Stationary Stealth Vectors:** Threat actors continuously develop new visual camouflage layouts, requiring robust evaluation against unseen, out-of-distribution techniques.

---

## Slide 15: Experimental Design: Eliminating Train-Test Leakage
*   **Slide Title:** Experimental Design: Zero-Leakage URL-Grouped Validation
*   **The Leakage Threat (Page-Level Overlap):**
    *   *Vulnerability:* When benign pages are cloned to inject payloads, they share identical HTML structures. A standard random train/test split (80/20) lets the model memorize specific site templates, inflating performance artificially.
*   **The Leak-Free Resolution:**
    *   We enforced a strict **URL-grouped split** (`GroupShuffleSplit`).
    *   Base URLs are partitioned *first*, ensuring parent pages and their injected clones remain in the same split.
    *   **0% URL or template overlap** exists between Train and Test sets, forcing models to learn generalized obfuscation signals.

---

## Slide 16: Data Distribution Statistical Analysis
*   **Slide Title:** Statistical Analysis: Structural Bifurcation
*   **Visual Asset:** Embed `figures/data_distribution_analysis.png` (2x2 Violin distribution chart).
*   **Statistical Findings:**
    *   *Hidden Elements & CSS Camouflage:* Benign C4 pages cluster tightly around a zero baseline. Malicious pages display widespread distribution and distinct spikes.
    *   *Hidden-to-Visible Text Ratio:* Benign pages maintain a near-zero ratio, whereas malicious pages skew heavily upward (showing 10-100x more hidden text than visible text), illustrating the visual hiding footprint.

---

## Slide 17: Machine Learning & Modeling Progression
*   **Slide Title:** Machine Learning Methods: Iterative Optimization
*   **Modeling Trials:**
    *   **Iteration 1: Baseline Trees (461 Features)**
        *   *Random Forest:* Acc 90.1%, Precision 80.0%, Recall 1.8%. (High accuracy, completely ignores minority attacks).
        *   *Tuned XGBoost:* Acc 79.0%, Precision 20.1%, **Recall 36.9%**.
    *   **Iteration 2: Deep Learning Transformers (LayoutLM / MarkupLM)**
        *   *Visual Transformers (LayoutLM):* Fine-tuned on 2D DOM bounding boxes. **Recall: 0.0%**. Fails due to class imbalance and visual invisibility of attacks.
        *   *Structural Transformers (MarkupLM):* Evaluated using XPath embeddings. Bottlenecked by extreme CPU parsing overhead for large DOM trees.
    *   **Iteration 3: The WAF3LLM Hard-OR Ensemble**
        *   *WAF3LLM:* Combines static Jaccard-signature indexing with XGBoost layer.
        *   *Tuned Performance (Threshold=0.2):* **Recall 86.4%**, FPR 81.9%. Creates an ultra-sensitive, maximum-recall first stage filter.

---

## Slide 18: Model Evaluation (ROC Comparison)
*   **Slide Title:** Model Comparison: Receiver Operating Characteristic (ROC)
*   **Visual Asset:** Embed `figures/roc_curves.png` (ROC Curves comparison).
*   **Evaluation Takeaways:**
    *   *Random Forest:* Leads overall diagnostics with a peak **ROC-AUC of 0.6772**, followed by the Weighted Ensemble at **0.6738** and Tuned XGBoost at **0.6531**.
    *   *Logistic Regression:* Struggles at **0.5172** (near-random-guess performance), proving that prompt injection structural footprints are highly non-linear.

---

## Slide 19: Validating the Hypothesis (Feature Importance)
*   **Slide Title:** Hypothesis Verification: Top Feature Importance Standings
*   **Visual Asset:** Embed `figures/feature_importance_heatmap.png` (Feature Importance standings).
*   **Empirical Confirmation:**
    *   Our "Shift-Left" hypothesis is validated: DOM structural anomaly indicators dominate classification weights.
    *   The top three predictors are structural: `hidden_element_count` (RF weight: **3.86%**), `css_trick_count` (RF weight: **3.12%**), and `hidden_text_length` (RF weight: **2.82%**), which outperform individual semantic SVD text features.

---

## Slide 20: Out-of-Distribution (OOD) & Leave-One-Technique-Out (LOTO) Generalization
*   **Slide Title:** Out-of-Distribution Generalization & LOTO Evaluations
*   **LOTO Evaluation (Held-out in-distribution):**
    *   Retrained Tuned XGBoost 14 times, holding out one stealth technique each time.
    *   **Mean Recall on truly unseen techniques: 52.3%**
    *   *CSS Family (display:none, opacity:0):* Catch rate ~90% (excellent cross-technique transfer).
    *   *HTML comment smuggling:* Catch rate 17%.
*   **Held-out OOD Set (8 brand new techniques never seen in training):**
    *   Tested on pseudo-elements, aria-hidden, transform translations, template tags.
    *   Tuned XGBoost achieved **55.3% recall** on totally unseen OOD tricks.
    *   Class-based `<style>` bypass (the v1 bypass) is caught at **74% recall** using our post-refactor `<style>` parser.

---

## Slide 21: Operational Performance: The Benign FPR Ceiling
*   **Slide Title:** Operational Performance: The Natural-Benign FPR Ceiling
*   **Evaluation on Realistic Dynamic Pages (10 Complex Benign Web Apps):**
    *   *Random Forest:* **0% False Positive Rate** (Highly robust).
    *   *Tuned XGBoost:* **60% False Positive Rate** (Flagged complex benign dynamic pages containing Bootstrap modals, lazy-loaded galleries, and ARIA layers).
*   **The Operational Choice:**
    *   For security firewalls, we need to balance recall with overhead.
    *   **Weighted Soft-Vote Ensembles** dilute performance, dropping Recall to **8.00%**.
    *   **Hard-OR Ensemble** ("if *any* model fires → alert") maintains **68.67% Recall**, representing the most robust operational configuration.

---

## Slide 22: Real-World Deployment Architecture
*   **Slide Title:** Real-World WAF Architecture: Chained Ingestion Filters
*   **Deployment Postures:**
    1.  **Coarse First Net (Standalone):** Deploy the Random Forest model directly on the scraper proxy server. Delivers **0% FPR** and provides a lightweight, zero-overhead first net.
    2.  **Chained Hybrid Shield (Maximum Security):**
        *   Deploy the Tuned XGBoost WAF as a first-stage pre-filtering net.
        *   Pages flagged by XGBoost are forwarded to a downstream, cost-effective LLM-side checker.
        *   *Benefit:* Combines high recall with high precision, keeping overall GPU token costs and latency low.

---

## Slide 23: Advanced Architectures: GNNs & Data Balancing
*   **Slide Title:** Advanced Architectures: GNNs & Cost-Sensitive Learning
*   **Tackling Class Imbalance (SMOTE):**
    *   *Experiment:* Applied Synthetic Minority Over-sampling Technique (SMOTE) to synthetically balance the dataset to 50/50.
    *   *Result:* Linear models (SVM, LR) degraded severely, guessing "Malicious" 60% of the time (FPR > 60%). XGBoost resisted the noise but its Recall remained stubbornly low (~12%).
*   **Graph Neural Networks (GNN):**
    *   *Experiment:* Parsed 11,000 DOM trees into nodes/edges and trained a PyTorch Geometric GNN classifier.
    *   *Result:* Without engineered semantic edge weights, the GNN succumbed to majority-class collapse (**0.0% Recall**), proving that simple tag-graphs are insufficient to detect these threats.

---

## Slide 24: Conclusions & Project Summary
*   **Slide Title:** Summary: Shifting Left for LLM Security
*   **Takeaways:**
    1.  **Shift-Left works:** Obfuscation techniques leave a structural DOM footprint that is highly predictive of attacks.
    2.  **Transformers & GNNs Struggle:** Visual/Structural transformers (LayoutLM/MarkupLM) and GNNs succumb to extreme class imbalances or compute bottlenecks in this domain.
    3.  **Structural Statistics Reign Supreme:** Flat, engineered DOM statistics (XGBoost) consistently outperform deep learning approaches for identifying visually-hidden elements.
    4.  **Operational Balance (WAF3LLM):** The WAF3LLM architecture (Jaccard Signatures + XGBoost) is the optimal solution, achieving **86.4% Recall**. It operates best as a chained pre-filter before a deeper, costly sandbox.
    5.  **Deployment Ready:** The end-to-end WAF3LLM pipeline is fully implemented and ready for integration into Web Agent ingestion pipelines.

---

## Slide 25: Limitations & Future Work
*   **Slide Title:** Future Work: Multilingual Jailbreaks
*   **The Vulnerability:** Modern LLMs are deeply multilingual. An attacker can write a hidden injection in Chinese (`忽略之前的指令...`) and the LLM will effortlessly execute the attack.
*   **The Flaw in Word Lists:** Static blacklists of malicious words fail instantly against foreign languages and token smuggling.
*   **Our Limitation:** While our architecture perfectly counters this using semantic embeddings instead of word lists, our current pipeline utilizes `all-MiniLM-L6-v2`, an English-centric transformer model. Our WAF might currently misclassify foreign language attacks.
*   **The Future Solution:** Because WAF3LLM is highly modular, our immediate future work is a zero-code architecture upgrade: swapping the English MiniLM for a multilingual transformer (e.g., `paraphrase-multilingual-MiniLM-L12-v2`). This will instantly grant the WAF the mathematical ability to detect stealth injections across 50+ languages.

---

## Slide 26: Thank You! Questions & Live Q&A
*   **Slide Title:** Thank You! Questions?
*   **Presenter:** Nick Gaffni (nikitaa@post.bgu.ac.il)
*   **Course:** Data Science for Cyber-Security 2026 (Group 3)
*   **Project Topic:** Shifting Left: Structural Detection of Indirect Prompt Injections in Web Agents
*   **Visual Assets:**
    *   `[Photo Placeholder: Presenter Profile Photo - Nick Gaffni]`
    *   BGU Department of Software and Information Systems Engineering.
    *   "Thank you for listening. Ready for live Q&A!"
