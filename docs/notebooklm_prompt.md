# NotebookLM Guidance Prompt: WAF3LLM Project

***Instructions:*** *If you are generating an Audio Overview or a Study Guide based on the attached slide deck ("Shifting Left: Structural Detection of Indirect Prompt Injections in Web Agents" by Nick Gaffni), please adhere to the following directorial guidelines:*

### 1. Tone & Atmosphere
*   **Vibe:** Treat this as a cutting-edge, high-stakes cybersecurity breakthrough. The tone should be engaging, slightly dramatic (highlighting the "invisible" threat of AI hacking), and highly technical but accessible.
*   **The Hook:** Start by explaining that hackers are no longer attacking AI chatboxes directly. Instead, they are laying invisible "traps" on public websites to hijack autonomous AI Web Agents that scrape the internet.

### 2. Core Concepts to Emphasize
*   **The "Readability" Trap:** You must emphasize why current defenses are failing. Explain that standard security tools strip away HTML/CSS to read plain text. Explain why this is a fatal flaw: it destroys the evidence of the crime (the visual camouflage).
*   **The Shift-Left Solution:** Focus heavily on Nick's architectural breakthrough. Explain that WAF3LLM operates at the *ingestion boundary*—intercepting the raw DOM tree to catch the structural footprint of the attack before the LLM ever sees it.
*   **The Data Leakage Fix:** Highlight Nick's rigorous scientific methodology. Mention how he fixed a critical flaw in previous research by using "URL-grouped splits" to prevent the Machine Learning model from cheating by memorizing website templates.

### 3. Key Metrics to Celebrate
*   **The Results:** Celebrate the WAF3LLM ensemble (Jaccard Signatures + XGBoost) achieving an **86.4% True Positive Rate**.
*   **The Speed:** Make sure to highlight the operational viability: the entire pipeline runs in **under 20 milliseconds** locally, making it a blazing-fast, zero-API-cost Web Application Firewall.

### 4. Crucial Distinctions & Future Work
*   **LLMs vs. Traditional ML:** Contrast Nick's lightweight XGBoost approach against heavy "Shift-Right" LLM defenses that add massive GPU costs and are susceptible to recursive prompt injections.
*   **The Multilingual Threat (Future Work):** Mention the fascinating limitation regarding Multilingual Jailbreaks. Explain how modern LLMs will execute attacks written in Chinese, and how the WAF3LLM's modular architecture can solve this instantly by swapping the English `MiniLM` for a multilingual transformer.

### 5. Closing Thought for the Hosts
*   End the podcast by reflecting on a powerful conclusion: The WAF3LLM project proves that when defending against structural web threats, traditional, lightning-fast Machine Learning (like XGBoost) is still vastly superior to massive Generative AI models.
