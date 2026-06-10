# NotebookLM Chat Prompt: Generate Slide Deck

***Instructions:** Upload all of your project files (the `Slide_Deck_Template.md`, your literature review, and the `taxonomy_graph.md`) into your NotebookLM workspace. Then, copy the text below the line and paste it directly into the NotebookLM Chat box.*

---

**Copy and Paste this into NotebookLM:**

Act as an expert cybersecurity presentation designer. I need you to generate the complete, polished text for a professional academic presentation based exclusively on the documents I have uploaded. 

The project is titled "Shifting Left: Structural Detection of Indirect Prompt Injections in Web Agents". 

Please output the presentation slide-by-slide using the following exact format for every slide:
- **Slide Title:** [Catchy, professional title]
- **Bullet Points:** [3 to 4 concise bullet points. Do not write walls of text. Make them punchy and easy to read on a screen.]
- **Speaker Notes:** [A short paragraph of exactly what I should say out loud to the audience when this slide is showing. Put the complex technical explanations in these notes so the slide itself stays clean.]

**Please ensure the presentation flow follows this structure:**
1. **The Introduction & The Threat:** Explain what autonomous Web Agents are and how attackers use CSS/HTML visual camouflage to plant Indirect Prompt Injections.
2. **The "Readability" Trap (The Gap):** Explain why current defenses fail because they strip HTML to read plain text, destroying the visual evidence of the attack.
3. **The Taxonomy (Direct vs. Indirect, LLM vs ML):** Break down the defensive landscape. Emphasize why our predictive ML approach is faster and safer than using Generative LLMs as classifiers.
4. **Our Solution (WAF3LLM):** Detail the "Shift-Left" architecture operating at the raw HTML ingestion boundary. 
5. **Methodology:** Explain the zero-leakage URL-grouped split and the messy, real-world Common Crawl (C4) dataset.
6. **Results & Tradeoffs:** Highlight the 90.62% True Positive Rate using the optimized WAF3LLM pipeline and its blazing-fast ~20ms inference time.
7. **Advanced Architectures (The Failures):** Briefly explain why GNNs and SMOTE failed to beat the baseline in this highly imbalanced environment.
8. **Limitations & Future Work:** Focus heavily on the Multilingual Jailbreak vulnerability (e.g., Chinese injections) and how simply swapping the English `MiniLM` for a multilingual transformer solves it.
9. **Conclusion:** Summarize why traditional ML dominates heavy LLMs for structural web defense.

Do not hallucinate or make up any data. Only use the metrics, architectures, and methodologies explicitly found in the uploaded sources.
