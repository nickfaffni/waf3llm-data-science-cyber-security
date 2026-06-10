# DS4CS Presentation: Kahoot Quiz
**Goal:** 5 engaging questions to test the class on the core concepts of our presentation.

---

### Question 1
**Question:** What does "IPI" stand for in the context of our project?
- 🟥 A) Internal Process Integration
- 🟦 B) Indirect Prompt Injection *(Correct)*
- 🟨 C) Intelligent Parsing Interface
- 🟩 D) Inline Payload Insertion

**Presenter Note:** IPI happens when an LLM reads external data (like a webpage) that contains hidden malicious instructions, tricking the LLM into executing them.

---

### Question 2
**Question:** Why do current LLM security defenses often fail against IPI attacks?
- 🟥 A) The LLMs are too slow to read the payloads.
- 🟦 B) The defenses don't check output.
- 🟨 C) They are "Shift-Right" (they check the prompt *after* the LLM has already read the malicious data). *(Correct)*
- 🟩 D) Hackers use quantum encryption.

**Presenter Note:** Current defenses usually rely on asking an LLM to check if a prompt is safe. By the time the LLM is checking the hidden webpage data, the attack has already breached the system. Our solution "Shifts-Left" by checking the HTML *before* it reaches the LLM.

---

### Question 3
**Question:** How did our project hide malicious payloads inside the benign HTML dataset?
- 🟥 A) By encrypting the text in a password-protected zip file.
- 🟦 B) By using CSS tricks (like `display:none`) and HTML comments to make the text invisible to humans. *(Correct)*
- 🟨 C) By deleting the HTML and replacing it with Python code.
- 🟩 D) By putting the text at the very bottom of the page.

**Presenter Note:** We used 14 different stealth techniques, including CSS obfuscation and hidden inputs, to ensure a human wouldn't see the attack, but a web-scraping LLM would.

---

### Question 4
**Question:** When our Machine Learning models analyzed the web pages, what was the #1 most important feature they used to detect an attack?
- 🟥 A) The length of the visible text.
- 🟦 B) The ratio of Hidden Text to Visible Text.
- 🟨 C) The semantic intent and instruction density (imperative command verbs) of the hidden text. *(Correct)*
- 🟩 D) The URL of the website.

**Presenter Note:** This was our biggest finding! While structural DOM features (like hidden element counts) provide a general signal, adding domain-expert features that measure the semantic intent (like the ratio of command/imperative verbs in the hidden text) boosted our tuned XGBoost recall from 54% to 78.8%.

---

### Question 5
**Question:** Because our dataset had 4 times as many benign pages as malicious ones (80/20), what problem did our initial XGBoost model have?
- 🟥 A) It had High Accuracy, but terrible Recall (it missed most of the attacks). *(Correct)*
- 🟦 B) It kept crashing our computer.
- 🟨 C) It thought every single page was malicious.
- 🟩 D) It refused to analyze the benign pages.

**Presenter Note:** The model was "lazy" and guessed benign most of the time to get an 80% accuracy score. We had to fix this by using `scale_pos_weight` to force the model to penalize itself harder if it missed an attack!
