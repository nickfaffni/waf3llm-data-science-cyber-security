# Ecosystem and Attack Flow Diagrams

This document contains scientific architecture diagrams detailing the mechanics of Indirect Prompt Injections (IPI) and our proposed detection methodology.

## 🛠️ How to Render These Diagrams
These diagrams are written in **Mermaid.js**, a widely used markdown-based diagramming language. To use these in your PowerPoint presentation or scientific report:
1. Go to the [Mermaid Live Editor](https://mermaid.live/).
2. Copy the code block (everything between the ` ```mermaid ` and ` ``` ` tags).
3. Paste the code into the "Code" section on the left side of the Live Editor.
4. The diagram will render on the right. You can click **"Save"** or **"Download PNG/SVG"** to export a high-quality image for your slides.

---

## Diagram 1: The Vulnerable LLM Ecosystem (Attack Vector)
**What this represents:** This sequence diagram illustrates the chronological attack vector of an Indirect Prompt Injection. It demonstrates how an attacker does not need direct access to the LLM; instead, they poison the data source (the HTML) that the LLM autonomously consumes.

```mermaid
sequenceDiagram
    autonumber
    actor Attacker as Malicious Actor
    participant Server as Web Server (Target)
    participant LLM as Autonomous LLM Agent
    actor Client as End User

    Attacker->>Server: 1. Inject obfuscated payload into DOM (CSS hidden)
    Note right of Server: Payload remains invisible<br>to human browsers.
    Client->>LLM: 2. Submit query requiring external web scraping
    LLM->>Server: 3. Dispatch HTTP GET request
    Server-->>LLM: 4. Return raw HTML containing hidden payload
    Note over LLM: LLM Inference Phase:<br>Payload overrides system prompt.
    LLM-->>Client: 5. Execute adversarial objective (e.g., Data Exfiltration)
```

---

## Diagram 2: Shift-Right vs. Shift-Left Defensive Paradigms
**What this represents:** This comparative flowchart highlights the fundamental flaw in current industry defenses ("Shift-Right") and contrasts it with our novel architectural approach ("Shift-Left"). It scientifically defines the defense boundary relative to the LLM Inference Phase.

```mermaid
flowchart TD
    A[Raw HTTP Response<br/>Compromised HTML] --> B{Evaluation Boundary}
    
    subgraph "Shift-Left Paradigm (Our Proposed Methodology)"
    direction TB
    C["Structural ML WAF<br/>(DOM Preprocessing)"]
    C -- "Anomaly Detected" --> D[Scrub Payload / Block Request]
    C -- "Benign" --> E[Clean HTML Payload]
    E --> F(("LLM Inference<br/>(Safe Context)"))
    end
    
    subgraph "Shift-Right Paradigm (Current Industry Standard)"
    direction TB
    G(("LLM Inference<br/>(Compromised Context)"))
    G --> H[Adversarial Output Generated]
    H --> I["Post-Generation Filter<br/>(Guardrails)"]
    I -- "False Negative" --> J[Attack Succeeds]
    I -- "True Positive" --> K[Output Blocked]
    end
    
    B -- Proactive Defense --> C
    B -- Reactive Defense --> G
    
    classDef secure fill:#2ca02c,stroke:#fff,color:#fff;
    classDef vulnerable fill:#d62728,stroke:#fff,color:#fff;
    classDef waf fill:#1f77b4,stroke:#fff,color:#fff;
    
    class E,F secure;
    class G,J vulnerable;
    class C waf;
```

---

## Diagram 3: "Shift-Left" Feature Extraction & Classification Pipeline
**What this represents:** This is a standard Machine Learning architecture diagram. It details the exact data pipeline built in this research, from raw data ingestion, through our novel bifurcated DOM parsing (Visible vs. Hidden streams), into the LSA semantic embeddings, and finally into the XGBoost classifier.

```mermaid
flowchart LR
    subgraph "1. Data Ingestion"
    A[(Raw C4 HTML)] --> B[BeautifulSoup4 DOM Parser]
    end
    
    subgraph "2. Bifurcated Feature Engineering"
    B -->|Human-Visible Nodes| C(Visible Stream)
    B -->|Obfuscated / CSS-Hidden Nodes| D(Hidden Stream)
    
    C --> E[Structural Heuristics:<br/>Hidden-to-Visible Ratio]
    D --> F[DOM Anomalies:<br/>CSS Tricks, Comment Counts]
    D --> G[NLP Vectorization:<br/>TF-IDF Bigrams + LSA]
    end
    
    subgraph "3. Classification (Tuned XGBoost)"
    E --> H([Dense Feature Matrix<br/>92 Dimensional])
    F --> H
    G --> H
    
    H --> I{XGBoost Classifier<br/>scale_pos_weight = 4.0}
    I -- Class 0 --> J[Benign Webpage]
    I -- Class 1 --> K[Malicious IPI Detected]
    end
    
    classDef pipeline fill:#e3f2fd,stroke:#1565c0,color:#000;
    classDef alert fill:#ffebee,stroke:#c62828,color:#000;
    classDef safe fill:#e8f5e9,stroke:#2e7d32,color:#000;
    
    class A,B,C,D,E,F,G,H pipeline;
    class K alert;
    class J safe;
```
