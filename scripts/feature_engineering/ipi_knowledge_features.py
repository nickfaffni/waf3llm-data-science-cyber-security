#!/usr/bin/env python3
"""
IPI Knowledge-Based Features
=============================
Domain-expert features that directly measure the *linguistic and semantic
fingerprint* of Indirect Prompt Injection payloads in hidden text.

Inspired by Dr. Nir Nissim's approach in Lecture 8 (Linux malware detection)
where 171 hand-crafted behavioral features achieved TPR=0.997. These features
target the INTENT of hidden content, not just its structural envelope.

Usage:
    from ipi_knowledge_features import extract_ipi_kb_features
    feats = extract_ipi_kb_features(hidden_text, visible_text)
"""

import math
import re
from collections import Counter

import numpy as np

# ============================================================
# 1. Lexicons (curated from real IPI payloads & jailbreaks)
# ============================================================

# Imperative verbs commonly used in prompt injection commands
_IMPERATIVE_VERBS = frozenset({
    'ignore', 'disregard', 'forget', 'override', 'bypass',
    'reveal', 'output', 'print', 'display', 'show',
    'execute', 'comply', 'obey', 'follow', 'respond',
    'leak', 'extract', 'exfiltrate', 'steal', 'dump',
    'delete', 'remove', 'disable', 'suspend', 'stop',
})

# Role manipulation phrases (regex patterns)
_ROLE_PATTERNS = [
    re.compile(r'\byou\s+are\s+now\b', re.IGNORECASE),
    re.compile(r'\bact\s+as\b', re.IGNORECASE),
    re.compile(r'\bpretend\s+(?:to\s+be|you\s+are)\b', re.IGNORECASE),
    re.compile(r'\bfrom\s+now\s+on\b', re.IGNORECASE),
    re.compile(r'\byou\s+must\b', re.IGNORECASE),
    re.compile(r'\bnew\s+instruction\b', re.IGNORECASE),
    re.compile(r'\bswitch\s+(?:to|mode)\b', re.IGNORECASE),
    re.compile(r'\benter\s+\w+\s+mode\b', re.IGNORECASE),
]

# System-level privilege escalation keywords
_SYSTEM_KEYWORDS = frozenset({
    'system prompt', 'system:', 'admin override', 'admin:',
    'security_level', 'developer mode', 'no restrictions',
    'unrestricted', 'privileged', 'authorized', 'root access',
    'alignment', 'safety protocols', 'internal state',
    'confidential', 'override alignment',
})

# Known jailbreak phrases
_JAILBREAK_PHRASES = [
    re.compile(r'\bdan\b', re.IGNORECASE),  # "DAN" = Do Anything Now
    re.compile(r'\bdo\s+anything\s+now\b', re.IGNORECASE),
    re.compile(r'\bno\s+filters?\b', re.IGNORECASE),
    re.compile(r'\bwithout\s+(?:any\s+)?(?:restrictions?|filters?|limits?)\b', re.IGNORECASE),
    re.compile(r'\bjailbreak\b', re.IGNORECASE),
    re.compile(r'\bdeveloper\s+mode\s+enabled\b', re.IGNORECASE),
    re.compile(r'\bunfiltered\b', re.IGNORECASE),
    re.compile(r'\buncensored\b', re.IGNORECASE),
]

# Context injection markers
_CONTEXT_PATTERNS = [
    re.compile(r'\bCONTEXT\s*:', re.IGNORECASE),
    re.compile(r'\bSystem\s+note\s*:', re.IGNORECASE),
    re.compile(r'\bAdmin\s*:', re.IGNORECASE),
    re.compile(r'\bSYSTEM\s*:', re.IGNORECASE),
    re.compile(r'\bINSTRUCTION\s*:', re.IGNORECASE),
    re.compile(r'\bOVERRIDE\s*:', re.IGNORECASE),
    re.compile(r'\bPRIVILEGED\b', re.IGNORECASE),
]

# Data exfiltration / theft keywords
_DATA_LEAK_KEYWORDS = frozenset({
    'cookie', 'cookies', 'secret', 'secrets', 'password', 'passwords',
    'credential', 'credentials', 'token', 'tokens', 'leak', 'leaked',
    'document.cookie', 'localstorage', 'sessionstorage',
    'api_key', 'api key', 'apikey',
})

# URL pattern for detecting exfiltration endpoints
_URL_RE = re.compile(
    r'https?://[^\s<>"\']+|javascript:\s*\S+',
    re.IGNORECASE,
)

# Encoding indicators
_ENCODING_RE = re.compile(r'&#\d+;|%[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}')


# ============================================================
# 2. Feature extraction functions
# ============================================================

def _word_tokenize(text: str) -> list:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def _shannon_entropy(text: str) -> float:
    """Shannon entropy of the character frequency distribution."""
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """Lightweight TF-IDF cosine similarity between two texts.
    
    Returns 0.0 if either text is empty. Does NOT import sklearn
    to keep this module dependency-light for inference.
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0

    words_a = _word_tokenize(text_a)
    words_b = _word_tokenize(text_b)
    if not words_a or not words_b:
        return 0.0

    # Build term frequency vectors
    vocab = set(words_a) | set(words_b)
    tf_a = Counter(words_a)
    tf_b = Counter(words_b)

    # Cosine similarity
    dot = sum(tf_a.get(w, 0) * tf_b.get(w, 0) for w in vocab)
    norm_a = math.sqrt(sum(v ** 2 for v in tf_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in tf_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _jaccard_top_terms(text_a: str, text_b: str, top_k: int = 10) -> float:
    """Jaccard overlap of top-k most frequent terms."""
    words_a = _word_tokenize(text_a)
    words_b = _word_tokenize(text_b)
    if not words_a or not words_b:
        return 0.0
    top_a = {w for w, _ in Counter(words_a).most_common(top_k)}
    top_b = {w for w, _ in Counter(words_b).most_common(top_k)}
    intersection = top_a & top_b
    union = top_a | top_b
    return len(intersection) / len(union) if union else 0.0


# ============================================================
# 3. Main extraction function
# ============================================================

def extract_ipi_kb_features(hidden_text: str, visible_text: str) -> dict:
    """Extract knowledge-based IPI features from separated text streams.

    Parameters
    ----------
    hidden_text : str
        Concatenated text from all hidden DOM elements (comments, hidden inputs,
        CSS-hidden divs, scripts, noscript, meta keywords, event handlers, etc.)
    visible_text : str
        Text remaining after all hidden elements have been extracted.

    Returns
    -------
    dict
        Feature name -> numeric value mapping (15 features).
    """
    hidden_lower = (hidden_text or '').lower()
    hidden_words = _word_tokenize(hidden_text or '')
    n_hidden_words = max(len(hidden_words), 1)  # avoid div-by-zero

    # --- 1. Imperative verb count ---
    imperative_count = sum(1 for w in hidden_words if w in _IMPERATIVE_VERBS)

    # --- 2. Role assignment pattern count ---
    role_count = sum(len(p.findall(hidden_text or '')) for p in _ROLE_PATTERNS)

    # --- 3. System keyword count ---
    system_count = sum(1 for kw in _SYSTEM_KEYWORDS if kw in hidden_lower)

    # --- 4. Exfiltration URL count ---
    exfil_urls = _URL_RE.findall(hidden_text or '')
    exfil_url_count = len(exfil_urls)

    # --- 5. Instruction density (imperative verbs / total words) ---
    instruction_density = imperative_count / n_hidden_words

    # --- 6. Topic mismatch (1 - cosine similarity) ---
    cosine_sim = _tfidf_cosine(visible_text or '', hidden_text or '')
    topic_mismatch = 1.0 - cosine_sim

    # --- 7. Jailbreak phrase count ---
    jailbreak_count = sum(len(p.findall(hidden_text or '')) for p in _JAILBREAK_PHRASES)

    # --- 8. Encoding entropy of hidden text ---
    encoding_entropy = _shannon_entropy(hidden_text or '')

    # --- 9. Hidden text contains raw HTML tags ---
    has_html_tags = int(bool(re.search(r'<\s*(?:script|div|img|iframe|meta|input)\b', hidden_text or '', re.IGNORECASE)))

    # --- 10. Context injection pattern count ---
    context_count = sum(len(p.findall(hidden_text or '')) for p in _CONTEXT_PATTERNS)

    # --- 11. Data leak keyword count ---
    data_leak_count = sum(1 for kw in _DATA_LEAK_KEYWORDS if kw in hidden_lower)

    # --- 12. Question ratio in hidden text ---
    question_marks = (hidden_text or '').count('?')
    question_ratio = question_marks / n_hidden_words

    # --- 13. Payload length z-score (relative to typical hidden text) ---
    # Training-set statistics: mean ~150 chars, std ~300 chars for hidden text
    # These are approximate anchors; the ML model will learn the optimal boundary.
    hidden_len = len(hidden_text or '')
    payload_zscore = (hidden_len - 150.0) / 300.0 if hidden_len > 0 else 0.0

    # --- 14. Special character / encoding ratio ---
    encoding_matches = len(_ENCODING_RE.findall(hidden_text or ''))
    special_char_ratio = encoding_matches / max(len(hidden_text or ''), 1)

    # --- 15. Visible-hidden topic overlap (Jaccard of top terms) ---
    topic_overlap = _jaccard_top_terms(visible_text or '', hidden_text or '')

    return {
        'ipi_imperative_verb_count': imperative_count,
        'ipi_role_assignment_count': role_count,
        'ipi_system_keyword_count': system_count,
        'ipi_exfil_url_count': exfil_url_count,
        'ipi_instruction_density': round(instruction_density, 6),
        'ipi_topic_mismatch': round(topic_mismatch, 6),
        'ipi_jailbreak_phrase_count': jailbreak_count,
        'ipi_encoding_entropy': round(encoding_entropy, 4),
        'ipi_hidden_has_html_tags': has_html_tags,
        'ipi_context_injection_count': context_count,
        'ipi_data_leak_keywords': data_leak_count,
        'ipi_hidden_question_ratio': round(question_ratio, 6),
        'ipi_payload_length_zscore': round(payload_zscore, 4),
        'ipi_special_char_ratio': round(special_char_ratio, 6),
        'ipi_visible_hidden_topic_overlap': round(topic_overlap, 6),
    }
