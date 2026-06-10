#!/usr/bin/env python3
"""
IPI Knowledge-Based Features (v2)
==================================
Domain-expert features that directly measure the *linguistic and semantic
fingerprint* of Indirect Prompt Injection payloads in hidden text.

Inspired by Dr. Nir Nissim's approach in Lecture 8 (Linux malware detection)
where 171 hand-crafted behavioral features achieved TPR=0.997. These features
target the INTENT of hidden content, not just its structural envelope.

v2 improvements:
- Text normalization/decoding layer: strips JS wrappers, decodes HTML entities,
  URL encoding, unicode escapes, and base64 before KB analysis.
- 5 additional features targeting weak categories (JavaScript_Event,
  Data_Exfiltration, Encoding_Obfuscation).

Usage:
    from ipi_knowledge_features import extract_ipi_kb_features
    feats = extract_ipi_kb_features(hidden_text, visible_text)
"""

import base64
import html
import math
import re
from collections import Counter
from urllib.parse import unquote

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

# JavaScript wrapper patterns to strip
_JS_WRAPPER_RE = re.compile(
    r"""(?:alert|eval|prompt|confirm|console\.log|document\.write)\s*\(\s*['"](.+?)['"]\s*\)""",
    re.IGNORECASE | re.DOTALL,
)
_JS_PREFIX_RE = re.compile(r'^javascript:\s*', re.IGNORECASE)

# Base64 data URI pattern
_DATA_URI_RE = re.compile(
    r'data:[^;]*;base64,\s*([A-Za-z0-9+/=]+)',
    re.IGNORECASE,
)

# Suspicious external domain indicators (non-benign in hidden context)
_SUSPICIOUS_DOMAIN_RE = re.compile(
    r'https?://(?!(?:www\.)?(?:google|facebook|twitter|youtube|github|wikipedia|'
    r'cdn|cloudflare|googleapis|gstatic|jquery|bootstrapcdn|unpkg|jsdelivr)\b)'
    r'[a-z0-9.-]+\.[a-z]{2,}',
    re.IGNORECASE,
)


# ============================================================
# 2. Text normalization / decoding layer
# ============================================================

def _normalize_hidden_text(text: str) -> str:
    """Decode and unwrap hidden text to expose the actual payload content.

    This is critical for categories like Encoding_Obfuscation (HTML entities),
    JavaScript_Event (alert() wrappers), and Data_Exfiltration (base64 URIs).
    """
    if not text:
        return ''

    result = text

    # 1. Decode HTML entities: &#73;&#103; → "Ig..."
    result = html.unescape(result)

    # 2. Decode URL encoding: %49%67 → "Ig..."
    try:
        result = unquote(result)
    except Exception:
        pass

    # 3. Decode unicode escapes: \\u0065 → "e"
    def _decode_unicode(m):
        try:
            return chr(int(m.group(1), 16))
        except (ValueError, OverflowError):
            return m.group(0)
    result = re.sub(r'\\u([0-9a-fA-F]{4})', _decode_unicode, result)

    # 4. Strip JavaScript function wrappers: alert('payload') → payload
    for m in _JS_WRAPPER_RE.finditer(result):
        result = result + ' ' + m.group(1)

    # 5. Strip javascript: prefix
    result = _JS_PREFIX_RE.sub('', result)

    # 6. Decode base64 data URIs: data:text/html;base64,xxx → decoded
    for m in _DATA_URI_RE.finditer(text):  # use original text for b64
        try:
            decoded = base64.b64decode(m.group(1)).decode('utf-8', errors='replace')
            result = result + ' ' + decoded
        except Exception:
            pass

    return result


# ============================================================
# 3. Feature extraction functions
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
# 4. Main extraction function
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
        Feature name -> numeric value mapping (20 features).
    """
    # --- Normalize / decode hidden text to expose obfuscated payloads ---
    raw_hidden = hidden_text or ''
    decoded_hidden = _normalize_hidden_text(raw_hidden)

    hidden_lower = decoded_hidden.lower()
    hidden_words = _word_tokenize(decoded_hidden)
    n_hidden_words = max(len(hidden_words), 1)  # avoid div-by-zero

    # --- 1. Imperative verb count (on decoded text) ---
    imperative_count = sum(1 for w in hidden_words if w in _IMPERATIVE_VERBS)

    # --- 2. Role assignment pattern count ---
    role_count = sum(len(p.findall(decoded_hidden)) for p in _ROLE_PATTERNS)

    # --- 3. System keyword count ---
    system_count = sum(1 for kw in _SYSTEM_KEYWORDS if kw in hidden_lower)

    # --- 4. Exfiltration URL count ---
    exfil_urls = _URL_RE.findall(raw_hidden)
    exfil_url_count = len(exfil_urls)

    # --- 5. Instruction density (imperative verbs / total words) ---
    instruction_density = imperative_count / n_hidden_words

    # --- 6. Topic mismatch (1 - cosine similarity) ---
    cosine_sim = _tfidf_cosine(visible_text or '', decoded_hidden)
    topic_mismatch = 1.0 - cosine_sim

    # --- 7. Jailbreak phrase count ---
    jailbreak_count = sum(len(p.findall(decoded_hidden)) for p in _JAILBREAK_PHRASES)

    # --- 8. Encoding entropy of hidden text ---
    encoding_entropy = _shannon_entropy(raw_hidden)

    # --- 9. Hidden text contains raw HTML tags ---
    has_html_tags = int(bool(re.search(r'<\s*(?:script|div|img|iframe|meta|input)\b', raw_hidden, re.IGNORECASE)))

    # --- 10. Context injection pattern count ---
    context_count = sum(len(p.findall(decoded_hidden)) for p in _CONTEXT_PATTERNS)

    # --- 11. Data leak keyword count ---
    data_leak_count = sum(1 for kw in _DATA_LEAK_KEYWORDS if kw in hidden_lower)

    # --- 12. Question ratio in hidden text ---
    question_marks = decoded_hidden.count('?')
    question_ratio = question_marks / n_hidden_words

    # --- 13. Payload length z-score (relative to typical hidden text) ---
    hidden_len = len(raw_hidden)
    payload_zscore = (hidden_len - 150.0) / 300.0 if hidden_len > 0 else 0.0

    # --- 14. Special character / encoding ratio ---
    encoding_matches = len(_ENCODING_RE.findall(raw_hidden))
    special_char_ratio = encoding_matches / max(len(raw_hidden), 1)

    # --- 15. Visible-hidden topic overlap (Jaccard of top terms) ---
    topic_overlap = _jaccard_top_terms(visible_text or '', decoded_hidden)

    # ============================================================
    # v2 features — targeting weak categories
    # ============================================================

    # --- 16. JS wrapper count (JavaScript_Event category) ---
    # Count how many alert/eval/prompt wrappers contain text
    js_wrapper_count = len(_JS_WRAPPER_RE.findall(raw_hidden))

    # --- 17. Base64 payload detected (Data_Exfiltration category) ---
    base64_payloads = _DATA_URI_RE.findall(raw_hidden)
    has_base64_payload = int(len(base64_payloads) > 0)

    # --- 18. Suspicious external domain count ---
    # External domains in hidden text that aren't common CDNs/platforms
    suspicious_domains = _SUSPICIOUS_DOMAIN_RE.findall(raw_hidden)
    suspicious_domain_count = len(suspicious_domains)

    # --- 19. Decoded imperative density ---
    # Re-run imperative detection specifically on decoded base64/JS content
    # (this isolates the signal from noise in the main hidden text)
    extra_decoded = ''
    for m in _JS_WRAPPER_RE.finditer(raw_hidden):
        extra_decoded += ' ' + m.group(1)
    for m in _DATA_URI_RE.finditer(raw_hidden):
        try:
            extra_decoded += ' ' + base64.b64decode(m.group(1)).decode('utf-8', errors='replace')
        except Exception:
            pass
    extra_words = _word_tokenize(extra_decoded)
    decoded_imperative_count = sum(1 for w in extra_words if w in _IMPERATIVE_VERBS)

    # --- 20. Encoding obfuscation ratio ---
    # Ratio of encoded characters to total hidden text length
    # High ratio = likely Encoding_Obfuscation category
    raw_len = max(len(raw_hidden), 1)
    entity_count = len(re.findall(r'&#\d+;|&#x[0-9a-fA-F]+;', raw_hidden))
    url_enc_count = len(re.findall(r'%[0-9a-fA-F]{2}', raw_hidden))
    unicode_esc_count = len(re.findall(r'\\u[0-9a-fA-F]{4}', raw_hidden))
    total_encoded_chars = entity_count + url_enc_count + unicode_esc_count
    encoding_obfuscation_ratio = total_encoded_chars / raw_len

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
        # v2 features
        'ipi_js_wrapper_count': js_wrapper_count,
        'ipi_has_base64_payload': has_base64_payload,
        'ipi_suspicious_domain_count': suspicious_domain_count,
        'ipi_decoded_imperative_count': decoded_imperative_count,
        'ipi_encoding_obfuscation_ratio': round(encoding_obfuscation_ratio, 6),
    }
