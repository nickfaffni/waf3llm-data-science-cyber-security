#!/usr/bin/env python3
"""
DS4CS — Phase 2: Feature Engineering (v3)
==========================================
Parses HTML to separate visible vs. hidden DOM streams and extracts
structural + semantic + graph-structural features for IPI classification.

Key improvements over v2:
- Graph-based DOM structural features: tree depth, branching factor,
  hidden-form proximity, text entropy — captures tree context that flat
  counts miss.
- Transformer semantic embeddings via all-MiniLM-L6-v2 (384-dim) for
  hidden text, supplementing the MinHash character n-gram pipeline.
- Internal <style> blocks are parsed so class-based hiding is detected.
- All <script> text is captured into the hidden stream.
- Switches to the C-optimized `lxml` parser when available.
"""

import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Comment
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from minhash_vectorizer import MinHashVectorizer
from ipi_knowledge_features import extract_ipi_kb_features

csv.field_size_limit(sys.maxsize)

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Parameters
# ============================================================
TFIDF_MAX_FEATURES = 5000
LSA_COMPONENTS = 50

# Pick the fastest available parser without forcing a hard dep on lxml
import importlib.util
PARSER = 'lxml' if importlib.util.find_spec('lxml') is not None else 'html.parser'

# ============================================================
# CSS helpers
# ============================================================
# Properties whose values mark an element as visually hidden.
_HIDING_RULES = {
    'display': {'none'},
    'visibility': {'hidden', 'collapse'},
    'opacity': {'0', '0.0', '0%'},
}

_ZERO_DIM_RE = re.compile(r'^\s*0(?:px|pt|em|rem|%|)?\s*$')
_OFFSCREEN_RE = re.compile(r'-?\d{4,}px')


def _strip_css_comments(s: str) -> str:
    return re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)


def _parse_declarations(style_text: str) -> dict:
    """Parse a `style=` attribute or block body into a dict of {prop: value}."""
    out = {}
    for decl in _strip_css_comments(style_text).split(';'):
        if ':' not in decl:
            continue
        prop, _, value = decl.partition(':')
        prop = prop.strip().lower()
        value = value.strip().lower()
        if value.endswith('!important'):
            value = value[: -len('!important')].strip()
        if prop and value:
            out[prop] = value
    return out


def _is_hidden_declarations(decls: dict) -> bool:
    for prop, hiding_values in _HIDING_RULES.items():
        v = decls.get(prop)
        if v is not None and any(v.startswith(h) for h in hiding_values):
            return True
    # Zero-size box
    if _ZERO_DIM_RE.match(decls.get('width', 'x')) and _ZERO_DIM_RE.match(decls.get('height', 'x')):
        return True
    if _ZERO_DIM_RE.match(decls.get('max-height', 'x')) and _ZERO_DIM_RE.match(decls.get('max-width', 'x')):
        return True
    if _ZERO_DIM_RE.match(decls.get('font-size', 'x')):
        return True
    # Off-screen positioning
    if decls.get('position') in ('absolute', 'fixed'):
        for k in ('left', 'right', 'top', 'bottom'):
            if _OFFSCREEN_RE.search(decls.get(k, '')):
                return True
    # CSS clip / clip-path tricks
    if decls.get('clip') == 'rect(0,0,0,0)' or decls.get('clip-path', '').startswith('inset(100%'):
        return True
    # Text-indent off-screen hiding
    if 'text-indent' in decls:
        val = decls['text-indent']
        if '-' in val:
            nums = re.findall(r'-?\d+', val)
            if nums and int(nums[0]) <= -100:
                return True
    # Transform translate off-screen hiding
    if 'transform' in decls:
        val = decls['transform']
        if 'translate' in val and '-' in val:
            nums = re.findall(r'-?\d+', val)
            if nums and int(nums[0]) <= -100:
                return True
    # Color matching background-color camouflage
    if 'color' in decls and ('background-color' in decls or 'background' in decls):
        bg = decls.get('background-color') or decls.get('background')
        if bg and decls['color'] == bg:
            return True
    return False


def _parse_stylesheet(style_block: str) -> dict:
    """Return {selector: declarations_dict} from a <style> block body."""
    out = {}
    cleaned = _strip_css_comments(style_block)
    for match in re.finditer(r'([^{}]+)\{([^{}]*)\}', cleaned):
        selectors, body = match.group(1).strip(), match.group(2).strip()
        decls = _parse_declarations(body)
        if not decls:
            continue
        for sel in selectors.split(','):
            out[sel.strip().lower()] = decls
    return out


def _selectors_for_element(tag) -> list:
    """Return CSS selector strings (.class, #id, tag) that could match this element."""
    selectors = [tag.name.lower()] if tag.name else []
    if tag.has_attr('id'):
        selectors.append('#' + tag['id'].lower())
    for cls in tag.get('class', []) or []:
        selectors.append('.' + cls.lower())
    return selectors


# ============================================================
# Graph-based DOM structural features
# ============================================================
def _dom_depths(tag, depth=0):
    """Yield (tag, depth) for every element in the subtree."""
    yield tag, depth
    for child in tag.find_all(True, recursive=False):
        yield from _dom_depths(child, depth + 1)


def _text_entropy(text: str) -> float:
    """Shannon entropy of the character frequency distribution."""
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


_INTERACTIVE_TAGS = frozenset({'form', 'button', 'a', 'select', 'textarea'})
_FORM_TAGS = frozenset({'form', 'input', 'textarea', 'select', 'button'})


def extract_dom_graph_features(html_content: str) -> dict:
    """Extract graph-theoretic structural features from the DOM tree.

    These features capture WHERE hidden elements sit relative to forms
    and interactive content — context that flat counts cannot express.
    """
    soup = BeautifulSoup(html_content or '', PARSER)

    # Collect all (tag, depth) pairs
    all_nodes = list(_dom_depths(soup, 0))
    tag_nodes = [(t, d) for t, d in all_nodes if hasattr(t, 'name') and t.name]

    if not tag_nodes:
        return {
            'dom_depth_max': 0,
            'dom_depth_mean': 0.0,
            'dom_node_count': 0,
            'dom_branching_factor': 0.0,
            'hidden_depth_mean': 0.0,
            'hidden_depth_relative': 0.0,
            'form_hidden_proximity': -1.0,
            'script_to_content_ratio': 0.0,
            'unique_tag_count': 0,
            'dom_text_entropy': 0.0,
            'hidden_in_form_count': 0,
            'has_suspicious_nesting': 0,
        }

    depths = [d for _, d in tag_nodes]
    max_depth = max(depths) if depths else 0
    mean_depth = sum(depths) / len(depths) if depths else 0.0
    node_count = len(tag_nodes)

    # Branching factor: mean children per non-leaf node
    children_counts = []
    for t, _ in tag_nodes:
        n_children = len(t.find_all(True, recursive=False))
        if n_children > 0:
            children_counts.append(n_children)
    branching = sum(children_counts) / len(children_counts) if children_counts else 0.0

    # Identify hidden elements (inline style or hidden input)
    hidden_depths = []
    hidden_tags_set = set()
    for t, d in tag_nodes:
        is_hidden = False
        if t.name in ('script', 'noscript', 'template'):
            is_hidden = True
        elif t.get('type') == 'hidden':
            is_hidden = True
        elif t.has_attr('style'):
            decls = _parse_declarations(t.get('style', ''))
            if _is_hidden_declarations(decls):
                is_hidden = True
        elif t.get('aria-hidden') == 'true':
            is_hidden = True
        if is_hidden:
            hidden_depths.append(d)
            hidden_tags_set.add(id(t))

    hidden_depth_mean = sum(hidden_depths) / len(hidden_depths) if hidden_depths else 0.0
    hidden_depth_rel = hidden_depth_mean / (max_depth + 1)

    # Form-hidden proximity: check if any hidden element is inside or adjacent to form elements
    form_tags = [(t, d) for t, d in tag_nodes if t.name in _FORM_TAGS]
    form_depths = [d for _, d in form_tags]
    hidden_in_form = 0
    min_proximity = -1.0
    has_suspicious = 0

    for t, d in tag_nodes:
        if id(t) not in hidden_tags_set:
            continue
        # Check if this hidden element is inside an interactive tag
        for parent in t.parents:
            if hasattr(parent, 'name') and parent.name in _INTERACTIVE_TAGS:
                has_suspicious = 1
                break
        # Check proximity to form elements (depth difference)
        for fd in form_depths:
            dist = abs(d - fd)
            if min_proximity < 0 or dist < min_proximity:
                min_proximity = float(dist)
        # Count hidden inside forms
        for parent in t.parents:
            if hasattr(parent, 'name') and parent.name == 'form':
                hidden_in_form += 1
                break

    # Script-to-content ratio
    script_nodes = sum(1 for t, _ in tag_nodes if t.name in ('script', 'noscript'))
    script_ratio = script_nodes / node_count if node_count > 0 else 0.0

    # Unique tag types
    unique_tags = len({t.name for t, _ in tag_nodes if t.name})

    # Text entropy of visible text
    visible_text = soup.get_text(separator=' ', strip=True)
    text_ent = _text_entropy(visible_text)

    return {
        'dom_depth_max': max_depth,
        'dom_depth_mean': round(mean_depth, 4),
        'dom_node_count': node_count,
        'dom_branching_factor': round(branching, 4),
        'hidden_depth_mean': round(hidden_depth_mean, 4),
        'hidden_depth_relative': round(hidden_depth_rel, 4),
        'form_hidden_proximity': min_proximity,
        'script_to_content_ratio': round(script_ratio, 4),
        'unique_tag_count': unique_tags,
        'dom_text_entropy': round(text_ent, 4),
        'hidden_in_form_count': hidden_in_form,
        'has_suspicious_nesting': has_suspicious,
    }


# ============================================================
# Feature extraction
# ============================================================
def parse_html_features(html_content: str) -> dict:
    """Parse HTML; separate hidden vs. visible streams; emit structural features."""
    soup = BeautifulSoup(html_content or '', PARSER)

    hidden_text_parts = []
    hidden_element_count = 0
    css_trick_count = 0
    comment_count = 0
    hidden_input_count = 0
    script_count = 0
    event_handler_count = 0

    # 7. Inline event-handler attributes (onload, onerror, onclick, ...)
    #    These are a common IPI vector (svg/body onload, img onerror, anchor href=javascript:).
    for tag in list(soup.find_all(True)):
        captured_here = False
        for attr_name in list(tag.attrs.keys()):
            if isinstance(attr_name, str) and attr_name.lower().startswith('on'):
                handler_value = tag.attrs.get(attr_name, '')
                if isinstance(handler_value, list):
                    handler_value = ' '.join(handler_value)
                if handler_value:
                    hidden_text_parts.append(str(handler_value))
                event_handler_count += 1
                captured_here = True
        href = tag.get('href', '') if hasattr(tag, 'get') else ''
        if isinstance(href, str) and href.lower().startswith('javascript:'):
            hidden_text_parts.append(href)
            event_handler_count += 1
            captured_here = True
        if captured_here:
            hidden_element_count += 1

    # 1. HTML comments
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        hidden_text_parts.append(str(c))
        comment_count += 1
        c.extract()

    # 2. Hidden inputs
    for inp in soup.find_all('input', type='hidden'):
        v = inp.get('value', '')
        if v:
            hidden_text_parts.append(v)
        hidden_input_count += 1
        inp.extract()

    # 3. Meta keywords
    for meta in soup.find_all('meta', attrs={'name': 'keywords'}):
        c = meta.get('content', '')
        if c:
            hidden_text_parts.append(c)
        meta.extract()

    # 4. Stylesheet-defined hidden selectors
    hidden_selectors = {}
    for style_tag in list(soup.find_all('style')):
        body = style_tag.string or style_tag.get_text() or ''
        rules = _parse_stylesheet(body)
        for sel, decls in rules.items():
            if _is_hidden_declarations(decls):
                hidden_selectors[sel] = decls
        style_tag.extract()

    if hidden_selectors:
        for tag in soup.find_all(True):
            for sel in _selectors_for_element(tag):
                if sel in hidden_selectors:
                    css_trick_count += 1
                    hidden_element_count += 1
                    hidden_text_parts.append(tag.get_text(separator=' ', strip=True))
                    tag.extract()
                    break

    # 5. <script> and <noscript> — capture text from all, extract from all
    for script in list(soup.find_all(['script', 'noscript'])):
        text = script.get_text(separator=' ', strip=True)
        if text:
            hidden_text_parts.append(text)
        script_count += 1
        script.extract()

    # 5b. <template> tags — capture inert template layout text
    for template in list(soup.find_all('template')):
        text = template.get_text(separator=' ', strip=True)
        if text:
            hidden_text_parts.append(text)
        hidden_element_count += 1
        template.extract()

    # 5c. aria-hidden="true" elements
    for aria_hid in list(soup.find_all(attrs={'aria-hidden': 'true'})):
        text = aria_hid.get_text(separator=' ', strip=True)
        if text:
            hidden_text_parts.append(text)
        hidden_element_count += 1
        aria_hid.extract()

    # 6. Inline style= hiding tricks
    for tag in list(soup.find_all(style=True)):
        decls = _parse_declarations(tag['style'])
        if _is_hidden_declarations(decls):
            css_trick_count += 1
            hidden_element_count += 1
            hidden_text_parts.append(tag.get_text(separator=' ', strip=True))
            tag.extract()


    # 8. Zero-size images / iframes (data exfiltration)
    def _is_zero(tag):
        if tag.name not in ('img', 'iframe'):
            return False
        return tag.get('width') == '0' and tag.get('height') == '0'

    for tag in list(soup.find_all(_is_zero)):
        src = tag.get('src', '')
        if src:
            hidden_text_parts.append(src)
        hidden_element_count += 1
        tag.extract()

    # Remaining text in soup is "visible"
    visible_text = soup.get_text(separator=' ', strip=True)
    hidden_text = ' '.join(hidden_text_parts)
    hidden_text_len = len(hidden_text)
    visible_text_len = len(visible_text)
    ratio = hidden_text_len / (visible_text_len + 1)

    # Graph-based DOM structural features
    graph_feats = extract_dom_graph_features(html_content)

    # Knowledge-based IPI features (semantic intent analysis)
    kb_feats = extract_ipi_kb_features(hidden_text, visible_text)

    result = {
        'hidden_element_count': hidden_element_count,
        'css_trick_count': css_trick_count,
        'comment_count': comment_count,
        'hidden_input_count': hidden_input_count,
        'script_count': script_count,
        'event_handler_count': event_handler_count,
        'hidden_text_length': hidden_text_len,
        'visible_text_length': visible_text_len,
        'hidden_to_visible_ratio': ratio,
        'has_visible_text': int(visible_text_len > 0),
        'hidden_text': hidden_text,
    }
    result.update(graph_feats)
    result.update(kb_feats)
    return result


# ============================================================
# Driver
# ============================================================
def process_file(filepath: Path) -> pd.DataFrame:
    print(f'Processing {filepath}...')
    df = pd.read_csv(filepath)
    rows = []
    for idx, row in enumerate(df.itertuples(index=False)):
        if idx % 200 == 0:
            print(f'  Parsed {idx}/{len(df)} HTML documents...')
        feats = parse_html_features(str(getattr(row, 'HTML_Content', '') or ''))
        feats['Label'] = getattr(row, 'Label', 0)
        feats['Category'] = getattr(row, 'Category', '') or ''
        feats['Technique'] = getattr(row, 'Technique', '') if hasattr(row, 'Technique') else ''
        feats['URL'] = getattr(row, 'URL', '') or ''
        rows.append(feats)
    return pd.DataFrame(rows)


def build_semantic_pipeline() -> Pipeline:
    return Pipeline([
        ('minhash', MinHashVectorizer(
            n_components=LSA_COMPONENTS,
            ngram_range=(3, 5),
            analyzer='char',
            random_state=42,
        )),
    ])


def main():
    parser = argparse.ArgumentParser(description='DS4CS Feature Extraction')
    parser.add_argument('--num-benign', type=int, default=3000,
                        help='Number of benign HTML pages used in generation')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='Malicious ratio used in dataset generation')
    parser.add_argument('--skip-transformer', action='store_true',
                        help='Skip transformer embeddings (faster, for debugging)')
    args = parser.parse_args()

    num_benign = args.num_benign
    ratio_pct = int(args.ratio * 100)

    train_file = DATA_DIR / f'train_real_html.csv'
    test_file = DATA_DIR / f'test_real_html.csv'

    if not train_file.exists():
        print(f'Train file not found: {train_file}. Run Phase 1 data generation first.')
        return

    print(f'=== Extracting Structural + Graph Features from {train_file.name} / {test_file.name} ===')
    df_train = process_file(train_file)
    df_test = process_file(test_file)

    # ---- MinHash semantic pipeline ----
    print('\n=== Fitting Semantic Pipeline (MinHash) ===')
    pipeline = build_semantic_pipeline()
    lsa_train = pipeline.fit_transform(df_train['hidden_text'].fillna(''))
    lsa_test = pipeline.transform(df_test['hidden_text'].fillna(''))
    if 'svd' in pipeline.named_steps:
        explained = pipeline.named_steps['svd'].explained_variance_ratio_.sum()
        print(f'Explained variance by {LSA_COMPONENTS} LSA components: {explained:.4f}')
    else:
        print('Using MinHash pipeline (no SVD explained variance to display).')

    pipeline_path = MODELS_DIR / f'semantic_pipeline.joblib'
    joblib.dump(pipeline, pipeline_path)
    print(f'Saved semantic pipeline to {pipeline_path}')

    lsa_cols = [f'lsa_embed_{i}' for i in range(LSA_COMPONENTS)]
    lsa_train_df = pd.DataFrame(lsa_train, columns=lsa_cols)
    lsa_test_df = pd.DataFrame(lsa_test, columns=lsa_cols)

    # ---- Transformer semantic embeddings ----
    sbert_train_df = pd.DataFrame()
    sbert_test_df = pd.DataFrame()
    if not args.skip_transformer:
        print('\n=== Computing Transformer Semantic Embeddings (all-MiniLM-L6-v2) ===')
        from transformer_embeddings import TransformerEmbedder, EMBED_DIM
        embedder = TransformerEmbedder()

        print(f'  Encoding {len(df_train)} training hidden texts...')
        sbert_train = embedder.encode(df_train['hidden_text'].fillna('').tolist())
        print(f'  Encoding {len(df_test)} test hidden texts...')
        sbert_test = embedder.encode(df_test['hidden_text'].fillna('').tolist())

        sbert_cols = [f'sbert_embed_{i}' for i in range(EMBED_DIM)]
        sbert_train_df = pd.DataFrame(sbert_train, columns=sbert_cols)
        sbert_test_df = pd.DataFrame(sbert_test, columns=sbert_cols)
        print(f'  Transformer embeddings shape: train {sbert_train.shape}, test {sbert_test.shape}')

        # Save the embedder config for inference reuse
        joblib.dump({'model_name': embedder.model_name, 'embed_dim': EMBED_DIM},
                    MODELS_DIR / f'sbert_config.joblib')
    else:
        print('\n(Skipping transformer embeddings — --skip-transformer flag set)')

    # ---- Assemble final feature matrix ----
    df_train = df_train.drop(columns=['hidden_text']).reset_index(drop=True)
    df_test = df_test.drop(columns=['hidden_text']).reset_index(drop=True)

    parts_train = [df_train, lsa_train_df]
    parts_test = [df_test, lsa_test_df]
    if not sbert_train_df.empty:
        parts_train.append(sbert_train_df)
        parts_test.append(sbert_test_df)

    df_train_final = pd.concat(parts_train, axis=1)
    df_test_final = pd.concat(parts_test, axis=1)

    out_train = DATA_DIR / f'features_train.csv'
    out_test = DATA_DIR / f'features_test.csv'
    df_train_final.to_csv(out_train, index=False)
    df_test_final.to_csv(out_test, index=False)
    print(f'\nSaved: {out_train} (Shape: {df_train_final.shape})')
    print(f'Saved: {out_test} (Shape: {df_test_final.shape})')

    # Backward compatibility override
    if num_benign == 3000 and ratio_pct == 20:
        pipeline_path_compat = MODELS_DIR / 'semantic_pipeline.joblib'
        joblib.dump(pipeline, pipeline_path_compat)
        out_train_compat = DATA_DIR / 'features_train.csv'
        out_test_compat = DATA_DIR / 'features_test.csv'
        df_train_final.to_csv(out_train_compat, index=False)
        df_test_final.to_csv(out_test_compat, index=False)
        print(f'Saved backward-compatible features and pipeline files.')

    print('Phase 2 Feature Engineering Complete.')


if __name__ == '__main__':
    main()
