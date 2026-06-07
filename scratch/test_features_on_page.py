import sys
from pathlib import Path
import json

BASE_DIR = Path('/Users/j.a.r.v.i.s/Research/Data Science for Cyber-Security 2026')
sys.path.insert(0, str(BASE_DIR / 'scripts' / 'feature_engineering'))
from extract_features import parse_html_features

html_content = (BASE_DIR / 'docs' / 'example_injected_page.html').read_text()
feats = parse_html_features(html_content)

print("=== WAF3LLM Structural Feature Extraction ===")
for k, v in feats.items():
    if k != 'hidden_text':
        print(f"{k:25} : {v}")

print("\n=== Extracted Hidden Text Payload (What the LLM sees) ===")
print(feats.get('hidden_text', ''))
