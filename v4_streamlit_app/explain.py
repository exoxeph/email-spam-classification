"""Rule-based explanation bullets for v4 predictions."""

from __future__ import annotations

import re

import pandas as pd

from common.features import SUSPICIOUS_WORDS, build_metadata_features

FALLBACK_EXPLANATION = "No strong suspicious signals detected."
_WORD_RE = re.compile(r"\b[\w']+\b")


def build_explanation(text: str, subject: str = "", metadata: dict | None = None) -> list[str]:
    feature_values = metadata if metadata is not None else _metadata_for_text(text, subject)
    bullets = []

    link_count = int(feature_values.get("link_count", 0))
    if link_count:
        bullets.append(f"Contains {link_count} link(s).")

    matched_words = suspicious_words_in_text(f"{subject} {text}")
    if matched_words:
        bullets.append(f"Contains suspicious word(s): {', '.join(matched_words)}.")

    exclamation_count = int(feature_values.get("exclamation_count", 0))
    if exclamation_count:
        bullets.append(f"Uses {exclamation_count} exclamation mark(s).")

    uppercase_ratio = float(feature_values.get("uppercase_ratio", 0))
    if uppercase_ratio > 0.3:
        bullets.append("Uses an unusually high uppercase ratio.")

    currency_count = int(feature_values.get("currency_symbol_count", 0))
    if currency_count:
        bullets.append("Contains currency symbol(s).")

    return bullets or [FALLBACK_EXPLANATION]


def suspicious_words_in_text(text: str) -> list[str]:
    found = {match.group(0).lower() for match in _WORD_RE.finditer(text) if match.group(0).lower() in SUSPICIOUS_WORDS}
    return sorted(found)


def _metadata_for_text(text: str, subject: str) -> dict:
    frame = pd.DataFrame({"subject": [subject], "body": [text]})
    return build_metadata_features(frame).iloc[0].to_dict()
