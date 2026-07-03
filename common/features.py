"""Metadata feature extractors and the shared v2+ ColumnTransformer builder."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from common.preprocess import clean_text

METADATA_COLUMNS = [
    "email_char_length",
    "email_word_count",
    "subject_char_length",
    "subject_exclaim_count",
    "link_count",
    "exclamation_count",
    "uppercase_ratio",
    "suspicious_word_count",
    "currency_symbol_count",
    "digit_count",
]

SUSPICIOUS_WORDS = {
    "urgent",
    "winner",
    "free",
    "claim",
    "prize",
    "cash",
    "verify",
    "account",
    "limited",
    "offer",
    "click",
    "password",
    "login",
    "reward",
}

_LINK_RE = re.compile(r"\b(?:https?://|www\.)\S+|\b\S+\.(?:com|net|org)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"\b[\w']+\b")


def count_links(text: str) -> int:
    return len(_LINK_RE.findall(_coerce_text(text)))


def count_exclamations(text: str) -> int:
    return _coerce_text(text).count("!")


def uppercase_ratio(text: str) -> float:
    letters = [char for char in _coerce_text(text) if char.isalpha()]
    if not letters:
        return 0.0
    return sum(char.isupper() for char in letters) / len(letters)


def count_suspicious_words(text: str, suspicious_words: Iterable[str] = SUSPICIOUS_WORDS) -> int:
    suspicious = {word.lower() for word in suspicious_words}
    words = (match.group(0).lower() for match in _WORD_RE.finditer(_coerce_text(text)))
    return sum(word in suspicious for word in words)


def count_currency_symbols(text: str) -> int:
    return sum(char in "$\u00a3\u20ac\u09f3" for char in _coerce_text(text))


def count_digits(text: str) -> int:
    return sum(char.isdigit() for char in _coerce_text(text))


def build_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(df, required={"subject", "body"})
    subject = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    full_text = (subject + " " + body).str.strip()

    features = pd.DataFrame(index=df.index)
    features["email_char_length"] = body.str.len()
    features["email_word_count"] = body.apply(lambda text: len(_WORD_RE.findall(text)))
    features["subject_char_length"] = subject.str.len()
    features["subject_exclaim_count"] = subject.apply(count_exclamations)
    features["link_count"] = full_text.apply(count_links)
    features["exclamation_count"] = full_text.apply(count_exclamations)
    features["uppercase_ratio"] = full_text.apply(uppercase_ratio)
    features["suspicious_word_count"] = full_text.apply(count_suspicious_words)
    features["currency_symbol_count"] = full_text.apply(count_currency_symbols)
    features["digit_count"] = full_text.apply(count_digits)
    return features[METADATA_COLUMNS]


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(df, required={"subject", "body"})
    raw = df[["subject", "body"]].fillna("").astype(str).copy()
    metadata = build_metadata_features(raw)

    cleaned_subject = raw["subject"].apply(clean_text)
    cleaned_body = raw["body"].apply(clean_text)
    feature_frame = metadata.copy()
    feature_frame["combined_text"] = (cleaned_subject + " " + cleaned_body).str.strip()
    return feature_frame


def build_column_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "word_tfidf",
                TfidfVectorizer(ngram_range=(1, 1), min_df=2, max_features=30_000, sublinear_tf=True),
                "combined_text",
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
                "combined_text",
            ),
            ("metadata", StandardScaler(with_mean=False), METADATA_COLUMNS),
        ]
    )


def _coerce_text(text: str) -> str:
    return "" if pd.isna(text) else str(text)


def _validate_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
