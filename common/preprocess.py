"""clean_text() and map_label() - basic text cleaning and label mapping shared by all versions."""

import re

import pandas as pd


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_label(label: str) -> int:
    normalized = label.strip().lower()
    if normalized == "spam":
        return 1
    if normalized == "ham":
        return 0
    raise ValueError(f"Unexpected label value: {label!r}")


def build_dataset(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    df = df.copy()
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df = df[df["body"].str.strip() != ""]

    combined = (df["subject"] + " " + df["body"]).apply(clean_text)
    labels = df["label"].apply(map_label)

    combined = combined.drop_duplicates()
    labels = labels.loc[combined.index]

    return combined.reset_index(drop=True), labels.reset_index(drop=True)
