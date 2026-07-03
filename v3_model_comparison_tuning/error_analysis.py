"""Export raw false positives and false negatives for manual review."""

from pathlib import Path

import pandas as pd


ERROR_COLUMNS = ["email_text", "actual_label", "predicted_label", "spam_probability"]


def build_prediction_frame(source_df: pd.DataFrame, y_true, spam_probabilities, threshold: float) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "email_text": (source_df["subject"].fillna("").astype(str) + " " + source_df["body"].fillna("").astype(str)).str.strip(),
            "actual_label": list(y_true),
            "spam_probability": list(spam_probabilities),
        }
    )
    frame["predicted_label"] = (frame["spam_probability"] >= threshold).astype(int)
    return frame[ERROR_COLUMNS]


def false_positives(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    return prediction_frame[
        (prediction_frame["actual_label"] == 0) & (prediction_frame["predicted_label"] == 1)
    ].reset_index(drop=True)


def false_negatives(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    return prediction_frame[
        (prediction_frame["actual_label"] == 1) & (prediction_frame["predicted_label"] == 0)
    ].reset_index(drop=True)


def save_error_exports(prediction_frame: pd.DataFrame, reports_dir: Path) -> None:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    false_positives(prediction_frame).to_csv(reports_dir / "false_positives.csv", index=False)
    false_negatives(prediction_frame).to_csv(reports_dir / "false_negatives.csv", index=False)
