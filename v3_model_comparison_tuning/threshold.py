"""Validation-set threshold sweep utilities for v3."""

import csv
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
THRESHOLD_COLUMNS = [
    "threshold",
    "precision",
    "recall",
    "f1",
    "false_positive_count",
    "false_negative_count",
]


def spam_probabilities(pipeline, X) -> list[float]:
    probabilities = pipeline.predict_proba(X)
    return probabilities[:, 1].tolist()


def sweep_thresholds(y_true, spam_probabilities, thresholds=DEFAULT_THRESHOLDS) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        predictions = [1 if probability >= threshold else 0 for probability in spam_probabilities]
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
                "false_positive_count": sum(actual == 0 and predicted == 1 for actual, predicted in zip(y_true, predictions)),
                "false_negative_count": sum(actual == 1 and predicted == 0 for actual, predicted in zip(y_true, predictions)),
            }
        )
    return pd.DataFrame(rows, columns=THRESHOLD_COLUMNS)


def save_threshold_analysis(table: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def predict_with_threshold(probability: float, threshold: float) -> int:
    return 1 if probability >= threshold else 0
