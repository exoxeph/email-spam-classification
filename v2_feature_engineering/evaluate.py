"""Evaluate v2 candidates and write metrics, comparison, and feature summary reports."""

import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from common.features import METADATA_COLUMNS


def compute_metrics(pipeline, X_test, y_test) -> dict:
    predictions = pipeline.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def evaluate_all(fitted_pipelines: dict, X_test, y_test) -> dict:
    return {name: compute_metrics(pipeline, X_test, y_test) for name, pipeline in fitted_pipelines.items()}


def save_reports(all_metrics: dict, best_name: str, reports_dir: Path) -> None:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {"best_model": best_name, **all_metrics[best_name]}
    (reports_dir / "metrics_v2.json").write_text(json.dumps(metrics_payload, indent=2))

    with open(reports_dir / "model_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "precision", "recall", "f1"])
        for name, metrics in all_metrics.items():
            writer.writerow([name, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]])

    write_feature_summary(reports_dir / "feature_summary.md")


def write_feature_summary(path: Path) -> None:
    lines = [
        "# v2 Feature Summary",
        "",
        "Text features:",
        "- word_tfidf: word unigrams on cleaned subject+body",
        "- char_tfidf: character 3-5 grams on cleaned subject+body",
        "",
        f"Metadata features ({len(METADATA_COLUMNS)} columns):",
    ]
    lines.extend(f"- {column}" for column in METADATA_COLUMNS)
    path.write_text("\n".join(lines) + "\n")
