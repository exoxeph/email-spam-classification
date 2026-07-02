"""Computes accuracy, precision, recall, F1, and confusion matrix for both v1 candidate models; writes reports/metrics.json and model_comparison.csv."""

import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(pipeline, X_test, y_test) -> dict:
    predictions = pipeline.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def evaluate_all(fitted_pipelines: dict, X_test, y_test) -> dict:
    return {name: compute_metrics(pipeline, X_test, y_test) for name, pipeline in fitted_pipelines.items()}


def save_reports(all_metrics: dict, best_name: str, reports_dir: Path) -> None:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {"best_model": best_name, **all_metrics[best_name]}
    (reports_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    with open(reports_dir / "model_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "precision", "recall", "f1"])
        for name, metrics in all_metrics.items():
            writer.writerow([name, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]])
