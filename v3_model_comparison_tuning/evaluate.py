"""Evaluation and report writing for v3."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, spam_probabilities, threshold: float) -> dict:
    predictions = [1 if probability >= threshold else 0 for probability in spam_probabilities]
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(y_true, predictions, target_names=["ham", "spam"], output_dict=True, zero_division=0),
    }


def evaluate_all(fitted_pipelines: dict, X_val, y_val) -> dict:
    metrics = {}
    for name, pipeline in fitted_pipelines.items():
        predictions = pipeline.predict(X_val)
        metrics[name] = {
            "accuracy": accuracy_score(y_val, predictions),
            "precision": precision_score(y_val, predictions, zero_division=0),
            "recall": recall_score(y_val, predictions, zero_division=0),
            "f1": f1_score(y_val, predictions, zero_division=0),
        }
    return metrics


def save_model_comparison(all_metrics: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "precision", "recall", "f1"])
        for name, metrics in all_metrics.items():
            writer.writerow([name, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]])


def save_metrics(metrics: dict, best_model: str, path: Path) -> None:
    payload = {"best_model": best_model, **metrics}
    Path(path).write_text(json.dumps(payload, indent=2))


def save_confusion_matrix(y_true, spam_probabilities, threshold: float, path: Path) -> None:
    predictions = [1 if probability >= threshold else 0 for probability in spam_probabilities]
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, predictions), display_labels=["ham", "spam"])
    display.plot(cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_precision_recall_curve(y_true, spam_probabilities, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, spam_probabilities)
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_error_analysis_template(path: Path, best_model: str, decision_threshold: float, low_threshold: float, high_threshold: float) -> None:
    text = f"""# v3 Error Analysis

Best validation model: `{best_model}`

Chosen thresholds:
- Low risk: spam probability < {low_threshold:.2f}
- Medium risk: {low_threshold:.2f} <= spam probability < {high_threshold:.2f}
- High risk: spam probability >= {high_threshold:.2f}
- Classification threshold used for final metrics: {decision_threshold:.2f}

Reasoning:
The threshold sweep is written to `threshold_analysis.csv`. The classification threshold is {decision_threshold:.2f} because it had the strongest validation F1 among the swept thresholds while keeping recall high. The test set remained untouched until the final evaluation pass.

Manual review notes:
- Review `false_positives.csv` for legitimate messages that look promotional or urgent.
- Review `false_negatives.csv` for spam that uses neutral wording or avoids obvious spam indicators.
- Add human categories here after inspecting the exported rows; this file intentionally avoids auto-tagging errors.
"""
    Path(path).write_text(text)
