import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from v1_basic_pipeline.evaluate import compute_metrics, evaluate_all, save_reports


def _toy_fitted_pipeline():
    X_train = pd.Series(["free money now", "win a prize", "meeting at 2pm", "see you tomorrow"] * 5)
    y_train = pd.Series([1, 1, 0, 0] * 5)
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipeline.fit(X_train, y_train)
    return pipeline


def test_compute_metrics_returns_expected_keys():
    pipeline = _toy_fitted_pipeline()
    X_test = pd.Series(["free prize now", "see you at the office"])
    y_test = pd.Series([1, 0])

    metrics = compute_metrics(pipeline, X_test, y_test)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "confusion_matrix"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["confusion_matrix"][0]) == 2


def test_evaluate_all_covers_every_model():
    pipeline = _toy_fitted_pipeline()
    fitted = {"model_a": pipeline, "model_b": pipeline}
    X_test = pd.Series(["free prize now", "see you at the office"])
    y_test = pd.Series([1, 0])

    all_metrics = evaluate_all(fitted, X_test, y_test)

    assert set(all_metrics.keys()) == {"model_a", "model_b"}


def test_save_reports_writes_metrics_json_and_comparison_csv(tmp_path):
    all_metrics = {
        "model_a": {"accuracy": 0.9, "precision": 0.8, "recall": 0.85, "f1": 0.82, "confusion_matrix": [[8, 1], [1, 10]]},
        "model_b": {"accuracy": 0.95, "precision": 0.9, "recall": 0.92, "f1": 0.91, "confusion_matrix": [[9, 0], [1, 10]]},
    }

    save_reports(all_metrics, best_name="model_b", reports_dir=tmp_path)

    metrics_json = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics_json["best_model"] == "model_b"
    assert metrics_json["f1"] == 0.91

    comparison_csv = (tmp_path / "model_comparison.csv").read_text()
    assert "model_a" in comparison_csv
    assert "model_b" in comparison_csv
