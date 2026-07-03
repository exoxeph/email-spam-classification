import json

import pandas as pd

from v2_feature_engineering.evaluate import compute_metrics, save_reports
from v2_feature_engineering.main import build_dataset_v2
from v2_feature_engineering.predict import predict
from v2_feature_engineering.train import build_pipelines, fit_pipelines, select_best, split_dataset


def _toy_dataset():
    spam_subjects = [f"WIN {i}" for i in range(12)]
    ham_subjects = [f"Meeting {i}" for i in range(12)]
    spam_bodies = [f"FREE cash click now claim prize {i} at x{i}.com" for i in range(12)]
    ham_bodies = [f"project meeting notes lunch schedule {i}" for i in range(12)]
    df = pd.DataFrame(
        {
            "subject": spam_subjects + ham_subjects,
            "body": spam_bodies + ham_bodies,
            "label": ["spam"] * 12 + ["ham"] * 12,
        }
    )
    return build_dataset_v2(df)


def test_build_dataset_v2_returns_feature_frame_and_labels():
    X, y = _toy_dataset()

    assert "combined_text" in X.columns
    assert set(y.unique()) == {0, 1}


def test_v2_pipelines_fit_and_select_best():
    X, y = _toy_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    fitted = fit_pipelines(build_pipelines(), X_train, y_train)
    best_name = select_best(fitted, X_test, y_test)

    assert set(fitted.keys()) == {"logistic_regression", "linear_svm"}
    assert best_name in fitted


def test_compute_metrics_and_predict_with_fitted_v2_pipeline():
    X, y = _toy_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    pipeline = fit_pipelines(build_pipelines(), X_train, y_train)["logistic_regression"]

    metrics = compute_metrics(pipeline, X_test, y_test)
    prediction = predict("FREE cash prize now", pipeline, subject="Winner")

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "confusion_matrix"}
    assert prediction in {"spam", "not spam"}


def test_save_reports_writes_v2_outputs(tmp_path):
    all_metrics = {
        "logistic_regression": {
            "accuracy": 0.9,
            "precision": 0.8,
            "recall": 0.85,
            "f1": 0.82,
            "confusion_matrix": [[8, 1], [1, 10]],
        },
        "linear_svm": {
            "accuracy": 0.95,
            "precision": 0.9,
            "recall": 0.92,
            "f1": 0.91,
            "confusion_matrix": [[9, 0], [1, 10]],
        },
    }

    save_reports(all_metrics, best_name="linear_svm", reports_dir=tmp_path)

    metrics = json.loads((tmp_path / "metrics_v2.json").read_text())
    assert metrics["best_model"] == "linear_svm"
    assert (tmp_path / "model_comparison.csv").exists()
    assert (tmp_path / "feature_summary.md").exists()
