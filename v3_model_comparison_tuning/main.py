"""Run v3 model comparison, threshold analysis, final evaluation, and error export."""

from pathlib import Path

import pandas as pd

from common.config import RAW_DATA_DIR
from common.download_data import download_data
from common.features import build_feature_frame
from common.io_utils import load_dataset, save_artifact
from common.preprocess import map_label
from v3_model_comparison_tuning.error_analysis import build_prediction_frame, save_error_exports
from v3_model_comparison_tuning.evaluate import (
    compute_metrics,
    evaluate_all,
    save_confusion_matrix,
    save_metrics,
    save_model_comparison,
    save_precision_recall_curve,
    write_error_analysis_template,
)
from v3_model_comparison_tuning.predict import CLASSIFICATION_THRESHOLD, HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD
from v3_model_comparison_tuning.threshold import save_threshold_analysis, spam_probabilities, sweep_thresholds
from v3_model_comparison_tuning.train import build_pipelines, fit_pipelines, select_best, split_dataset

PIPELINE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PIPELINE_DIR / "models"
REPORTS_DIR = PIPELINE_DIR / "reports"


def build_dataset_v3(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    data = df.copy()
    data["subject"] = data["subject"].fillna("").astype(str)
    data["body"] = data["body"].fillna("").astype(str)
    data = data[data["body"].str.strip() != ""]
    data["label"] = data["label"].apply(map_label)
    data = data.drop_duplicates(subset=["subject", "body", "label"]).reset_index(drop=True)

    X = build_feature_frame(data)
    y = data["label"].reset_index(drop=True)
    source = data[["subject", "body", "label"]].reset_index(drop=True)
    return X.reset_index(drop=True), y, source


def main() -> None:
    csv_path = download_data(RAW_DATA_DIR)
    df = load_dataset(csv_path)
    X, y, source = build_dataset_v3(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    source_test = source.loc[X_test.index].reset_index(drop=True)

    fitted = fit_pipelines(build_pipelines(), X_train, y_train)
    val_metrics = evaluate_all(fitted, X_val, y_val)
    save_model_comparison(val_metrics, REPORTS_DIR / "model_comparison.csv")

    best_name = select_best(fitted, X_val, y_val)
    best_pipeline = fitted[best_name]

    val_probabilities = spam_probabilities(best_pipeline, X_val)
    threshold_table = sweep_thresholds(y_val, val_probabilities)
    save_threshold_analysis(threshold_table, REPORTS_DIR / "threshold_analysis.csv")

    test_probabilities = spam_probabilities(best_pipeline, X_test)
    test_metrics = compute_metrics(y_test, test_probabilities, CLASSIFICATION_THRESHOLD)
    save_metrics(test_metrics, best_name, REPORTS_DIR / "metrics_v3.json")
    save_confusion_matrix(y_test, test_probabilities, CLASSIFICATION_THRESHOLD, REPORTS_DIR / "confusion_matrix_v3.png")
    save_precision_recall_curve(y_test, test_probabilities, REPORTS_DIR / "precision_recall_curve.png")

    prediction_frame = build_prediction_frame(source_test, y_test, test_probabilities, CLASSIFICATION_THRESHOLD)
    save_error_exports(prediction_frame, REPORTS_DIR)
    write_error_analysis_template(
        REPORTS_DIR / "error_analysis.md",
        best_name,
        CLASSIFICATION_THRESHOLD,
        LOW_RISK_THRESHOLD,
        HIGH_RISK_THRESHOLD,
    )

    save_artifact(best_pipeline, MODELS_DIR / "best_model_v3.pkl")

    print(f"Best model: {best_name}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
