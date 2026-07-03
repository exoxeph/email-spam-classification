"""Entry point for v2: load data, engineer features, train, evaluate, and save best model."""

from pathlib import Path

import pandas as pd

from common.config import RAW_DATA_DIR
from common.download_data import download_data
from common.features import build_feature_frame
from common.io_utils import load_dataset, save_artifact
from common.preprocess import map_label
from v2_feature_engineering.evaluate import evaluate_all, save_reports
from v2_feature_engineering.train import build_pipelines, fit_pipelines, select_best, split_dataset

PIPELINE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PIPELINE_DIR / "models"
REPORTS_DIR = PIPELINE_DIR / "reports"


def build_dataset_v2(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = df.copy()
    data["subject"] = data["subject"].fillna("").astype(str)
    data["body"] = data["body"].fillna("").astype(str)
    data = data[data["body"].str.strip() != ""]
    data["label"] = data["label"].apply(map_label)
    data = data.drop_duplicates(subset=["subject", "body", "label"])

    X = build_feature_frame(data)
    y = data["label"].reset_index(drop=True)
    return X.reset_index(drop=True), y


def main() -> None:
    csv_path = download_data(RAW_DATA_DIR)
    df = load_dataset(csv_path)
    X, y = build_dataset_v2(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    pipelines = build_pipelines()
    fitted = fit_pipelines(pipelines, X_train, y_train)
    best_name = select_best(fitted, X_test, y_test)

    all_metrics = evaluate_all(fitted, X_test, y_test)
    save_reports(all_metrics, best_name, REPORTS_DIR)

    save_artifact(fitted[best_name], MODELS_DIR / "best_model_v2.pkl")

    print(f"Best model: {best_name}")
    print(f"F1: {all_metrics[best_name]['f1']:.4f}")
    print(f"Accuracy: {all_metrics[best_name]['accuracy']:.4f}")


if __name__ == "__main__":
    main()
