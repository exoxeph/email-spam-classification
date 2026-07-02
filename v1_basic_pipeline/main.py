"""Entry point: download data if missing, then run load -> preprocess -> train -> evaluate."""

from pathlib import Path

from common.config import RAW_DATA_DIR
from common.download_data import download_data
from common.io_utils import load_dataset, save_artifact
from common.preprocess import build_dataset
from v1_basic_pipeline.evaluate import evaluate_all, save_reports
from v1_basic_pipeline.train import build_pipelines, fit_pipelines, select_best, split_dataset

PIPELINE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PIPELINE_DIR / "models"
REPORTS_DIR = PIPELINE_DIR / "reports"


def main() -> None:
    csv_path = download_data(RAW_DATA_DIR)
    df = load_dataset(csv_path)
    X, y = build_dataset(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    pipelines = build_pipelines()
    fitted = fit_pipelines(pipelines, X_train, y_train)
    best_name = select_best(fitted, X_test, y_test)

    all_metrics = evaluate_all(fitted, X_test, y_test)
    save_reports(all_metrics, best_name, REPORTS_DIR)

    save_artifact(fitted[best_name], MODELS_DIR / "best_model.pkl")

    print(f"Best model: {best_name}")
    print(f"F1: {all_metrics[best_name]['f1']:.4f}")
    print(f"Accuracy: {all_metrics[best_name]['accuracy']:.4f}")


if __name__ == "__main__":
    main()
