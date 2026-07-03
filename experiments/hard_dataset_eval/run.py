"""Run the v1 TF-IDF baselines on a harder phishing email dataset."""

import shutil
import sys
from pathlib import Path

import kagglehub
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.preprocess import build_dataset
from v1_basic_pipeline.evaluate import evaluate_all, save_reports
from v1_basic_pipeline.train import build_pipelines, fit_pipelines, select_best, split_dataset

DATASET_SLUG = "naserabdullahalam/phishing-email-dataset"
DATASET_CSV_FILENAME = "phishing_email.csv"

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = EXPERIMENT_DIR / "reports"
RAW_DATA_DIR = REPO_ROOT / "data" / "raw" / "hard_dataset"


def download_data(dest_dir: Path = RAW_DATA_DIR) -> Path:
    dest_dir = Path(dest_dir)
    dest_path = dest_dir / DATASET_CSV_FILENAME
    if dest_path.exists():
        return dest_path

    load_dotenv()

    download_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    dest_dir.mkdir(parents=True, exist_ok=True)

    source_csv = download_path / DATASET_CSV_FILENAME
    if not source_csv.exists():
        raise FileNotFoundError(f"Expected {DATASET_CSV_FILENAME} not found in downloaded dataset at {download_path}")

    shutil.copy(source_csv, dest_path)
    return dest_path


def main() -> None:
    csv_path = download_data()
    df = pd.read_csv(csv_path)

    prepared = pd.DataFrame(
        {
            "subject": "",
            "body": df["text_combined"],
            "label": df["label"].map({1: "spam", 0: "ham"}),
        }
    )
    X, y = build_dataset(prepared)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    pipelines = build_pipelines()
    fitted = fit_pipelines(pipelines, X_train, y_train)
    best_name = select_best(fitted, X_test, y_test)

    all_metrics = evaluate_all(fitted, X_test, y_test)
    save_reports(all_metrics, best_name, REPORTS_DIR)

    print(f"Best model: {best_name}")
    print(f"F1: {all_metrics[best_name]['f1']:.4f}")
    print(f"Accuracy: {all_metrics[best_name]['accuracy']:.4f}")


if __name__ == "__main__":
    main()
