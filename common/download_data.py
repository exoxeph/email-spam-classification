"""Idempotent Kaggle dataset download into data/raw/, via kagglehub."""

import shutil
from pathlib import Path

import kagglehub
from dotenv import load_dotenv

from common.config import RAW_DATA_DIR

DATASET_SLUG = "bayes2003/emails-for-spam-or-ham-classification-enron-2006"
# This dataset's download folder contains two CSVs (email_origin.csv,
# email_text.csv) - we need the cleaned text+label one specifically, not
# just "the first CSV found" (which would pick the wrong file
# alphabetically).
DATASET_CSV_FILENAME = "email_text.csv"


def download_data(dest_dir: Path = RAW_DATA_DIR) -> Path:
    dest_dir = Path(dest_dir)
    existing = list(dest_dir.glob("*.csv"))
    if existing:
        return existing[0]

    load_dotenv()

    download_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    dest_dir.mkdir(parents=True, exist_ok=True)

    source_csv = download_path / DATASET_CSV_FILENAME
    if not source_csv.exists():
        raise FileNotFoundError(f"Expected {DATASET_CSV_FILENAME} not found in downloaded dataset at {download_path}")

    dest_path = dest_dir / source_csv.name
    shutil.copy(source_csv, dest_path)
    return dest_path
