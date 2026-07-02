"""Dataset loading (subject/body/label columns) and artifact save/load helpers, shared by all versions."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Subject": "subject", "Message": "body", "Spam/Ham": "label"})
    return df[["subject", "body", "label"]]


def save_artifact(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: Path) -> Any:
    # joblib.load uses pickle under the hood, which can execute arbitrary code
    # for untrusted input. This is safe here: every caller in this repo only
    # loads model files that this same codebase trained and wrote to its own
    # models/ directory (e.g. v1_basic_pipeline/models/best_model.pkl) -
    # never a file from an external or user-supplied source.
    return joblib.load(path)
