"""Dataset loading (subject/body/label columns) and artifact save/load helpers, shared by all versions."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_dataset(csv_path: Path) -> pd.DataFrame:
    # Source CSV (bayes2003/emails-for-spam-or-ham-classification-enron-2006,
    # file email_text.csv) has only "text" and "label" (int 0/1) columns - no
    # separate subject field. "subject" is kept as an always-empty column so
    # the (subject, body, label) interface stays consistent for later
    # versions, even though this particular source has nothing to put in it.
    df = pd.read_csv(csv_path)
    label_map = {1: "spam", 0: "ham"}
    return pd.DataFrame(
        {
            "subject": "",
            "body": df["text"],
            "label": df["label"].map(label_map),
        }
    )


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
