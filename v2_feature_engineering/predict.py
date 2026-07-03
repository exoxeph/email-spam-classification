"""Load v2's bundled model and predict spam/not-spam for new email content."""

from pathlib import Path

import pandas as pd

from common.features import build_feature_frame
from common.io_utils import load_artifact


def load_model(model_path: Path):
    return load_artifact(model_path)


def predict(text: str, pipeline, subject: str = "") -> str:
    frame = pd.DataFrame({"subject": [subject], "body": [text]})
    features = build_feature_frame(frame)
    prediction = pipeline.predict(features)[0]
    return "spam" if prediction == 1 else "not spam"
