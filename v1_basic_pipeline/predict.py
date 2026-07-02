"""Loads models/best_model.pkl and predicts spam/not-spam for new email text, reusing common.preprocess.clean_text."""

from pathlib import Path

from common.io_utils import load_artifact
from common.preprocess import clean_text


def load_model(model_path: Path):
    return load_artifact(model_path)


def predict(text: str, pipeline) -> str:
    cleaned = clean_text(text)
    prediction = pipeline.predict([cleaned])[0]
    return "spam" if prediction == 1 else "not spam"
