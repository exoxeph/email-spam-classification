"""Load v3's bundled model and map spam probability to low/medium/high risk."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from common.config import REPO_ROOT
from common.features import build_feature_frame
from common.io_utils import load_artifact

LOW_RISK_THRESHOLD = 0.40
HIGH_RISK_THRESHOLD = 0.80
CLASSIFICATION_THRESHOLD = 0.60
DEFAULT_MODEL_PATH = REPO_ROOT / "v3_model_comparison_tuning" / "models" / "best_model_v3.pkl"


def load_model(model_path: Path):
    return load_artifact(model_path)


@lru_cache(maxsize=1)
def load_default_model():
    return load_model(DEFAULT_MODEL_PATH)


def predict_risk(text: str, pipeline, subject: str = "") -> dict:
    frame = pd.DataFrame({"subject": [subject], "body": [text]})
    features = build_feature_frame(frame)
    probability = float(pipeline.predict_proba(features)[0, 1])
    return {
        "spam_probability": probability,
        "predicted_label": "spam" if probability >= CLASSIFICATION_THRESHOLD else "not spam",
        "risk_level": risk_level(probability),
    }


def predict_email_risk(text: str, subject: str = "", pipeline=None) -> dict:
    model = pipeline if pipeline is not None else load_default_model()
    result = predict_risk(text, model, subject=subject)
    risk = result["risk_level"]
    return {
        "prediction_label": prediction_label(risk),
        "spam_probability": result["spam_probability"],
        "risk_level": risk.title(),
        "recommended_action": recommended_action(risk),
    }


def risk_level(spam_probability: float) -> str:
    if spam_probability < LOW_RISK_THRESHOLD:
        return "low"
    if spam_probability < HIGH_RISK_THRESHOLD:
        return "medium"
    return "high"


def prediction_label(risk: str) -> str:
    return {"low": "Not Spam", "medium": "Needs Review", "high": "Spam"}[risk]


def recommended_action(risk: str) -> str:
    return {"low": "Allow", "medium": "Warn User", "high": "Quarantine"}[risk]
