"""Append user feedback for v4 predictions to data/feedback/feedback.csv."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from common.config import FEEDBACK_DATA_DIR

FEEDBACK_COLUMNS = [
    "timestamp_utc",
    "email_text",
    "prediction_label",
    "spam_probability",
    "risk_level",
    "recommended_action",
    "user_feedback",
]


def log_feedback(
    email_text: str,
    prediction: dict,
    user_feedback: str,
    feedback_path: Path = FEEDBACK_DATA_DIR / "feedback.csv",
) -> Path:
    feedback_path = Path(feedback_path)
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "email_text": email_text,
                "prediction_label": prediction["prediction_label"],
                "spam_probability": prediction["spam_probability"],
                "risk_level": prediction["risk_level"],
                "recommended_action": prediction["recommended_action"],
                "user_feedback": user_feedback,
            }
        ],
        columns=FEEDBACK_COLUMNS,
    )
    row.to_csv(feedback_path, mode="a", header=not feedback_path.exists(), index=False)
    return feedback_path
