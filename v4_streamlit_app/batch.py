"""Batch CSV prediction helpers for v4."""

from __future__ import annotations

import pandas as pd

from v3_model_comparison_tuning.predict import predict_email_risk

REQUIRED_COLUMNS = {"email_id", "email_text"}
OUTPUT_COLUMNS = [
    "email_id",
    "email_text",
    "prediction_label",
    "spam_probability",
    "risk_level",
    "recommended_action",
]


def predict_batch(df: pd.DataFrame, predictor=predict_email_risk) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for _, row in df.iterrows():
        subject = row.get("subject", "")
        result = predictor(str(row["email_text"]), subject=str(subject))
        rows.append(
            {
                "email_id": row["email_id"],
                "email_text": row["email_text"],
                "prediction_label": result["prediction_label"],
                "spam_probability": result["spam_probability"],
                "risk_level": result["risk_level"],
                "recommended_action": result["recommended_action"],
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
