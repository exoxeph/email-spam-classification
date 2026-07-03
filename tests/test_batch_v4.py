import pandas as pd
import pytest

from v4_streamlit_app.batch import OUTPUT_COLUMNS, predict_batch


def fake_predictor(text: str, subject: str = "") -> dict:
    return {
        "prediction_label": "Spam" if "win" in text.lower() else "Not Spam",
        "spam_probability": 0.9 if "win" in text.lower() else 0.1,
        "risk_level": "High" if "win" in text.lower() else "Low",
        "recommended_action": "Quarantine" if "win" in text.lower() else "Allow",
    }


def test_predict_batch_returns_one_output_row_per_input():
    df = pd.DataFrame({"email_id": ["a", "b"], "email_text": ["hello team", "win cash"]})

    result = predict_batch(df, predictor=fake_predictor)

    assert len(result) == 2
    assert list(result.columns) == OUTPUT_COLUMNS


def test_predict_batch_preserves_email_id():
    df = pd.DataFrame({"email_id": [101, 102], "email_text": ["hello", "win"]})

    result = predict_batch(df, predictor=fake_predictor)

    assert result["email_id"].tolist() == [101, 102]


def test_predict_batch_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        predict_batch(pd.DataFrame({"email_text": ["hello"]}), predictor=fake_predictor)
