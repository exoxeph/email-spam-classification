import pandas as pd

from v3_model_comparison_tuning.error_analysis import build_prediction_frame, false_negatives, false_positives


def _prediction_frame():
    return pd.DataFrame(
        {
            "email_text": ["ham flagged", "spam missed", "ham ok", "spam caught"],
            "actual_label": [0, 1, 0, 1],
            "predicted_label": [1, 0, 0, 1],
            "spam_probability": [0.8, 0.2, 0.1, 0.9],
        }
    )


def test_false_positive_finder_returns_only_actual_ham_predicted_spam():
    result = false_positives(_prediction_frame())

    assert len(result) == 1
    assert result.loc[0, "email_text"] == "ham flagged"
    assert result.loc[0, "actual_label"] == 0
    assert result.loc[0, "predicted_label"] == 1


def test_false_negative_finder_returns_only_actual_spam_predicted_ham():
    result = false_negatives(_prediction_frame())

    assert len(result) == 1
    assert result.loc[0, "email_text"] == "spam missed"
    assert result.loc[0, "actual_label"] == 1
    assert result.loc[0, "predicted_label"] == 0


def test_false_positive_finder_returns_empty_frame_when_none_exist():
    frame = pd.DataFrame(
        {
            "email_text": ["ham ok", "spam caught"],
            "actual_label": [0, 1],
            "predicted_label": [0, 1],
            "spam_probability": [0.1, 0.9],
        }
    )

    assert false_positives(frame).empty


def test_build_prediction_frame_uses_threshold_and_preserves_text():
    source = pd.DataFrame({"subject": ["Hello", "Win"], "body": ["meeting", "cash"]})

    frame = build_prediction_frame(source, [0, 1], [0.4, 0.6], threshold=0.5)

    assert frame["email_text"].tolist() == ["Hello meeting", "Win cash"]
    assert frame["predicted_label"].tolist() == [0, 1]
