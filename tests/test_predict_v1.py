import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from v1_basic_pipeline.predict import predict


def _toy_pipeline():
    X_train = pd.Series(
        [
            "free money win prize now click",
            "urgent claim your cash reward",
            "meeting scheduled for tomorrow afternoon",
            "please see the attached document",
        ]
        * 10
    )
    y_train = pd.Series([1, 1, 0, 0] * 10)
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipeline.fit(X_train, y_train)
    return pipeline


def test_predict_returns_spam_for_spammy_text():
    pipeline = _toy_pipeline()
    result = predict("FREE money win a prize, click now!!!", pipeline)
    assert result == "spam"


def test_predict_returns_not_spam_for_normal_text():
    pipeline = _toy_pipeline()
    result = predict("please see the attached document for tomorrow's meeting", pipeline)
    assert result == "not spam"
