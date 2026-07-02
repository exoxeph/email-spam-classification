"""Stratified train/test split, TF-IDF fit on train only, trains Naive Bayes and Logistic Regression, saves the best as one bundled sklearn Pipeline."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from common.config import RANDOM_SEED, TEST_SIZE


def split_dataset(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)


def build_pipelines() -> dict[str, Pipeline]:
    return {
        "naive_bayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("clf", MultinomialNB()),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                (
                    "clf",
                    LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED, max_iter=1000),
                ),
            ]
        ),
    }


def fit_pipelines(pipelines: dict[str, Pipeline], X_train, y_train) -> dict[str, Pipeline]:
    for pipeline in pipelines.values():
        pipeline.fit(X_train, y_train)
    return pipelines


def select_best(fitted_pipelines: dict[str, Pipeline], X_test, y_test) -> str:
    scores = {name: f1_score(y_test, pipeline.predict(X_test)) for name, pipeline in fitted_pipelines.items()}
    return max(scores, key=scores.get)
