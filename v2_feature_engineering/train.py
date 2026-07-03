"""Train v2 Logistic Regression and Linear SVM pipelines on shared engineered features."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common.config import RANDOM_SEED, TEST_SIZE
from common.features import build_column_transformer


def split_dataset(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)


def build_pipelines() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("features", build_column_transformer()),
                (
                    "clf",
                    LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED, max_iter=1000),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("features", build_column_transformer()),
                ("clf", LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=5000)),
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
