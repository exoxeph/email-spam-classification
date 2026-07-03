"""Train v3 candidate models with a 70/15/15 stratified split."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common.config import RANDOM_SEED, V3_TEST_SIZE, V3_VAL_SIZE
from common.features import build_column_transformer


def split_dataset(X, y):
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X,
        y,
        test_size=V3_TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_remaining,
        y_remaining,
        test_size=V3_VAL_SIZE,
        stratify=y_remaining,
        random_state=RANDOM_SEED,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


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
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=5000),
                        cv=3,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("features", build_column_transformer()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                        max_features="sqrt",
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
    }


def fit_pipelines(pipelines: dict[str, Pipeline], X_train, y_train) -> dict[str, Pipeline]:
    for pipeline in pipelines.values():
        pipeline.fit(X_train, y_train)
    return pipelines


def select_best(fitted_pipelines: dict[str, Pipeline], X_val, y_val) -> str:
    scores = {name: f1_score(y_val, pipeline.predict(X_val), zero_division=0) for name, pipeline in fitted_pipelines.items()}
    return max(scores, key=scores.get)
