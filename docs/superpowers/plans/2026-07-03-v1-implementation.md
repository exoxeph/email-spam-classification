# v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the actual v1 pipeline logic (data loading, cleaning, TF-IDF, Naive Bayes / Logistic Regression training, evaluation, prediction) on top of the existing scaffolded stub files, so `poetry run python -m v1_basic_pipeline.main` runs end-to-end against the real Kaggle dataset and produces a trained model + metrics reports.

**Architecture:** Pure functions in `common/` (cleaning, I/O, download) feed a straight-line pipeline in `v1_basic_pipeline/`: split -> fit TF-IDF+model pairs -> pick winner by F1 -> evaluate all -> save reports + one bundled `sklearn.Pipeline` artifact. No classes, no framework — just functions with explicit inputs/outputs, matching the existing stub file layout.

**Tech Stack:** Python 3.11, scikit-learn, pandas, joblib, kagglehub, python-dotenv, pytest. All already installed (from repo scaffolding).

## Global Constraints

- Dataset: Kaggle slug `marcelwiechmann/enron-spam-data`. Raw CSV columns: `Subject`, `Message`, `Spam/Ham`, `Date`. Label values are the strings `"spam"` / `"ham"` (case-insensitive-safe).
- `RANDOM_SEED = 42`, `TEST_SIZE = 0.2` — already defined in `common/config.py`, must be imported, never hardcoded again.
- TF-IDF vectorizer must be fit ONLY on training data (never on the full dataset or test data) — this is the leakage fix from the v1 spec.
- Split must use `stratify=y` and `random_state=RANDOM_SEED`.
- Logistic Regression must use `class_weight="balanced"`.
- The final saved artifact is ONE `sklearn.Pipeline` (vectorizer + classifier bundled together) at `v1_basic_pipeline/models/best_model.pkl` — never two separate files.
- Reports go to `v1_basic_pipeline/reports/metrics.json` and `v1_basic_pipeline/reports/model_comparison.csv`.
- `common/preprocess.py` functions (`clean_text`, `map_label`, `build_dataset`) must be usable by v2/v3/v4 later without modification — keep them pure (no file I/O, no globals beyond what's passed in).
- Stick to plain ASCII in all file content (this repo's sandbox has a recurring mojibake bug with non-ASCII characters like em-dashes).

---

### Task 1: `common/preprocess.py` — text cleaning and label mapping

**Files:**
- Modify: `common/preprocess.py` (currently a one-line docstring stub)
- Test: `tests/test_preprocess_v1.py` (new)

**Interfaces:**
- Produces: `clean_text(text: str) -> str`, `map_label(label: str) -> int`, `build_dataset(df: pandas.DataFrame) -> tuple[pandas.Series, pandas.Series]` — all imported by Task 4 (train.py), Task 6 (predict.py), and Task 7 (main.py). `build_dataset` expects a DataFrame with columns `subject`, `body`, `label` (raw strings) and returns `(X, y)` where `X` is a `pandas.Series` of cleaned combined text and `y` is a `pandas.Series` of int labels (0/1), same length, index-aligned, deduplicated.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_preprocess_v1.py`:

```python
import pandas as pd

from common.preprocess import build_dataset, clean_text, map_label


def test_clean_text_lowercases():
    assert clean_text("HELLO World") == "hello world"


def test_clean_text_strips_extra_whitespace():
    assert clean_text("hello    world\n\nfoo") == "hello world foo"
    assert clean_text("  padded text  ") == "padded text"


def test_map_label_spam_to_1_and_ham_to_0():
    assert map_label("spam") == 1
    assert map_label("ham") == 0
    assert map_label("SPAM") == 1
    assert map_label("Ham") == 0


def test_map_label_rejects_unknown_value():
    import pytest

    with pytest.raises(ValueError):
        map_label("unknown")


def test_build_dataset_drops_empty_rows():
    df = pd.DataFrame(
        {
            "subject": ["Meeting tomorrow", "", None],
            "body": ["See you at 2pm", "", "   "],
            "label": ["ham", "spam", "spam"],
        }
    )
    X, y = build_dataset(df)
    assert len(X) == 1
    assert X.iloc[0] == "meeting tomorrow see you at 2pm"
    assert y.iloc[0] == 0


def test_build_dataset_drops_duplicates():
    df = pd.DataFrame(
        {
            "subject": ["Win now", "Win now"],
            "body": ["Click here", "Click here"],
            "label": ["spam", "spam"],
        }
    )
    X, y = build_dataset(df)
    assert len(X) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_preprocess_v1.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `clean_text`/`map_label`/`build_dataset` (they don't exist yet — `common/preprocess.py` is still just a docstring).

- [ ] **Step 3: Implement `common/preprocess.py`**

```python
"""clean_text() and map_label() - basic text cleaning and label mapping shared by all versions."""

import re

import pandas as pd


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_label(label: str) -> int:
    normalized = label.strip().lower()
    if normalized == "spam":
        return 1
    if normalized == "ham":
        return 0
    raise ValueError(f"Unexpected label value: {label!r}")


def build_dataset(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    df = df.copy()
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df = df[df["body"].str.strip() != ""]

    combined = (df["subject"] + " " + df["body"]).apply(clean_text)
    labels = df["label"].apply(map_label)

    combined = combined.drop_duplicates()
    labels = labels.loc[combined.index]

    return combined.reset_index(drop=True), labels.reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_preprocess_v1.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add common/preprocess.py tests/test_preprocess_v1.py
git commit -m "feat: implement common/preprocess.py cleaning and label mapping"
```

---

### Task 2: `common/io_utils.py` — dataset loading and artifact I/O

**Files:**
- Modify: `common/io_utils.py` (currently a one-line docstring stub)
- Test: `tests/test_io_utils_v1.py` (new)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `load_dataset(csv_path: pathlib.Path) -> pandas.DataFrame` (columns `subject`, `body`, `label`), `save_artifact(obj, path: pathlib.Path) -> None`, `load_artifact(path: pathlib.Path) -> Any`. Used by Task 3 (download_data doesn't need these), Task 6 (predict.py uses `load_artifact`), Task 7 (main.py uses `load_dataset` and `save_artifact`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_io_utils_v1.py`:

```python
import pandas as pd

from common.io_utils import load_artifact, load_dataset, save_artifact


def test_load_dataset_renames_columns(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "Subject,Message,Spam/Ham,Date\n"
        "Win now,Click here,spam,2006-01-01\n"
        "Meeting,See you at 2pm,ham,2006-01-02\n"
    )

    df = load_dataset(csv_path)

    assert list(df.columns) == ["subject", "body", "label"]
    assert df.iloc[0]["subject"] == "Win now"
    assert df.iloc[0]["body"] == "Click here"
    assert df.iloc[0]["label"] == "spam"


def test_save_and_load_artifact_roundtrip(tmp_path):
    artifact_path = tmp_path / "nested" / "artifact.pkl"
    obj = {"hello": "world", "numbers": [1, 2, 3]}

    save_artifact(obj, artifact_path)
    loaded = load_artifact(artifact_path)

    assert artifact_path.exists()
    assert loaded == obj
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_io_utils_v1.py -v`
Expected: FAIL with `ImportError` for `load_dataset`/`save_artifact`/`load_artifact`.

- [ ] **Step 3: Implement `common/io_utils.py`**

```python
"""Dataset loading (subject/body/label columns) and artifact save/load helpers, shared by all versions."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Subject": "subject", "Message": "body", "Spam/Ham": "label"})
    return df[["subject", "body", "label"]]


def save_artifact(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: Path) -> Any:
    # joblib.load uses pickle under the hood, which can execute arbitrary code
    # for untrusted input. This is safe here: every caller in this repo only
    # loads model files that this same codebase trained and wrote to its own
    # models/ directory (e.g. v1_basic_pipeline/models/best_model.pkl) -
    # never a file from an external or user-supplied source.
    return joblib.load(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_io_utils_v1.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add common/io_utils.py tests/test_io_utils_v1.py
git commit -m "feat: implement common/io_utils.py dataset loading and artifact I/O"
```

---

### Task 3: `common/download_data.py` — Kaggle dataset download

**Files:**
- Modify: `common/download_data.py` (currently a one-line docstring stub)

**Interfaces:**
- Consumes: `common.config.RAW_DATA_DIR` (already exists).
- Produces: `download_data(dest_dir: pathlib.Path = RAW_DATA_DIR) -> pathlib.Path` — returns the path to the CSV file in `data/raw/`. Used by Task 7 (main.py).

No unit test for this task: it requires live network access and real Kaggle credentials, which can't be exercised in an isolated/offline test. Verified manually in Step 3 below instead.

- [ ] **Step 1: Implement `common/download_data.py`**

```python
"""Idempotent Kaggle dataset download into data/raw/, via kagglehub."""

import shutil
from pathlib import Path

import kagglehub
from dotenv import load_dotenv

from common.config import RAW_DATA_DIR

DATASET_SLUG = "marcelwiechmann/enron-spam-data"


def download_data(dest_dir: Path = RAW_DATA_DIR) -> Path:
    dest_dir = Path(dest_dir)
    existing = list(dest_dir.glob("*.csv"))
    if existing:
        return existing[0]

    load_dotenv()

    download_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    dest_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(download_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in downloaded dataset at {download_path}")

    dest_path = dest_dir / csv_files[0].name
    shutil.copy(csv_files[0], dest_path)
    return dest_path
```

- [ ] **Step 2: Verify idempotency logic with a quick manual check (no network needed for this part)**

Run: `poetry run python -c "
from pathlib import Path
import tempfile
from common.download_data import download_data

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / 'existing.csv').write_text('a,b\n1,2\n')
    result = download_data(d)
    assert result == d / 'existing.csv', result
    print('idempotency check OK')
"`
Expected: `idempotency check OK` (this exercises the early-return path without touching the network).

- [ ] **Step 3: Verify the real download (requires the .env Kaggle credentials already set up in this repo, and network access)**

Run: `poetry run python -c "
from common.download_data import download_data
path = download_data()
print(path)
print(path.exists())
"`
Expected: prints a path under `data/raw/` ending in `.csv`, then `True`. This actually contacts Kaggle and downloads the real dataset (a few MB) — only run this once per machine, since the idempotency check means later runs just return the cached path.

- [ ] **Step 4: Commit**

```bash
git add common/download_data.py
git commit -m "feat: implement common/download_data.py Kaggle dataset download"
```

---

### Task 4: `v1_basic_pipeline/train.py` — split, TF-IDF pipelines, training, model selection

**Files:**
- Modify: `v1_basic_pipeline/train.py` (currently a one-line docstring stub)
- Test: `tests/test_train_v1.py` (new)

**Interfaces:**
- Consumes: `common.config.RANDOM_SEED`, `common.config.TEST_SIZE`.
- Produces: `split_dataset(X, y) -> tuple` (X_train, X_test, y_train, y_test), `build_pipelines() -> dict[str, sklearn.pipeline.Pipeline]` (keys `"naive_bayes"`, `"logistic_regression"`), `fit_pipelines(pipelines: dict, X_train, y_train) -> dict[str, Pipeline]` (fits in place, returns same dict), `select_best(fitted_pipelines: dict, X_test, y_test) -> str` (name of best pipeline by F1). All consumed by Task 7 (main.py) and Task 5 (evaluate.py takes the fitted dict this task produces).

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_v1.py`:

```python
import pandas as pd

from v1_basic_pipeline.train import split_dataset


def test_split_dataset_preserves_class_ratio():
    # 90 ham (0), 10 spam (1) - a 10% minority class, like real spam datasets.
    y = pd.Series([0] * 90 + [1] * 10)
    X = pd.Series([f"email {i}" for i in range(100)])

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    overall_ratio = y.mean()
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    assert abs(train_ratio - overall_ratio) < 0.05
    assert abs(test_ratio - overall_ratio) < 0.05
    assert len(X_test) == len(y_test) == 20  # TEST_SIZE=0.2 of 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_train_v1.py -v`
Expected: FAIL with `ImportError` for `split_dataset` (doesn't exist yet).

- [ ] **Step 3: Implement `v1_basic_pipeline/train.py`**

```python
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
                ("clf", LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED, max_iter=1000)),
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_train_v1.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Manually verify the training functions work end-to-end on synthetic data**

Run: `poetry run python -c "
import pandas as pd
from v1_basic_pipeline.train import build_pipelines, fit_pipelines, select_best, split_dataset

X = pd.Series(['free money now click', 'win a prize today', 'meeting at 2pm tomorrow', 'see you at the office', 'urgent claim your cash', 'lunch plans for friday'] * 10)
y = pd.Series([1, 1, 0, 0, 1, 0] * 10)

X_train, X_test, y_train, y_test = split_dataset(X, y)
pipelines = build_pipelines()
fitted = fit_pipelines(pipelines, X_train, y_train)
best = select_best(fitted, X_test, y_test)
print('best model:', best)
assert best in ('naive_bayes', 'logistic_regression')
print('OK')
"`
Expected: prints `best model: <name>` then `OK`.

- [ ] **Step 6: Commit**

```bash
git add v1_basic_pipeline/train.py tests/test_train_v1.py
git commit -m "feat: implement v1_basic_pipeline/train.py split, pipelines, and model selection"
```

---

### Task 5: `v1_basic_pipeline/evaluate.py` — metrics computation and report saving

**Files:**
- Modify: `v1_basic_pipeline/evaluate.py` (currently a one-line docstring stub)
- Test: `tests/test_evaluate_v1.py` (new)

**Interfaces:**
- Consumes: fitted pipelines dict from Task 4's `fit_pipelines`/`select_best` (same shape: `dict[str, sklearn.pipeline.Pipeline]`).
- Produces: `compute_metrics(pipeline, X_test, y_test) -> dict` (keys `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`), `evaluate_all(fitted_pipelines: dict, X_test, y_test) -> dict[str, dict]`, `save_reports(all_metrics: dict, best_name: str, reports_dir: pathlib.Path) -> None`. Consumed by Task 7 (main.py).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluate_v1.py`:

```python
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from v1_basic_pipeline.evaluate import compute_metrics, evaluate_all, save_reports


def _toy_fitted_pipeline():
    X_train = pd.Series(["free money now", "win a prize", "meeting at 2pm", "see you tomorrow"] * 5)
    y_train = pd.Series([1, 1, 0, 0] * 5)
    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipeline.fit(X_train, y_train)
    return pipeline


def test_compute_metrics_returns_expected_keys():
    pipeline = _toy_fitted_pipeline()
    X_test = pd.Series(["free prize now", "see you at the office"])
    y_test = pd.Series([1, 0])

    metrics = compute_metrics(pipeline, X_test, y_test)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "confusion_matrix"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["confusion_matrix"][0]) == 2


def test_evaluate_all_covers_every_model():
    pipeline = _toy_fitted_pipeline()
    fitted = {"model_a": pipeline, "model_b": pipeline}
    X_test = pd.Series(["free prize now", "see you at the office"])
    y_test = pd.Series([1, 0])

    all_metrics = evaluate_all(fitted, X_test, y_test)

    assert set(all_metrics.keys()) == {"model_a", "model_b"}


def test_save_reports_writes_metrics_json_and_comparison_csv(tmp_path):
    all_metrics = {
        "model_a": {"accuracy": 0.9, "precision": 0.8, "recall": 0.85, "f1": 0.82, "confusion_matrix": [[8, 1], [1, 10]]},
        "model_b": {"accuracy": 0.95, "precision": 0.9, "recall": 0.92, "f1": 0.91, "confusion_matrix": [[9, 0], [1, 10]]},
    }

    save_reports(all_metrics, best_name="model_b", reports_dir=tmp_path)

    metrics_json = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics_json["best_model"] == "model_b"
    assert metrics_json["f1"] == 0.91

    comparison_csv = (tmp_path / "model_comparison.csv").read_text()
    assert "model_a" in comparison_csv
    assert "model_b" in comparison_csv
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_evaluate_v1.py -v`
Expected: FAIL with `ImportError` for `compute_metrics`/`evaluate_all`/`save_reports`.

- [ ] **Step 3: Implement `v1_basic_pipeline/evaluate.py`**

```python
"""Computes accuracy, precision, recall, F1, and confusion matrix for both v1 candidate models; writes reports/metrics.json and model_comparison.csv."""

import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(pipeline, X_test, y_test) -> dict:
    predictions = pipeline.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def evaluate_all(fitted_pipelines: dict, X_test, y_test) -> dict:
    return {name: compute_metrics(pipeline, X_test, y_test) for name, pipeline in fitted_pipelines.items()}


def save_reports(all_metrics: dict, best_name: str, reports_dir: Path) -> None:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {"best_model": best_name, **all_metrics[best_name]}
    (reports_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    with open(reports_dir / "model_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "precision", "recall", "f1"])
        for name, metrics in all_metrics.items():
            writer.writerow([name, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_evaluate_v1.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add v1_basic_pipeline/evaluate.py tests/test_evaluate_v1.py
git commit -m "feat: implement v1_basic_pipeline/evaluate.py metrics and report saving"
```

---

### Task 6: `v1_basic_pipeline/predict.py` — load model and predict on new text

**Files:**
- Modify: `v1_basic_pipeline/predict.py` (currently a one-line docstring stub)
- Test: `tests/test_predict_v1.py` (new)

**Interfaces:**
- Consumes: `common.io_utils.load_artifact` (Task 2), `common.preprocess.clean_text` (Task 1).
- Produces: `load_model(model_path: pathlib.Path) -> sklearn.pipeline.Pipeline`, `predict(text: str, pipeline) -> str` (returns `"spam"` or `"not spam"`). Used interactively and by Task 8's manual verification.

- [ ] **Step 1: Write the failing test**

Create `tests/test_predict_v1.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_predict_v1.py -v`
Expected: FAIL with `ImportError` for `predict` (doesn't exist yet).

- [ ] **Step 3: Implement `v1_basic_pipeline/predict.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_predict_v1.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add v1_basic_pipeline/predict.py tests/test_predict_v1.py
git commit -m "feat: implement v1_basic_pipeline/predict.py model loading and prediction"
```

---

### Task 7: `v1_basic_pipeline/main.py` — wire the full pipeline together

**Files:**
- Modify: `v1_basic_pipeline/main.py` (currently a one-line docstring stub)

**Interfaces:**
- Consumes: `common.download_data.download_data` (Task 3), `common.io_utils.load_dataset` + `save_artifact` (Task 2), `common.preprocess.build_dataset` (Task 1), `v1_basic_pipeline.train.split_dataset` + `build_pipelines` + `fit_pipelines` + `select_best` (Task 4), `v1_basic_pipeline.evaluate.evaluate_all` + `save_reports` (Task 5).
- Produces: `main() -> None`, callable via `python -m v1_basic_pipeline.main`.

No new unit test for this task — it's pure wiring of already-tested functions. Verified via the real end-to-end run in Task 8.

- [ ] **Step 1: Implement `v1_basic_pipeline/main.py`**

```python
"""Entry point: download data if missing, then run load -> preprocess -> train -> evaluate."""

from pathlib import Path

from common.config import RAW_DATA_DIR
from common.download_data import download_data
from common.io_utils import load_dataset, save_artifact
from common.preprocess import build_dataset
from v1_basic_pipeline.evaluate import evaluate_all, save_reports
from v1_basic_pipeline.train import build_pipelines, fit_pipelines, select_best, split_dataset

PIPELINE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PIPELINE_DIR / "models"
REPORTS_DIR = PIPELINE_DIR / "reports"


def main() -> None:
    csv_path = download_data(RAW_DATA_DIR)
    df = load_dataset(csv_path)
    X, y = build_dataset(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    pipelines = build_pipelines()
    fitted = fit_pipelines(pipelines, X_train, y_train)
    best_name = select_best(fitted, X_test, y_test)

    all_metrics = evaluate_all(fitted, X_test, y_test)
    save_reports(all_metrics, best_name, REPORTS_DIR)

    save_artifact(fitted[best_name], MODELS_DIR / "best_model.pkl")

    print(f"Best model: {best_name}")
    print(f"F1: {all_metrics[best_name]['f1']:.4f}")
    print(f"Accuracy: {all_metrics[best_name]['accuracy']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add v1_basic_pipeline/main.py
git commit -m "feat: wire v1_basic_pipeline/main.py end-to-end pipeline"
```

---

### Task 8: End-to-end verification against the real dataset

**Files:** none created — this task runs the assembled pipeline for real and checks its output.

- [ ] **Step 1: Run the full pipeline**

Run: `poetry run python -m v1_basic_pipeline.main`
Expected: prints `Best model: <naive_bayes|logistic_regression>`, `F1: 0.XXXX`, `Accuracy: 0.XXXX` (real numbers from the real dataset — don't expect a specific value, but F1 and accuracy should both be well above 0.5 given this is a genuine spam/ham dataset with clear signal). Takes a minute or two the first time (downloads the dataset).

- [ ] **Step 2: Verify the artifacts exist**

Run: `poetry run python -c "
from pathlib import Path
assert Path('v1_basic_pipeline/models/best_model.pkl').exists()
assert Path('v1_basic_pipeline/reports/metrics.json').exists()
assert Path('v1_basic_pipeline/reports/model_comparison.csv').exists()
print('all artifacts present')
"`
Expected: `all artifacts present`

- [ ] **Step 3: Inspect the actual metrics**

Run: `cat v1_basic_pipeline/reports/metrics.json && echo --- && cat v1_basic_pipeline/reports/model_comparison.csv`
Expected: readable JSON with `best_model`, `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix` keys, and a 2-row (plus header) CSV comparing both models.

- [ ] **Step 4: Manually test prediction on a hand-written email**

Run: `poetry run python -c "
from v1_basic_pipeline.predict import load_model, predict

pipeline = load_model('v1_basic_pipeline/models/best_model.pkl')
print(predict('URGENT: You have won a free prize! Click here to claim your cash reward now!!!', pipeline))
print(predict('Hi, just confirming our meeting tomorrow at 2pm. See you then.', pipeline))
"`
Expected: first line `spam`, second line `not spam` (this is the real, trained model — if it gets these two obvious examples wrong, something is off and should be investigated before moving on, not silently accepted).

- [ ] **Step 5: Run the full test suite once more, end to end**

Run: `poetry run pytest -v`
Expected: all tests across `test_preprocess_v1.py`, `test_io_utils_v1.py`, `test_train_v1.py`, `test_evaluate_v1.py`, `test_predict_v1.py` pass (13 passed).

- [ ] **Step 6: Commit the generated reports (not the model file or data - those stay gitignored)**

```bash
git add v1_basic_pipeline/reports/metrics.json v1_basic_pipeline/reports/model_comparison.csv
git commit -m "docs: add v1 baseline metrics from first real training run"
```
