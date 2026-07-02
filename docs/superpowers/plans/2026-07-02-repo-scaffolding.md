# Repo Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the full repo skeleton (Poetry project, `common/` package, `v1_basic_pipeline/` through `v4_streamlit_app/` folders, `app/`, `data/`, `tests/`) matching the four approved design specs, with empty/stub modules and no business logic yet.

**Architecture:** Single Poetry project (`package-mode = false`), one virtualenv for the whole repo. `common/` holds shared constants and empty stub modules for logic that will be filled in during each version's own implementation plan. Each `vN_*/` folder gets its own empty stub modules, `models/`, `reports/`, and `README.md`. No business logic, no tests with real assertions — this plan produces structure only, verified by successful imports and `poetry check`, not by testing behavior (there is no behavior yet).

**Tech Stack:** Python 3.11, Poetry 2.2, scikit-learn, pandas, numpy, joblib, kagglehub, matplotlib, streamlit, pytest.

## Global Constraints

- Python version floor: `^3.11` (confirmed installed: 3.11.9).
- Poetry `package-mode = false` — this is an app-style repo, not a distributable package.
- No cross-version imports except v4 → v3 (`v3_model_comparison_tuning.predict`), per the v4 spec's explicit exception.
- Do not commit anything — per user's global instruction, commits happen only when explicitly requested. Every task ends with a verification step, not a commit.
- All stub Python files contain only a one-line module docstring describing the file's intended responsibility (copied from the relevant spec) — no `TODO`/`pass`-only bodies with fake signatures, no placeholder functions.
- `data/raw/`, `data/processed/`, `data/feedback/`, all `models/` folders, and all `reports/` folders are gitignored but must exist in the working tree — use `.gitkeep` files.

---

### Task 1: Poetry project + dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`

**Interfaces:**
- Produces: a working Poetry environment (`poetry install` succeeds) that Tasks 2-9 assume is available via `poetry run`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[tool.poetry]
name = "email-spam-classification"
version = "0.1.0"
description = "Progressive versions of an email spam risk triage system (v1-v4)"
authors = ["mannafee <mannafee@gmail.com>"]
package-mode = false

[tool.poetry.dependencies]
python = "^3.11"
scikit-learn = "^1.5"
pandas = "^2.2"
numpy = "^1.26"
joblib = "^1.4"
kagglehub = "^0.3"
matplotlib = "^3.9"
streamlit = "^1.38"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.pyc
.venv/

# Poetry
dist/

# Data & artifacts (per specs — raw/processed data and trained models are not committed)
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
data/feedback/*
!data/feedback/.gitkeep
**/models/*
!**/models/.gitkeep
*.pkl

# Kaggle credentials
kaggle.json

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 3: Verify Poetry can resolve and install**

Run: `poetry install --no-root`
Expected: Poetry creates a `.venv`, resolves all dependencies, writes `poetry.lock`, exits 0. (`--no-root` because `package-mode = false` means there is no project package to install.)

- [ ] **Step 4: Verify**

Run: `poetry check`
Expected: `All set!`

---

### Task 2: `common/` package

**Files:**
- Create: `common/__init__.py`
- Create: `common/config.py`
- Create: `common/io_utils.py`
- Create: `common/preprocess.py`
- Create: `common/features.py`
- Create: `common/download_data.py`

**Interfaces:**
- Produces: `common.config.RANDOM_SEED`, `common.config.TEST_SIZE`, and path constants that v1-v4 tasks (in future plans) will import. Real values now; everything else in this task is stub-only.

- [ ] **Step 1: Create `common/__init__.py`**

```python
"""Shared package: constants, data I/O, cleaning, and feature-extraction logic reused across v1-v4."""
```

- [ ] **Step 2: Create `common/config.py` with real constants (not a stub — these are simple values needed for the split logic across all versions)**

```python
"""Repo-wide constants: reproducibility seed, split ratios, and shared paths."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RANDOM_SEED = 42
TEST_SIZE = 0.2  # used by v1 and v2 (simple train/test split)

# v3 uses a three-way split instead of TEST_SIZE above:
V3_TEST_SIZE = 0.15
V3_VAL_SIZE = 0.15 / 0.85  # yields 70/15/15 of the original whole

DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEEDBACK_DATA_DIR = DATA_DIR / "feedback"
```

- [ ] **Step 3: Create the four stub modules**

`common/io_utils.py`:
```python
"""Dataset loading (subject/body/label columns) and artifact save/load helpers, shared by all versions."""
```

`common/preprocess.py`:
```python
"""clean_text() and map_label() — basic text cleaning and label mapping shared by all versions."""
```

`common/features.py`:
```python
"""Metadata feature extractors (link count, uppercase ratio, etc.) and the ColumnTransformer builder, shared by v2, v3, and v4."""
```

`common/download_data.py`:
```python
"""Idempotent Kaggle dataset download into data/raw/, via kagglehub."""
```

- [ ] **Step 4: Verify the package imports cleanly**

Run: `poetry run python -c "import common; from common import config; print(config.RANDOM_SEED, config.TEST_SIZE)"`
Expected: prints `42 0.2`, exits 0.

---

### Task 3: `data/` directory structure

**Files:**
- Create: `data/README.md`
- Create: `data/raw/.gitkeep`
- Create: `data/processed/.gitkeep`
- Create: `data/feedback/.gitkeep`

- [ ] **Step 1: Create the four files**

`data/README.md`:
```markdown
# Data

- `raw/` — downloaded via `common/download_data.py` (Kaggle Enron-based spam dataset). Gitignored.
- `processed/` — intermediate cleaned data, if any version writes it. Gitignored.
- `feedback/` — user feedback collected by the v4 Streamlit app (`feedback.csv`). Gitignored.

None of these are committed to the repo. Run the download script to populate `raw/` before training any version.
```

`data/raw/.gitkeep`, `data/processed/.gitkeep`, `data/feedback/.gitkeep`: empty files.

- [ ] **Step 2: Verify**

Run: `ls data/raw data/processed data/feedback`
Expected: each lists a `.gitkeep` file.

---

### Task 4: `v1_basic_pipeline/` folder

**Files:**
- Create: `v1_basic_pipeline/__init__.py`
- Create: `v1_basic_pipeline/train.py`
- Create: `v1_basic_pipeline/evaluate.py`
- Create: `v1_basic_pipeline/predict.py`
- Create: `v1_basic_pipeline/main.py`
- Create: `v1_basic_pipeline/models/.gitkeep`
- Create: `v1_basic_pipeline/reports/.gitkeep`
- Create: `v1_basic_pipeline/README.md`

- [ ] **Step 1: Create `__init__.py` and the four stub modules**

`v1_basic_pipeline/__init__.py`:
```python
"""v1: word-level TF-IDF + Naive Bayes / Logistic Regression baseline spam classifier."""
```

`v1_basic_pipeline/train.py`:
```python
"""Stratified train/test split, TF-IDF fit on train only, trains Naive Bayes and Logistic Regression, saves the best as one bundled sklearn Pipeline."""
```

`v1_basic_pipeline/evaluate.py`:
```python
"""Computes accuracy, precision, recall, F1, and confusion matrix for both v1 candidate models; writes reports/metrics.json and model_comparison.csv."""
```

`v1_basic_pipeline/predict.py`:
```python
"""Loads models/best_model.pkl and predicts spam/not-spam for new email text, reusing common.preprocess.clean_text."""
```

`v1_basic_pipeline/main.py`:
```python
"""Entry point: download data if missing, then run load -> preprocess -> train -> evaluate."""
```

- [ ] **Step 2: Create `models/.gitkeep` and `reports/.gitkeep`** (empty files)

- [ ] **Step 3: Create `v1_basic_pipeline/README.md`**

```markdown
# v1: Basic Pipeline

Word-level TF-IDF + Naive Bayes / Logistic Regression baseline.

Design: `docs/superpowers/specs/2026-07-01-email-spam-v1-design.md`

Status: scaffolded, not yet implemented.

Run (once implemented): `poetry run python -m v1_basic_pipeline.main`
```

- [ ] **Step 4: Verify**

Run: `poetry run python -c "import v1_basic_pipeline"`
Expected: exits 0, no output.

---

### Task 5: `v2_feature_engineering/` folder

**Files:**
- Create: `v2_feature_engineering/__init__.py`
- Create: `v2_feature_engineering/train.py`
- Create: `v2_feature_engineering/evaluate.py`
- Create: `v2_feature_engineering/predict.py`
- Create: `v2_feature_engineering/main.py`
- Create: `v2_feature_engineering/models/.gitkeep`
- Create: `v2_feature_engineering/reports/.gitkeep`
- Create: `v2_feature_engineering/README.md`

- [ ] **Step 1: Create `__init__.py` and the four stub modules**

`v2_feature_engineering/__init__.py`:
```python
"""v2: word+char TF-IDF combined with engineered metadata features (links, uppercase ratio, etc.) via ColumnTransformer."""
```

`v2_feature_engineering/train.py`:
```python
"""Builds the combined ColumnTransformer (word_tfidf, char_tfidf, scaled metadata), trains Logistic Regression and Linear SVM, saves the best as one bundled Pipeline."""
```

`v2_feature_engineering/evaluate.py`:
```python
"""Computes accuracy, precision, recall, F1, confusion matrix for both v2 candidate models; writes reports/metrics_v2.json, model_comparison.csv, feature_summary.md."""
```

`v2_feature_engineering/predict.py`:
```python
"""Loads models/best_model_v2.pkl and predicts spam/not-spam for new email text, reusing common.preprocess and common.features."""
```

`v2_feature_engineering/main.py`:
```python
"""Entry point: load -> clean -> build metadata features -> train -> evaluate."""
```

- [ ] **Step 2: Create `models/.gitkeep` and `reports/.gitkeep`** (empty files)

- [ ] **Step 3: Create `v2_feature_engineering/README.md`**

```markdown
# v2: Feature Engineering

Word + character TF-IDF combined with 9 engineered metadata features (links, uppercase ratio, suspicious words, etc.).

Design: `docs/superpowers/specs/2026-07-02-email-spam-v2-design.md`

Status: scaffolded, not yet implemented.

Run (once implemented): `poetry run python -m v2_feature_engineering.main`

## v1 vs v2 comparison

_(Filled in by hand once both v1 and v2 have been trained.)_
```

- [ ] **Step 4: Verify**

Run: `poetry run python -c "import v2_feature_engineering"`
Expected: exits 0, no output.

---

### Task 6: `v3_model_comparison_tuning/` folder

**Files:**
- Create: `v3_model_comparison_tuning/__init__.py`
- Create: `v3_model_comparison_tuning/train.py`
- Create: `v3_model_comparison_tuning/evaluate.py`
- Create: `v3_model_comparison_tuning/threshold.py`
- Create: `v3_model_comparison_tuning/error_analysis.py`
- Create: `v3_model_comparison_tuning/predict.py`
- Create: `v3_model_comparison_tuning/main.py`
- Create: `v3_model_comparison_tuning/models/.gitkeep`
- Create: `v3_model_comparison_tuning/reports/.gitkeep`
- Create: `v3_model_comparison_tuning/README.md`

- [ ] **Step 1: Create `__init__.py` and the six stub modules**

`v3_model_comparison_tuning/__init__.py`:
```python
"""v3: 70/15/15 train/val/test split, compares Logistic Regression, Linear SVM, and Random Forest, tunes risk thresholds on validation, does error analysis on test."""
```

`v3_model_comparison_tuning/train.py`:
```python
"""Three-way stratified split; fits the shared ColumnTransformer on train; trains Logistic Regression, Linear SVM, and Random Forest."""
```

`v3_model_comparison_tuning/evaluate.py`:
```python
"""Evaluates all 3 candidates on validation to pick a winner by F1; final classification report, confusion matrix, and precision-recall curve on test, evaluated once."""
```

`v3_model_comparison_tuning/threshold.py`:
```python
"""Sweeps decision thresholds (0.30-0.90) on the winning model's validation-set probabilities; writes reports/threshold_analysis.csv."""
```

`v3_model_comparison_tuning/error_analysis.py`:
```python
"""Finds false positives and false negatives in test-set predictions; writes raw reports/false_positives.csv and false_negatives.csv (no auto-categorization)."""
```

`v3_model_comparison_tuning/predict.py`:
```python
"""predict_email_risk(text): loads models/best_model_v3.pkl, applies the documented risk thresholds, and returns prediction, probability, risk level, and action. Imported directly by v4_streamlit_app."""
```

`v3_model_comparison_tuning/main.py`:
```python
"""Entry point: load -> clean -> features -> three-way split -> train -> compare -> tune thresholds -> final test evaluation -> error analysis."""
```

- [ ] **Step 2: Create `models/.gitkeep` and `reports/.gitkeep`** (empty files)

- [ ] **Step 3: Create `v3_model_comparison_tuning/README.md`**

```markdown
# v3: Model Comparison, Threshold Tuning, Error Analysis

Compares Logistic Regression, Linear SVM, and Random Forest on v2's feature set; tunes a risk threshold on a validation set; analyzes false positives/negatives on a held-out test set.

Design: `docs/superpowers/specs/2026-07-02-email-spam-v3-design.md`

Status: scaffolded, not yet implemented.

Run (once implemented): `poetry run python -m v3_model_comparison_tuning.main`

## Chosen risk thresholds

_(Filled in by hand after reading `reports/threshold_analysis.csv` — this is a manual judgment call, not an automated rule. See `reports/error_analysis.md` for reasoning.)_
```

- [ ] **Step 4: Verify**

Run: `poetry run python -c "import v3_model_comparison_tuning"`
Expected: exits 0, no output.

---

### Task 7: `v4_streamlit_app/` folder + top-level `app/`

**Files:**
- Create: `v4_streamlit_app/__init__.py`
- Create: `v4_streamlit_app/explain.py`
- Create: `v4_streamlit_app/batch.py`
- Create: `v4_streamlit_app/feedback.py`
- Create: `v4_streamlit_app/main.py`
- Create: `v4_streamlit_app/README.md`
- Create: `app/streamlit_app.py`

**Interfaces:**
- Consumes (once implemented, not in this scaffolding task): `v3_model_comparison_tuning.predict.predict_email_risk`.

- [ ] **Step 1: Create `__init__.py` and the four stub modules**

`v4_streamlit_app/__init__.py`:
```python
"""v4: rule-based explanation, batch CSV prediction, and feedback logging for the Streamlit app in app/streamlit_app.py."""
```

`v4_streamlit_app/explain.py`:
```python
"""build_explanation(text, metadata_features): rule-based, model-agnostic explanation bullets sourced from common.features' metadata (links, suspicious words, exclamations, uppercase ratio, currency symbols)."""
```

`v4_streamlit_app/batch.py`:
```python
"""predict_batch(df): runs v3_model_comparison_tuning.predict.predict_email_risk over each row of an uploaded CSV (email_id, email_text)."""
```

`v4_streamlit_app/feedback.py`:
```python
"""log_feedback(...): appends one row (email text, prediction, probability, risk level, user feedback, timestamp) to data/feedback/feedback.csv, creating it if missing."""
```

`v4_streamlit_app/main.py`:
```python
"""Optional CLI smoke-test entry point for predict/explain logic, outside of Streamlit."""
```

- [ ] **Step 2: Create `v4_streamlit_app/README.md`**

```markdown
# v4: Streamlit Prototype App

Interactive app: paste an email, get spam probability, risk level, recommended action, and a plain-language explanation. Also supports batch CSV upload and feedback logging.

Design: `docs/superpowers/specs/2026-07-02-email-spam-v4-design.md`

Status: scaffolded, not yet implemented.

Requires v3 to be trained first (`v3_model_comparison_tuning/models/best_model_v3.pkl` must exist).

Run (once implemented): `poetry run streamlit run app/streamlit_app.py`

This is a prototype demonstrating spam-risk classification using supervised machine learning. It should not be used as a production security system without additional validation, sender/domain features, adversarial testing, and privacy controls.
```

- [ ] **Step 3: Create `app/streamlit_app.py`**

```python
"""Streamlit UI: calls v3_model_comparison_tuning.predict.predict_email_risk() and v4_streamlit_app.explain.build_explanation(); renders single-email, batch upload, and feedback sections."""
```

- [ ] **Step 4: Verify**

Run: `poetry run python -c "import v4_streamlit_app"`
Expected: exits 0, no output.

---

### Task 8: `tests/` directory

**Files:**
- Create: `tests/.gitkeep`

**Interfaces:**
- Produces: an empty `tests/` directory ready for each version's own implementation plan to add real test files into (`test_preprocess_v1.py`, `test_features_v2.py`, `test_threshold_v3.py`, `test_error_analysis_v3.py`, `test_explain_v4.py`, `test_batch_v4.py` — none created here, since there is no logic yet to test).

- [ ] **Step 1: Create `tests/.gitkeep`** (empty file — no test files yet, since `pythonpath = ["."]` in `pyproject.toml` from Task 1 already makes `common` and `vN_*` importable from test files once they're written)

- [ ] **Step 2: Verify pytest runs cleanly with zero tests**

Run: `poetry run pytest`
Expected: `no tests ran` (exit code 5) or `collected 0 items` — not an error, just nothing to run yet.

---

### Task 9: Top-level `README.md`

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the top-level README**

```markdown
# Email Spam Classification — Risk Triage System

An intern learning project: an email spam classifier built as four progressively deeper versions, each with its own design spec.

| Version | Focus | Spec | Status |
|---|---|---|---|
| v1 | Basic TF-IDF + Naive Bayes / Logistic Regression pipeline | [spec](docs/superpowers/specs/2026-07-01-email-spam-v1-design.md) | scaffolded |
| v2 | Word+char TF-IDF + engineered metadata features | [spec](docs/superpowers/specs/2026-07-02-email-spam-v2-design.md) | scaffolded |
| v3 | Model comparison, threshold tuning, error analysis | [spec](docs/superpowers/specs/2026-07-02-email-spam-v3-design.md) | scaffolded |
| v4 | Streamlit prototype app | [spec](docs/superpowers/specs/2026-07-02-email-spam-v4-design.md) | scaffolded |

## Setup

```bash
poetry install --no-root
poetry run python -m common.download_data   # once implemented — pulls the Kaggle dataset into data/raw/
```

## Repo layout

- `common/` — code shared across all versions (config, data loading, cleaning, feature extraction)
- `v1_basic_pipeline/`, `v2_feature_engineering/`, `v3_model_comparison_tuning/`, `v4_streamlit_app/` — one self-contained pipeline per version
- `app/` — the Streamlit entry point for v4
- `data/` — gitignored raw/processed/feedback data
- `docs/superpowers/specs/` — design docs for each version
- `tests/` — pytest unit tests, organized per version
```

- [ ] **Step 2: Verify**

Run: `poetry run python -c "import pathlib; assert pathlib.Path('README.md').exists()"`
Expected: exits 0, no output.

---

### Task 10: Final structure verification (no commit)

**Files:** none created — verification only.

- [ ] **Step 1: Confirm the full tree matches the specs**

Run: `poetry run python -c "
import pathlib
expected = ['common', 'v1_basic_pipeline', 'v2_feature_engineering', 'v3_model_comparison_tuning', 'v4_streamlit_app', 'app', 'data', 'tests', 'docs']
missing = [p for p in expected if not pathlib.Path(p).exists()]
assert not missing, missing
print('all top-level paths present')
"`
Expected: `all top-level paths present`

- [ ] **Step 2: Run the full check suite one more time**

Run: `poetry check && poetry run pytest && poetry run python -c "import common, v1_basic_pipeline, v2_feature_engineering, v3_model_comparison_tuning, v4_streamlit_app"`
Expected: all three succeed with no errors.

- [ ] **Step 3: Show the user what's staged, without committing**

Run: `git status`
Expected: lists all newly created files as untracked (or staged, if `git add` was run) — do NOT run `git commit`. Ask the user before committing, per standing instructions.
