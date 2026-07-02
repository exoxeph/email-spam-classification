# Email Spam Classification — Repo Architecture + Version 1 Design

Date: 2026-07-01
Status: Approved (pending user sign-off on written spec)
Amended: 2026-07-02 — see "Amendment (2026-07-02)" section, made while
designing v2, before any v1 code was written.

## Context

This is an intern learning project: build an email spam classifier, but instead
of a single script, build it as a sequence of progressively deeper versions
(v1 → v4) to demonstrate growth in ML engineering skill. v5 ("final polish")
was dropped as a distinct version — it's not substantive enough to warrant its
own design/build cycle.

The user brought a detailed v1 outline (dataset → clean → split → TF-IDF →
Naive Bayes / Logistic Regression → evaluate). This spec captures the
corrected, buildable version of that outline plus the repo-wide structure
that v2–v4 will slot into later.

## Repo-wide architecture

Single Poetry project, one virtualenv for the whole repo. A shared `common/`
package holds only things that are truly identical across every version
(data download, artifact I/O, shared constants). Each version folder
(`v1_basic_pipeline/`, `v2_feature_engineering/`, `v3_model_comparison_tuning/`,
`v4_streamlit_app/`) is a self-contained, independently readable pipeline that
imports from `common/` but never from another version folder.

```text
email-spam-classification/
├── pyproject.toml / poetry.lock
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/                    # gitignored
│   ├── processed/              # gitignored
│   └── README.md
│
├── common/
│   ├── __init__.py
│   ├── download_data.py        # kagglehub -> data/raw/, idempotent
│   ├── io_utils.py             # load_dataset(), save_artifact(), load_artifact()
│   ├── preprocess.py           # clean_text(), map_label() — shared, see amendment
│   └── config.py               # RANDOM_SEED, TEST_SIZE, paths
│
├── v1_basic_pipeline/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── main.py
│   ├── models/                 # gitignored
│   ├── reports/                # metrics.json, model_comparison.csv
│   └── README.md
│
├── v2_feature_engineering/      # scaffolded empty now, designed later
├── v3_model_comparison_tuning/  # scaffolded empty now, designed later
├── v4_streamlit_app/            # scaffolded empty now, designed later
│
└── tests/
    ├── test_preprocess_v1.py
    └── test_train_v1.py
```

v2–v4 folders are created empty (with a placeholder `README.md` stating
"designed later") in this pass. Their internals get their own brainstorm/spec
cycle when the user is ready to build them.

## Dataset

Enron-based spam email dataset from Kaggle (subject + body + spam/ham label).
Exact Kaggle slug to be confirmed at download-script implementation time (not
yet picked by user — `common/download_data.py` will target whichever public
Enron spam dataset is best-maintained on Kaggle).

## Data flow (v1) — with leakage/reproducibility fixes

```text
download_data.py (if data/raw/ empty)
        |
        v
load_dataset()               [common/io_utils.py]
        |
        v
preprocess.clean_text() + preprocess.map_label()   [pure functions]
        |
        v
train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)
        |
        +--> X_train, y_train --> TfidfVectorizer.fit_transform()
        |
        +--> X_test,  y_test  --> (same vectorizer).transform()
        |
        v
train Naive Bayes AND Logistic Regression(class_weight='balanced')
        |
        v
evaluate both: accuracy, precision, recall, f1, confusion matrix
        |
        v
pick best by F1 -> wrap as sklearn.Pipeline([("tfidf", vectorizer), ("clf", model)])
        |
        v
save single bundle: v1_basic_pipeline/models/best_model.pkl
        |
        v
write v1_basic_pipeline/reports/metrics.json + model_comparison.csv
```

`predict.py` loads `best_model.pkl` (one artifact — vectorizer and model can
never mismatch) and reuses `preprocess.clean_text()` before calling `.predict()`.

## Issues fixed vs. the original outline

1. **TF-IDF leakage** — vectorizer fit only on train, never on full/test data.
2. **No stratified split** — `stratify=y` added, keeps class ratio consistent
   between train/test.
3. **No fixed seed** — `RANDOM_SEED = 42` centralized in `common/config.py`.
4. **Class imbalance unaddressed** — `class_weight='balanced'` on Logistic
   Regression; imbalance ratio logged in evaluation output.
5. **Vectorizer/model saved separately** — bundled into one `sklearn.Pipeline`,
   one `.pkl` file, eliminating mismatch risk.
6. **Cleaning logic duplicated** — `preprocess.clean_text()` is a single pure
   function imported by both `train.py` and `predict.py`.
7. **Hardcoded paths/params scattered** — centralized in `common/config.py`.
8. **Raw data risk of being committed** — `.gitignore` excludes `data/raw/`,
   `data/processed/`, `*.pkl`, `kaggle.json`; `data/README.md` documents
   provenance instead.

## Tooling

- **Poetry** for dependency management (`pyproject.toml` + `poetry.lock`),
  single project for the whole repo.
- Dependencies: `scikit-learn`, `pandas`, `numpy`, `joblib`, `kagglehub`.
- Dev dependency: `pytest`.

## Testing (v1)

Unit tests target pure functions only — not model accuracy (data-dependent,
not a pass/fail concern):

- `tests/test_preprocess_v1.py`
  - `clean_text` lowercases input
  - `clean_text` strips extra whitespace
  - `map_label` maps spam→1, ham→0
  - `build_dataset` drops empty/null rows
- `tests/test_train_v1.py`
  - train/test split preserves class ratio within tolerance (stratification
    check)

Run via `poetry run pytest`.

## Explicitly out of scope for v1

BERT, deep learning, LLMs, phishing-specific detection, user feedback loops,
Streamlit UI, database, Docker, cloud deployment, complex dashboards. These
are reserved for v2–v4 (feature engineering, model comparison/tuning,
Streamlit app respectively), designed in their own future spec passes.

## Open items for later (not blocking v1)

- Exact Kaggle dataset slug for `download_data.py` (to be confirmed when
  implementing the download script).
- v2/v3/v4 internal design — out of scope for this spec, to be brainstormed
  separately when the user is ready to build each one.

## Amendment (2026-07-02)

Made while designing v2, before any v1 code existed — no rework involved,
just relocating where two pieces of logic live:

1. **`clean_text()` and `map_label()` move to `common/preprocess.py`** (out of
   `v1_basic_pipeline/`). This cleaning logic doesn't change across versions,
   so v2/v3/v4 import it from `common` instead of duplicating it. v1's
   pipeline imports these two functions from `common.preprocess` instead of
   defining them itself; behavior is unchanged.
2. **`common/io_utils.load_dataset()` returns `subject` and `body` as
   separate columns**, not pre-combined. v1's own step still concatenates
   them into one `text` field before vectorizing (v1's output/behavior does
   not change) — but v2 needs subject and body separately for subject-specific
   metadata features, so the split happens once at the loader level rather
   than v2 trying to re-derive subject from a combined string.
