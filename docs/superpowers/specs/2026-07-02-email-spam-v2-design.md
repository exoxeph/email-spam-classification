# Email Spam Classification — Version 2 Design (Feature Engineering)

Date: 2026-07-02
Status: Approved (pending user sign-off on written spec)
Depends on: `docs/superpowers/specs/2026-07-01-email-spam-v1-design.md`
(including its 2026-07-02 amendment — `common/preprocess.py` and separate
`subject`/`body` columns must exist before v2 is implemented)
Amended: 2026-07-02 — see "Amendment (2026-07-02, during v3 design)" section,
made before any v2 code was written.

## Context

v1 proved a clean baseline: word-level TF-IDF + Naive Bayes / Logistic
Regression. v2's job is to prove the user can go beyond "just words" by
engineering features that capture *how* an email is written, not just what
words it contains — link counts, urgency markers, obfuscated-word detection
via character n-grams, etc. The user provided a detailed outline for this;
this spec is the corrected, buildable version of it.

## Key decisions from the original outline

1. **Naive Bayes is dropped in v2.** `MultinomialNB` requires non-negative
   input features. Once numeric metadata features (uppercase ratio, link
   count, etc.) are scaled with `StandardScaler`, values can go negative,
   which breaks NB. Rather than special-casing NB to only see TF-IDF (as the
   original outline vaguely implied), v2 uses **Logistic Regression
   (`class_weight='balanced'`) and Linear SVM** — both handle the full
   combined sparse+dense feature set natively. Multi-model comparison is
   v3's explicit job anyway, so v2 doesn't need three models.
2. **Feature combination via `ColumnTransformer`**, not manual
   `scipy.sparse.hstack`. Keeps everything as one `sklearn.Pipeline` object,
   consistent with how v1 bundles vectorizer+model — `predict.py` stays a
   one-line `pipeline.predict(...)` call.
3. **Numeric features are scaled with `StandardScaler(with_mean=False)`**
   (not the default centering), so they stay compatible with the sparse
   TF-IDF matrices they're combined with in the same `ColumnTransformer`.
4. **v1-vs-v2 comparison table is written by hand** into v2's README once
   both are trained — not computed by having v2 read v1's `metrics.json`
   programmatically, which would violate the "versions don't reference each
   other" rule from the v1 spec.

## Architecture

```text
v2_feature_engineering/
├── train.py
├── evaluate.py
├── predict.py
├── main.py
├── models/            # gitignored — best_model_v2.pkl (one bundled Pipeline)
├── reports/            # metrics_v2.json, model_comparison.csv, feature_summary.md
└── README.md          # includes hand-written v1-vs-v2 comparison table
```

Imports `common/preprocess.py` (clean_text, map_label),
`common/features.py` (metadata extractors + ColumnTransformer builder — see
2026-07-02 amendment below), `common/io_utils.py` (load_dataset — now
returning separate subject/body), and `common/config.py` (RANDOM_SEED,
TEST_SIZE) exactly like v1 does, per the "self-contained version, shared
common only" rule. No import from `v1_basic_pipeline/`.

## Feature list

```text
Text representations (both fit only on training data, per v1's leakage fix):
  - word_tfidf:  TfidfVectorizer(ngram_range=(1,1)) on combined subject+body
  - char_tfidf:  TfidfVectorizer(analyzer="char", ngram_range=(3,5)) on
                 combined subject+body — catches obfuscation like "fr33"

Metadata features (9 numeric columns, computed by common/features.py):
  - email_char_length, email_word_count       (from body)
  - subject_char_length, subject_exclaim_count (from subject, kept separate)
  - link_count            (http/https/www./.com/.net/.org occurrences)
  - exclamation_count     (from full text)
  - uppercase_ratio        (fraction of alphabetic chars that are uppercase)
  - suspicious_word_count  (hand-picked list: urgent, winner, free, claim,
                            prize, cash, verify, account, limited, offer,
                            click, password, login, reward)
  - currency_symbol_count  ($, £, €, ৳)
  - digit_count
```

That's 11 feature-producing units total (2 text representations + 9 numeric
columns), matching the outline's list but pruned of ambiguity about where
subject-only features get their input.

## Data flow

```text
load_dataset()  [common/io_utils.py]  -> subject, body, label columns
        |
        v
common.preprocess.clean_text() on subject and body separately
common.preprocess.map_label()  on label
        |
        v
common.features.build_metadata_features(df) -> 9 numeric columns
combined_text = subject + " " + body   (for the two TF-IDF branches only)
        |
        v
train_test_split(..., stratify=y, random_state=RANDOM_SEED, test_size=TEST_SIZE)
   [same config values as v1, for a fair comparison]
        |
        v
ColumnTransformer(
    ("word_tfidf", TfidfVectorizer(ngram_range=(1,1)),                "combined_text"),
    ("char_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3,5)),"combined_text"),
    ("metadata",   StandardScaler(with_mean=False),                   [9 numeric cols]),
)
   fit_transform on train only, transform on test only  [leakage fix carried over]
        |
        v
train Logistic Regression(class_weight='balanced') AND Linear SVM
        |
        v
evaluate both: accuracy, precision, recall, f1, confusion matrix
        |
        v
pick best by F1 -> wrap as ONE sklearn.Pipeline([("features", column_transformer),
                                                   ("clf", model)])
        |
        v
save single bundle: v2_feature_engineering/models/best_model_v2.pkl
        |
        v
write reports/metrics_v2.json + model_comparison.csv + feature_summary.md
```

`predict.py` loads `best_model_v2.pkl`, re-applies `common.preprocess.clean_text`
and `common.features.build_metadata_features` to new input, then calls
`pipeline.predict(...)`.

## Testing (v2)

Same philosophy as v1 — unit test pure functions, not model accuracy:

- `tests/test_features_v2.py`
  - `count_links` finds http/https/www variants
  - `count_exclamations` counts correctly on mixed punctuation
  - `uppercase_ratio` returns 0 for all-lowercase, 1 for all-uppercase
  - `count_suspicious_words` is case-insensitive
  - `count_currency_symbols` / `count_digits` basic correctness
  - `build_metadata_features` returns all 9 expected columns, no NaNs

Run via `poetry run pytest` (same single test suite root as v1,
`tests/test_features_v2.py` alongside `test_preprocess_v1.py`).

## Explicitly out of scope for v2

Model comparison beyond LogReg/SVM, threshold tuning, error analysis (all
v3), Streamlit/deployment/Docker/database (v4 or never), BERT/deep
learning/LLMs (never, per original roadmap), automated cross-version metrics
comparison (kept manual, see key decision 4).

## Open items for later (not blocking v2)

- Exact suspicious-word list may get refined once real data is seen — not a
  blocker, just noted as a hand-tuned heuristic likely to be adjusted.
- v3 internal design — out of scope for this spec.

## Amendment (2026-07-02, during v3 design)

Made before any v2 code existed — relocating logic, not reworking it:

- **`features.py` (pure metadata extractors + the `ColumnTransformer`
  builder) moves to `common/features.py`.** v3 needs the identical feature
  engineering v2 built (same word/char TF-IDF + 9 metadata columns) to run
  its model comparison and threshold tuning on. This feature engineering
  isn't part of what makes v3 "deeper" than v2 — v3's depth comes from model
  comparison, threshold tuning, and error analysis — so duplicating it into
  a third copy would just be code rot. v2's own behavior/output is
  unchanged; it now imports from `common.features` instead of defining the
  functions itself.
