# Email Spam Classification — Version 3 Design (Model Comparison, Threshold Tuning, Error Analysis)

Date: 2026-07-02
Status: Approved (pending user sign-off on written spec)
Depends on:
- `docs/superpowers/specs/2026-07-01-email-spam-v1-design.md` (as amended)
- `docs/superpowers/specs/2026-07-02-email-spam-v2-design.md` (as amended —
  `common/features.py` must exist before v3 is implemented)

## Context

v1 proved a baseline pipeline; v2 proved better feature engineering. v3's job
is to prove the user can think past a single accuracy number: compare model
architectures honestly, tune a decision threshold using a proper validation
set (not the final test set), and manually study *why* the model fails on
specific emails. The user provided a detailed outline for this; this spec is
the corrected, buildable version of it, expanded per user decision to include
a model-comparison step (the original 5-version roadmap called v3 "model
comparison + threshold tuning + error analysis," but the pasted outline only
covered the latter two — resolved by adding Random Forest as a third
candidate model, see decisions below).

## Key decisions

1. **Model comparison added: Random Forest joins Logistic Regression and
   Linear SVM as a third candidate.** Tree-based, handles the mixed sparse
   TF-IDF + dense metadata feature set without extra tuning, trains fast, and
   gives `feature_importances_` as a bonus for the write-up. XGBoost was
   considered but rejected — an extra dependency and tuning surface not
   worth it for a portfolio-scoped project.
2. **`common/features.py` (moved there during this design, see v2
   amendment) is reused as-is.** v3 is not about new features — it's about
   evaluating and choosing between models built on the same feature set v2
   already engineered.
3. **Three-way stratified split: 70% train / 15% validation / 15% test.**
   Training set fits the 3 candidate models. Validation set is used for BOTH
   picking the winning model AND sweeping decision thresholds — this keeps
   the true test set completely unseen until the single final evaluation
   pass, matching the outline's stated reasoning for why a validation set is
   needed at all.
4. **Threshold selection is a manual, documented judgment call — not an
   automated rule.** `threshold.py` produces the full sweep table
   (`threshold_analysis.csv`); the low/medium/high cutoffs are chosen by
   reading that table and the reasoning is written into `error_analysis.md`.
   A fixed automated rule would just relocate the arbitrary decision into
   code instead of making it, which defeats the point of this version (the
   narrative that a human reasoned about the precision/recall tradeoff).
5. **Error categorization is manual, not auto-tagged.** `error_analysis.py`
   exports raw `false_positives.csv` / `false_negatives.csv` (email text,
   actual label, predicted label, spam probability). Categorizing them
   (`promotional_language`, `formal_phishing_style`, etc.) is a human review
   task written into `error_analysis.md` — keyword-based auto-tagging would
   be circular (tagging on the same words the model itself reacts to).
6. **New dependency: `matplotlib`**, for `confusion_matrix_v3.png` and
   `precision_recall_curve.png`.

## Architecture

```text
v3_model_comparison_tuning/
├── train.py             # 70/15/15 split, trains LogReg + SVM + Random Forest
├── evaluate.py           # classification report, confusion matrix, PR curve
├── threshold.py          # threshold sweep on validation set -> threshold_analysis.csv
├── error_analysis.py     # exports false_positives.csv / false_negatives.csv from test set
├── predict.py             # loads best_model_v3.pkl, applies chosen risk thresholds
├── main.py
├── models/                 # gitignored — best_model_v3.pkl (bundled Pipeline, winner)
├── reports/
│   ├── metrics_v3.json
│   ├── model_comparison.csv          # LogReg vs SVM vs RandomForest, on validation set
│   ├── threshold_analysis.csv
│   ├── confusion_matrix_v3.png
│   ├── precision_recall_curve.png
│   ├── false_positives.csv
│   ├── false_negatives.csv
│   └── error_analysis.md             # hand-written narrative + chosen thresholds + reasoning
└── README.md
```

Imports `common/preprocess.py`, `common/features.py`, `common/io_utils.py`,
`common/config.py`. No import from `v1_basic_pipeline/` or
`v2_feature_engineering/`, per the standing "versions don't reference each
other" rule.

## Data flow

```text
load_dataset() -> common.preprocess.clean_text() -> common.features.build_metadata_features()
        |
        v
train_test_split(..., stratify=y, random_state=RANDOM_SEED, test_size=0.15)
   -> test_data (15%) carved off first, untouched until the final step
        |
        v
train_test_split(remaining_85pct, stratify=y, random_state=RANDOM_SEED,
                  test_size=0.15/0.85)   # ~0.176, yields 70/15/15 of the original whole
        |
        v
train_data (70%)            val_data (15%)             test_data (15%, still untouched)
        |
        v
ColumnTransformer (word_tfidf, char_tfidf, metadata) — same shape as v2's —
   fit on train_data only
        |
        v
train Logistic Regression(class_weight='balanced'), Linear SVM,
      Random Forest(class_weight='balanced') on train_data
        |
        v
evaluate all 3 on val_data -> reports/model_comparison.csv, pick winner by F1
        |
        v
threshold.py: sweep thresholds [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
              using the WINNING model's probabilities on val_data
              -> for each threshold: precision, recall, f1, false_positive_count,
                 false_negative_count
              -> reports/threshold_analysis.csv
        |
        v
[MANUAL STEP] read threshold_analysis.csv, choose low/medium/high risk cutoffs,
              write the reasoning into error_analysis.md
        |
        v
FINAL EVALUATION (test_data touched exactly once, here):
   winning model + chosen thresholds evaluated on test_data
   -> reports/metrics_v3.json (classification report: precision/recall/f1 per class)
   -> reports/confusion_matrix_v3.png
   -> reports/precision_recall_curve.png
        |
        v
error_analysis.py: on test_data predictions, find rows where
      actual=0 & predicted=1 (false positives) and actual=1 & predicted=0
      (false negatives)
   -> reports/false_positives.csv, reports/false_negatives.csv
      (raw: email_text, actual_label, predicted_label, spam_probability —
       no auto-generated reason_category column)
        |
        v
save winning model as ONE bundled Pipeline: models/best_model_v3.pkl
```

`predict.py` loads `best_model_v3.pkl`, applies `common.preprocess.clean_text`
+ `common.features.build_metadata_features` to new input, gets a spam
probability from `pipeline.predict_proba(...)`, and maps it to
low/medium/high risk using the thresholds chosen and documented in
`error_analysis.md` (hardcoded as named constants at the top of `predict.py`,
sourced from that document — not re-derived at runtime).

## Testing (v3)

Unit tests target the sweep/lookup logic, not model accuracy:

- `tests/test_threshold_v3.py`
  - given a small synthetic array of (probability, actual_label) pairs,
    sweeping thresholds produces the expected precision/recall at each
    threshold (hand-computed expected values)
  - threshold table has the expected columns and no NaNs
- `tests/test_error_analysis_v3.py`
  - false-positive finder returns only rows where `actual=0, predicted=1`
  - false-negative finder returns only rows where `actual=1, predicted=0`
  - on a synthetic set with zero false positives, returns an empty result
    (not an error)

Run via `poetry run pytest`, alongside the v1/v2 test files already in
`tests/`.

## Explicitly out of scope for v3

XGBoost or any further model types beyond LogReg/SVM/RandomForest, automated
threshold selection, automated error categorization (see decisions 3-5
above), Streamlit/deployment (v4), BERT/deep learning/LLMs (never, per
original roadmap).

## Open items for later (not blocking v3)

- The exact chosen low/medium/high thresholds are a data-dependent outcome
  of the manual step — not fixed in this spec, since they can only be picked
  after seeing real `threshold_analysis.csv` numbers.
- v4 internal design — out of scope for this spec.
