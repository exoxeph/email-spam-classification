# Email Spam Classification — Version 4 Design (Streamlit Prototype App)

Date: 2026-07-02
Status: Approved (pending user sign-off on written spec)
Depends on:
- `docs/superpowers/specs/2026-07-01-email-spam-v1-design.md` (as amended)
- `docs/superpowers/specs/2026-07-02-email-spam-v2-design.md` (as amended)
- `docs/superpowers/specs/2026-07-02-email-spam-v3-design.md` — v4 requires
  v3's trained `best_model_v3.pkl` and its documented risk thresholds to
  already exist (v3 must be implemented and run before v4 can work)

## Context

v1-v3 built and evaluated the model. v4 turns it into something a recruiter
can actually run and click through: a Streamlit app where a user pastes an
email and gets back a prediction, spam probability, risk level, recommended
action, and a plain-language explanation. The user provided a detailed
outline for this; this spec is the corrected, buildable version of it.

Unlike v1→v2→v3 (where each version built new modeling logic that later
versions reused via `common/`), v4 is fundamentally a consumption/demo layer
for v3's specific output — a trained model file and a set of thresholds
someone reasoned about and documented. That's an intentional, direct
dependency, not the kind of peer-version coupling the earlier specs avoided.

## Key decisions

1. **v4 imports directly from `v3_model_comparison_tuning/`** — specifically
   `predict.predict_email_risk()` and the model path
   `v3_model_comparison_tuning/models/best_model_v3.pkl`. This avoids a
   second copy of the risk thresholds that could silently drift out of sync
   with the reasoning written in v3's `error_analysis.md`.
2. **Full scope in one pass**: single-email prediction, batch CSV upload,
   and feedback logging are all built now (not deferred), per user decision.
   Batch and feedback both reuse `predict_email_risk()` directly, so the
   incremental cost is low once single-email prediction works.
3. **Explanation is rule-based only**, sourced entirely from
   `common/features.py`'s already-computed metadata (link count, suspicious
   word matches, exclamation count, uppercase ratio, currency symbols). This
   is model-agnostic — it works identically regardless of which of the 3
   candidate models (Logistic Regression, Linear SVM, Random Forest) v3
   actually picked as the winner. Model-weight-based explanation (e.g. top
   TF-IDF coefficients) was considered and rejected: it would only work
   cleanly if the winner is Logistic Regression, adding a conditional path
   for a benefit that may not even apply. Noted as a future improvement in
   the README, not built.
4. **Three-way prediction labeling**, per the outline: probability < 0.40 →
   "Not Spam" / Low / Allow; 0.40-0.79 → "Needs Review" / Medium / Warn
   User; >= 0.80 → "Spam" / High / Quarantine. (Exact cutoffs come from
   whatever v3's `error_analysis.md` documents, not hardcoded here.)
5. **New dependency: `streamlit`**, added to `pyproject.toml`.

## Architecture

```text
app/
└── streamlit_app.py       # UI only — calls into v3's predict_email_risk() + v4's explain.py

v4_streamlit_app/
├── explain.py                # rule-based explanation bullets from metadata features
├── batch.py                   # batch CSV predict loop, reuses predict_email_risk()
├── feedback.py                 # append feedback row to data/feedback/feedback.csv
├── main.py                      # optional CLI smoke-test entrypoint (non-Streamlit)
└── README.md

data/feedback/
└── feedback.csv               # gitignored, created at runtime if missing
```

`streamlit_app.py` imports `predict_email_risk` from
`v3_model_comparison_tuning.predict` and `build_explanation` from
`v4_streamlit_app.explain`. It reaches `common.preprocess` /
`common.features` only indirectly, through `predict_email_risk`.

## Data flow

```text
User pastes email text into st.text_area, clicks "Analyze Email"
        |
        v
v3_model_comparison_tuning.predict.predict_email_risk(text)
   -> common.preprocess.clean_text -> common.features.build_metadata_features
   -> pipeline.predict_proba -> maps probability to
      (prediction_label, risk_level, action) using v3's documented thresholds
        |
        v
v4_streamlit_app.explain.build_explanation(text, metadata_features)
   -> rule-based bullets, each shown only if the underlying count > 0:
        "Contains N link(s)"
        "Contains suspicious word(s): <matched words>"
        "Uses N exclamation mark(s)"
        "Unusually high uppercase ratio" (if uppercase_ratio > 0.3)
        "Contains currency symbol(s)"
      Fallback if none apply: "No strong suspicious signals detected."
        |
        v
Streamlit renders: prediction label, probability %, risk level, action, explanation bullets
        |
        v
[optional] user clicks "Was this correct? Yes/No"
   -> v4_streamlit_app.feedback.log_feedback(text, prediction, probability,
      risk_level, user_feedback, timestamp)
   -> appends one row to data/feedback/feedback.csv (created if missing)

Batch path (separate tab/section in the app):
User uploads CSV (columns: email_id, email_text)
        |
        v
v4_streamlit_app.batch.predict_batch(df)
   -> loops predict_email_risk() per row
        |
        v
Streamlit shows results table (email_id, email_text, spam_probability,
   risk_level, recommended_action) + a "Download results CSV" button
```

## App sections (per outline)

```text
1. Title: "Email Spam Risk Triage System"
   Subtitle: "Paste an email to estimate whether it should be allowed,
   warned, or quarantined."
2. Email input box (st.text_area)
3. "Analyze Email" button
4. Prediction label (Not Spam / Needs Review / Spam)
5. Spam probability (%)
6. Risk level (Low / Medium / High)
7. Recommended action (Allow / Warn User / Quarantine)
8. Explanation bullets (see above)
9. Three preset example buttons: safe email, spam email, medium-risk email
   (hardcoded strings in streamlit_app.py, taken from the outline's examples)
10. Batch CSV upload tab/section
11. Feedback Yes/No buttons under the single-email result
```

README includes the outline's disclaimer: this is a prototype demonstrating
spam-risk classification, not a production security system, and should not
be relied on without further validation, sender/domain features, and
adversarial testing.

## Testing (v4)

- `tests/test_explain_v4.py`
  - a synthetic metadata dict with `link_count=2` produces a bullet
    containing "2 link(s)"
  - an all-zero metadata dict produces the "no strong signals" fallback
  - suspicious word bullet lists the matched words, not just a count
- `tests/test_batch_v4.py`
  - `predict_batch` on a small synthetic DataFrame (2-3 rows) returns one
    output row per input row with the expected columns
  - preserves `email_id` from input to output
- No test drives the Streamlit UI itself — out of scope for pytest. A manual
  run-through checklist (from the outline: can paste safe/spam/medium email,
  probability shows, risk level shows, action shows, explanation shows,
  batch upload works, feedback logging works) goes into v4's README instead
  of an automated test.

## Explicitly out of scope for v4

Gmail login/integration, real email scraping, browser extension, cloud
deployment, Docker, database, LLM chatbot, automatic retraining from
feedback, user accounts, payment features — all per the outline's exclusion
list.

## Open items for later (not blocking v4)

- Exact screenshot set (`app_home.png`, `spam_prediction.png`,
  `medium_risk_warning.png`, `batch_upload_result.png`) is captured manually
  by the user after the app runs, not generated by any script.
- This is the last version in the roadmap — v5 ("final polish") was already
  dropped as a distinct design/build cycle back in the v1 spec.
