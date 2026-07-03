# v4: Streamlit Prototype App

Interactive spam-risk triage UI backed by the v3 trained model.

Run v3 first so `v3_model_comparison_tuning/models/best_model_v3.pkl` exists:

```bash
poetry run python -m v3_model_comparison_tuning.main
```

Run the app:

```bash
poetry run streamlit run app/streamlit_app.py
```

Features:

- Single-email prediction with probability, risk level, and recommended action.
- Rule-based explanation bullets based on metadata signals.
- Batch CSV upload with required columns `email_id,email_text`.
- Optional `subject` column in batch uploads.
- Feedback logging to `data/feedback/feedback.csv`.

Manual checklist:

- Paste the safe, spam, and medium-risk examples.
- Confirm probability, risk level, action, and explanation bullets render.
- Upload a CSV with `email_id,email_text` and download the result CSV.
- Click feedback Yes/No and confirm `data/feedback/feedback.csv` is appended.

Disclaimer: this is a prototype demonstrating spam-risk classification. It is not a production security system and should not be relied on without further validation, sender/domain features, adversarial testing, monitoring, and operational controls.
