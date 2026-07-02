# v1: Basic Pipeline

Word-level TF-IDF + Naive Bayes / Logistic Regression baseline.

Design: `docs/superpowers/specs/2026-07-01-email-spam-v1-design.md`

Status: implemented and verified end-to-end. Best model: naive_bayes (F1 0.9813, accuracy 0.9818 on the Kaggle enron-2006 dataset).

Run: `poetry run python -m v1_basic_pipeline.main`
