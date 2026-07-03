# v3: Model Comparison, Threshold Tuning, Error Analysis

Compares Logistic Regression, calibrated Linear SVM, and Random Forest on the shared v2 feature set. Uses a 70/15/15 split: validation selects the best model and drives threshold analysis, while the test set is used once for final metrics and error exports.

Design: `docs/superpowers/specs/2026-07-02-email-spam-v3-design.md`

Run:

```bash
poetry run python -m v3_model_comparison_tuning.main
```

Outputs:

- `models/best_model_v3.pkl`
- `reports/model_comparison.csv`
- `reports/threshold_analysis.csv`
- `reports/metrics_v3.json`
- `reports/confusion_matrix_v3.png`
- `reports/precision_recall_curve.png`
- `reports/false_positives.csv`
- `reports/false_negatives.csv`
- `reports/error_analysis.md`
