# v3 Error Analysis

Best validation model: `linear_svm`

Chosen thresholds:
- Low risk: spam probability < 0.40
- Medium risk: 0.40 <= spam probability < 0.80
- High risk: spam probability >= 0.80
- Classification threshold used for final metrics: 0.60

Reasoning:
The threshold sweep is written to `threshold_analysis.csv`. The classification threshold is 0.60 because it had the strongest validation F1 among the swept thresholds while keeping recall high. The test set remained untouched until the final evaluation pass.

Manual review notes:
- Review `false_positives.csv` for legitimate messages that look promotional or urgent.
- Review `false_negatives.csv` for spam that uses neutral wording or avoids obvious spam indicators.
- Add human categories here after inspecting the exported rows; this file intentionally avoids auto-tagging errors.
