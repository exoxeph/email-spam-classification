# v2: Feature Engineering

Word and character TF-IDF features combined with engineered email metadata.

Design: `docs/superpowers/specs/2026-07-02-email-spam-v2-design.md`

Run:

```bash
poetry run python -m v2_feature_engineering.main
```

Outputs:

- `models/best_model_v2.pkl`
- `reports/metrics_v2.json`
- `reports/model_comparison.csv`
- `reports/feature_summary.md`

## v1 vs v2 comparison

Filled in by hand after training both versions on the same dataset.
