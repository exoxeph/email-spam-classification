# Email Spam Classification - Risk Triage System

An intern learning project: an email spam classifier built as four progressively deeper versions, each with its own design spec.

| Version | Focus | Spec | Status |
|---|---|---|---|
| v1 | Basic TF-IDF + Naive Bayes / Logistic Regression pipeline | [spec](docs/superpowers/specs/2026-07-01-email-spam-v1-design.md) | implemented |
| v2 | Word+char TF-IDF + engineered metadata features | [spec](docs/superpowers/specs/2026-07-02-email-spam-v2-design.md) | planned |
| v3 | Model comparison, threshold tuning, error analysis | [spec](docs/superpowers/specs/2026-07-02-email-spam-v3-design.md) | planned |
| v4 | Streamlit prototype app | [spec](docs/superpowers/specs/2026-07-02-email-spam-v4-design.md) | planned |

## Setup

```bash
poetry install --no-root
cp .env.example .env   # fill in KAGGLE_USERNAME and KAGGLE_KEY
poetry run python -m v1_basic_pipeline.main   # downloads the dataset, trains, evaluates, saves the model
```

## Repo layout

- `common/` - code shared across all versions (config, data loading, cleaning, feature extraction)
- `v1_basic_pipeline/` - the v1 pipeline (implemented); `v2_feature_engineering/`, `v3_model_comparison_tuning/`, `v4_streamlit_app/` will be added when those versions start
- `data/` - gitignored raw/processed/feedback data
- `docs/superpowers/specs/` - design docs for each version
- `tests/` - pytest unit tests, organized per version
