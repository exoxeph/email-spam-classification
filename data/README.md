# Data

- `raw/` - downloaded via `common/download_data.py` (Kaggle Enron-based spam dataset). Gitignored.
- `processed/` - intermediate cleaned data, if any version writes it. Gitignored.
- `feedback/` - user feedback collected by the v4 Streamlit app (`feedback.csv`). Gitignored.

None of these are committed to the repo. Run the download script to populate `raw/` before training any version.
