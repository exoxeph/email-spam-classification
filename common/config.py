"""Repo-wide constants: reproducibility seed, split ratios, and shared paths."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RANDOM_SEED = 42
TEST_SIZE = 0.2  # used by v1 and v2 (simple train/test split)

# v3 uses a three-way split instead of TEST_SIZE above:
V3_TEST_SIZE = 0.15
V3_VAL_SIZE = 0.15 / 0.85  # yields 70/15/15 of the original whole

DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEEDBACK_DATA_DIR = DATA_DIR / "feedback"
