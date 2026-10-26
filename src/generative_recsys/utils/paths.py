"""Repository paths."""

from pathlib import Path

# Assumes the editable (src-layout) install this repo uses.
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"  # gitignored; filled by scripts/download.py
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"  # gitignored; filled by scripts/preprocess.py
