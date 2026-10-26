"""Download a dataset's raw files into data/raw/<dataset>/ and verify their checksums.

    uv run python scripts/download.py --dataset beauty
    uv run python scripts/download.py --dataset ml-1m

Files that are already present and valid are skipped. This is a one-time setup
step, not a run: training and evaluation only read the local files.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from generative_recsys.data.download import DATASETS, DownloadError, download_dataset
from generative_recsys.utils.paths import RAW_DATA_DIR


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="raw data root; files go to <data-dir>/<dataset>/ (default: data/raw)",
    )
    parser.add_argument(
        "--force", action="store_true", help="download every file again, even if present"
    )
    args = parser.parse_args(argv)
    try:
        download_dataset(args.dataset, args.data_dir, force=args.force)
    except DownloadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
