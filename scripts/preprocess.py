"""Build data/processed/<dataset>/ from the raw files and print the dataset's stats.

    uv run python scripts/preprocess.py --dataset beauty
    uv run python scripts/preprocess.py --dataset ml-1m

The raw files must already be downloaded (scripts/download.py). The same raw files
always produce byte-identical output. Fails if the result doesn't have the expected size.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from generative_recsys.data.download import DownloadError
from generative_recsys.data.preprocess import READERS, StatsMismatchError, preprocess
from generative_recsys.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=sorted(READERS))
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR, help="default: data/raw")
    parser.add_argument(
        "--out-dir", type=Path, default=PROCESSED_DATA_DIR, help="default: data/processed"
    )
    args = parser.parse_args(argv)
    try:
        stats = preprocess(args.dataset, args.raw_dir, args.out_dir)
    except (FileNotFoundError, DownloadError, StatsMismatchError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    lengths = stats["history_length"]
    print(f"{args.dataset} → {args.out_dir / args.dataset}")
    print(f"  users         {stats['n_users']:>10,}")
    print(f"  items         {stats['n_items']:>10,}")
    print(f"  interactions  {stats['n_interactions']:>10,}")
    print(f"  density       {stats['density']:>10.4%}")
    print(
        f"  history       median {lengths['median']:g}, p90 {lengths['p90']:g}, "
        f"max {lengths['max']} (min {lengths['min']})"
    )
    print(f"  {stats['kcore']['k']}-core rounds {stats['kcore']['rounds']}")
    print(f"  last two interactions tied: {stats['last_two_share_timestamp']:.1%} of users")
    return 0


if __name__ == "__main__":
    sys.exit(main())
