"""Leave-one-out split: per user, last item → test, second-to-last → validation, rest → training.

For a user whose history is ``[a, b, c, d, e, f]``:

    training      learns from [a, b, c, d]
    validation    history [a, b, c, d]     target e
    test          history [a, b, c, d, e]  target f

The test history includes the validation item: by then it has happened. Training never
sees either target. The split is computed in memory from
``data/processed/<dataset>/interactions.parquet`` (it takes milliseconds) and never
written to disk, so it can't go stale when preprocessing changes. It relies on
``position`` and row order only and never re-sorts by timestamp, since timestamp
ties were already broken in preprocessing.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from generative_recsys.utils.paths import PROCESSED_DATA_DIR

EvalSplit = Literal["valid", "test"]
MIN_INTERACTIONS = 3  # at least one training item, plus the two targets


@dataclass(frozen=True)
class Split:
    """Every user's sequence, stored flat.

    User ``u``'s items are ``items[offsets[u]:offsets[u + 1]]``: the standard way to hold
    lists of different lengths without one Python list per user.
    """

    n_users: int
    n_items: int
    items: np.ndarray  # int64 [n_interactions], users' sequences concatenated in user_id order
    offsets: np.ndarray  # int64 [n_users + 1]
    train: pd.DataFrame  # training interactions: user_id, item_id, timestamp

    @property
    def pad_id(self) -> int:
        """The padding ID: one past the last item, so it's never a real item or a score column."""
        return self.n_items

    @property
    def lengths(self) -> np.ndarray:
        """Each user's full sequence length (training + validation + test)."""
        return np.diff(self.offsets)

    @property
    def valid_targets(self) -> np.ndarray:
        """int64 [n_users]: each user's validation target, indexed by user_id."""
        return self.items[self.target_index("valid")]

    @property
    def test_targets(self) -> np.ndarray:
        return self.items[self.target_index("test")]

    def target_index(self, split: EvalSplit) -> np.ndarray:
        """Index into ``items`` of each user's ``split`` target; its history is what precedes it."""
        return self.offsets[1:] - {"valid": 2, "test": 1}[split]

    def history(self, user_id: int, split: EvalSplit) -> np.ndarray:
        """The user's full (untruncated) history for ``split``."""
        return self.items[self.offsets[user_id] : self.target_index(split)[user_id]]

    @classmethod
    def from_interactions(cls, interactions: pd.DataFrame, n_users: int, n_items: int) -> "Split":
        """Build from a processed interactions table, checking that it's in the processed layout."""
        users = interactions["user_id"].to_numpy(np.int64)
        if (np.diff(users) < 0).any():
            raise ValueError("interactions must be sorted by user_id")
        counts = np.bincount(users, minlength=n_users)
        if len(counts) != n_users:
            raise ValueError(f"user_id must be in 0..{n_users - 1}")
        if (counts < MIN_INTERACTIONS).any():
            raise ValueError(f"every user needs at least {MIN_INTERACTIONS} interactions")
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

        # Row order is the chronological order; `position` must agree with it.
        position = np.arange(len(users)) - np.repeat(offsets[:-1], counts)
        if not np.array_equal(interactions["position"].to_numpy(), position):
            raise ValueError("position must run 0..n-1 within each user, in row order")
        items = interactions["item_id"].to_numpy(np.int64)
        if items.min() < 0 or items.max() >= n_items:
            raise ValueError(f"item_id must be in 0..{n_items - 1}")

        is_train = position < np.repeat(counts - 2, counts)
        train = interactions.loc[is_train, ["user_id", "item_id", "timestamp"]]
        return cls(n_users, n_items, items, offsets, train.reset_index(drop=True))


def load_split(dataset: str, processed_dir: str | Path = PROCESSED_DATA_DIR) -> Split:
    """Load ``processed_dir/<dataset>/`` (built by scripts/preprocess.py) and split it."""
    folder = Path(processed_dir) / dataset
    path = folder / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first: "
            f"uv run python scripts/preprocess.py --dataset {dataset}"
        )
    stats = json.loads((folder / "stats.json").read_text())
    columns = ["user_id", "item_id", "timestamp", "position"]
    interactions = pd.read_parquet(path, columns=columns)
    return Split.from_interactions(interactions, stats["n_users"], stats["n_items"])
