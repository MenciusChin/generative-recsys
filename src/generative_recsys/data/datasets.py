"""Torch datasets of ``{"user_id", "history", "target"}`` examples over a leave-one-out split.

- ``eval_dataset(split, "valid" | "test", max_len)``: one example per user.
- ``train_dataset(split, max_len)``: one example per training position. A training
  history ``[a, b, c, d]`` yields ``[a] → b``, ``[a, b] → c`` and ``[a, b, c] → d``,
  which any next-item model can use. (SASRec's cheaper whole-sequence form, valid only
  for causally masked models, comes in Phase 3.)

Histories keep the most recent ``max_len`` items and are left-padded with
``split.pad_id`` (``n_items``), so the latest item is always at index -1:

    max_len = 4, history [3, 7, 2]        → [PAD, 3, 7, 2]
    max_len = 4, history [9, 1, 4, 8, 6]  → [1, 4, 8, 6]

``max_len`` has no default: it's a model setting, not a data one.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from generative_recsys.data.split import EvalSplit, Split


class SequenceDataset(Dataset):
    """Examples built on request from ``(user, end)`` pairs.

    ``end`` indexes ``split.items``: the target is ``items[end]`` and the history is the
    user's items before it. Storing only these pairs keeps ML-1M's ~1M training
    examples to a few MB instead of one padded array each.
    """

    def __init__(self, split: Split, users: np.ndarray, ends: np.ndarray, max_len: int):
        if max_len < 1:
            raise ValueError(f"max_len must be at least 1, got {max_len}")
        self.split = split
        self.users = np.asarray(users, dtype=np.int64)
        self.ends = np.asarray(ends, dtype=np.int64)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user, end = self.users[index], self.ends[index]
        start = max(self.split.offsets[user], end - self.max_len)
        recent = self.split.items[start:end]
        history = np.full(self.max_len, self.split.pad_id, dtype=np.int64)
        history[self.max_len - len(recent) :] = recent
        return {
            "user_id": torch.tensor(user),
            "history": torch.from_numpy(history),
            "target": torch.tensor(self.split.items[end]),
        }


def eval_dataset(split: Split, which: EvalSplit, max_len: int) -> SequenceDataset:
    """One example per user, in user_id order: their ``which`` target and the history before it."""
    return SequenceDataset(split, np.arange(split.n_users), split.target_index(which), max_len)


def train_dataset(split: Split, max_len: int) -> SequenceDataset:
    """One example per training item after each user's first; never reaches the targets."""
    # A user with n interactions has n - 2 training items, giving n - 3 examples.
    per_user = split.lengths - 3
    users = np.repeat(np.arange(split.n_users), per_user)
    # Within each user, targets are training positions 1 .. n - 3.
    first = np.repeat(split.offsets[:-1] + 1, per_user)
    step = np.arange(len(users)) - np.repeat(np.cumsum(per_user) - per_user, per_user)
    return SequenceDataset(split, users, first + step, max_len)
