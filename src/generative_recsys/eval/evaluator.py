"""The shared evaluator: every model is scored by this one code path.

    result = evaluate(model, split, "valid", device=device)
    result.metrics["valid/ndcg@10"]  # model selection
    result.per_user["ndcg@10"]  # float32 [n_users], indexed by user_id

For each batch of users, in user_id order:

1. ``model.score(user_ids, histories)`` gives ``[B, n_items]`` scores. Histories are
   truncated to ``model.max_len`` and left-padded, as in ``eval_dataset``.
2. **Every item in the user's full history is masked to -inf**, not just the items in the
   truncated tensor. Preprocessing keeps one interaction per (user, item), so a seen item
   can never be the target: masking only removes certain misses. For the test split this
   also masks the validation item, which is part of the test history.
3. Accuracy metrics (``metrics.py``) and catalog metrics (``catalog.py``) are computed on
   the masked scores, without leaving the device.

Metrics are reported for all users and for two slices: ``same_time``, users whose target
shares a timestamp with their last history item (so only preprocessing's tie-break put it
second), and ``later_time``, everyone else. A model that gains mostly on ``same_time`` may
be learning the tie-break rather than the users.
"""

from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import DataLoader

from generative_recsys.data.datasets import eval_dataset
from generative_recsys.data.split import EvalSplit, Split
from generative_recsys.eval.catalog import (
    inclusion_probabilities,
    popularity_percentiles,
    tail_items,
)
from generative_recsys.eval.metrics import DEFAULT_KS, accuracy_metrics


class Recommender(Protocol):
    """What a model provides to be evaluated."""

    max_len: int  # how many of the most recent history items ``score`` reads

    def score(self, user_ids: torch.Tensor, histories: torch.Tensor) -> torch.Tensor:
        """``[B]`` user IDs and ``[B, max_len]`` padded histories → ``[B, n_items]`` scores."""
        ...


@dataclass(frozen=True)
class EvalResult:
    metrics: dict[str, float]  # "test/recall@10", "test/same_time/recall@10", …
    per_user: dict[str, np.ndarray]  # "recall@10", "popularity_bias@10", …: [n_users]


def evaluate(
    model: Recommender,
    split: Split,
    which: EvalSplit,
    *,
    device: torch.device | str,
    batch_size: int = 1024,
    ks: Sequence[int] = DEFAULT_KS,
) -> EvalResult:
    """Score every user's ``which`` target and return the metrics.

    The model must already be on ``device``. An ``nn.Module`` is put in eval mode for the
    duration and then restored. Results don't depend on ``batch_size``, beyond float32
    rounding.
    """
    device = torch.device(device)
    _check_histories(split, which, max(ks))
    counts = np.bincount(split.train["item_id"], minlength=split.n_items)
    percentile = torch.tensor(popularity_percentiles(counts), dtype=torch.float32, device=device)
    is_tail = torch.tensor(tail_items(counts), dtype=torch.float32, device=device)

    loader = DataLoader(eval_dataset(split, which, model.max_len), batch_size=batch_size)
    with _eval_mode(model), torch.inference_mode():
        per_user = defaultdict(list)
        log_never_in = {k: torch.zeros(split.n_items, device=device) for k in ks}
        for batch in loader:
            users = batch["user_id"]
            scores = model.score(users.to(device), batch["history"].to(device))
            if scores.shape != (len(users), split.n_items):
                raise ValueError(
                    f"score() must return [{len(users)}, {split.n_items}], got {list(scores.shape)}"
                )
            seen = _history_mask(split, which, users.numpy(), device)
            scores = scores.masked_fill(seen, -torch.inf)  # a new tensor: scores may be a view

            for name, values in accuracy_metrics(scores, batch["target"].to(device), ks).items():
                per_user[name].append(values)
            for k in ks:
                p = inclusion_probabilities(scores, k, excluded=seen)
                per_user[f"popularity_bias@{k}"].append(p @ percentile / k)
                per_user[f"tail_share@{k}"].append(p @ is_tail / k)
                # Coverage needs P(item in nobody's top k) = product over users of (1 - p).
                log_never_in[k] += torch.log1p(-p).sum(0)

    values = {name: torch.cat(parts).cpu().numpy() for name, parts in per_user.items()}
    metrics = {f"{which}/{name}": _mean(v) for name, v in values.items()}
    for k in ks:
        metrics[f"{which}/coverage@{k}"] = (1 - log_never_in[k].exp()).mean().item()

    same_time = split.same_time(which)
    metrics[f"{which}/same_time/share"] = _mean(same_time)
    for slice_name, users in {"same_time": same_time, "later_time": ~same_time}.items():
        if users.any():
            for name, v in values.items():
                metrics[f"{which}/{slice_name}/{name}"] = _mean(v[users])
    return EvalResult(metrics, values)


def _history_items(
    split: Split, which: EvalSplit, users: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Every (row, item) pair in ``users``' full ``which`` histories; row indexes ``users``."""
    starts = split.offsets[users]
    lengths = split.target_index(which)[users] - starts
    rows = np.repeat(np.arange(len(users)), lengths)
    step = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return rows, split.items[np.repeat(starts, lengths) + step]


def _history_mask(
    split: Split, which: EvalSplit, users: np.ndarray, device: torch.device
) -> torch.Tensor:
    rows, items = _history_items(split, which, users)
    mask = torch.zeros(len(users), split.n_items, dtype=torch.bool, device=device)
    mask[torch.from_numpy(rows).to(device), torch.from_numpy(items).to(device)] = True
    return mask


def _check_histories(split: Split, which: EvalSplit, k: int) -> None:
    """Fail before scoring if masking could hide a target or leave fewer than k items."""
    rows, items = _history_items(split, which, np.arange(split.n_users))
    targets = split.items[split.target_index(which)]
    if (items == targets[rows]).any():
        raise ValueError("a target also appears in its own history, so masking would hide it")
    longest = (split.target_index(which) - split.offsets[:-1]).max()
    if longest > split.n_items - k:
        raise ValueError(f"a history of {longest} items leaves fewer than {k} to recommend")


@contextmanager
def _eval_mode(model: Recommender) -> Iterator[None]:
    """Turn off dropout and the like while evaluating, then restore the previous mode."""
    if not isinstance(model, torch.nn.Module):
        yield
        return
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values, dtype=np.float64))
