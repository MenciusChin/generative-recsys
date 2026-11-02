"""Catalog-level metrics: which items the top-K lists reach, and how popular they are.

No top-K list is ever materialised, because which tied item takes the last slots is
arbitrary. Instead each item gets the probability that it is in the user's top K when
ties are broken in a random order, as #13 does for the target's rank:

    scores [0.9, 0.5, 0.5, 0.5, 0.2], K = 2  →  p = [1, 1/3, 1/3, 1/3, 0]

Item 0 is certainly in; one slot is left for three tied items. Every catalog metric is an
expectation over these probabilities, so it never depends on the device or on item IDs.

- **coverage@K**: the expected share of the catalog that is in at least one user's top K.
- **popularity_bias@K**: the mean popularity percentile of a user's top K. Random is 0.5.
- **tail_share@K**: the share of a user's top K outside the head (the most popular items,
  which together hold 20% of training interactions).

Popularity always comes from training interactions only, so targets never count.
"""

import numpy as np
import torch

HEAD_SHARE = 0.2


def inclusion_probabilities(
    scores: torch.Tensor, k: int, excluded: torch.Tensor | None = None
) -> torch.Tensor:
    """float32 ``[B, n_items]``: each item's chance of being in the user's top ``k``.

    Items scoring above the k-th highest score are certainly in. Items scoring exactly
    that share the slots that are left. ``excluded`` (bool, same shape) marks items that
    can't be recommended, such as masked history items: they never share those slots,
    even when the k-th highest score is -inf.
    """
    kth = torch.topk(scores, k, dim=1).values[:, -1:]  # the same whatever the tie order
    above = scores > kth
    tied = scores == kth
    if excluded is not None:
        tied &= ~excluded
    slots_left = k - above.sum(1, keepdim=True)
    n_tied = tied.sum(1, keepdim=True).clamp(min=1)  # 0 only if fewer than k items are left
    return above.float() + tied.float() * (slots_left / n_tied)


def popularity_percentiles(counts: np.ndarray) -> np.ndarray:
    """float64 ``[n_items]``: each item's popularity percentile in [0, 1].

    Items with the same count share the middle of their range: (number of items with a
    lower count + half the number with the same count) / n_items. For counts
    ``[0, 0, 5, 9]`` that's ``[0.25, 0.25, 0.625, 0.875]``, and a random recommender
    averages exactly 0.5.
    """
    ordered = np.sort(counts)
    lower = np.searchsorted(ordered, counts, side="left")
    same = np.searchsorted(ordered, counts, side="right") - lower
    return (lower + 0.5 * same) / len(counts)


def tail_items(counts: np.ndarray, head_share: float = HEAD_SHARE) -> np.ndarray:
    """bool ``[n_items]``: items outside the head.

    The head is the most popular items that together hold ``head_share`` of all
    interactions, plus every item tied with the last of them, so the split never depends
    on item IDs. For counts ``[50, 30, 10, 10]`` (100 in total) the head is the first item.
    """
    ordered = np.sort(counts)[::-1]
    last_head = np.searchsorted(np.cumsum(ordered), head_share * counts.sum())
    return counts < ordered[last_head]
