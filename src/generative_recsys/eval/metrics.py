"""Full-ranking accuracy metrics for one held-out target per user.

Each metric depends only on the target's rank ``r`` among all items (1 is best):

    Recall@K   1               if r <= K, else 0
    NDCG@K     1 / log2(r + 1) if r <= K, else 0
    MRR@K      1 / r           if r <= K, else 0

With a single relevant item, Recall@K is the same as Hit Rate@K, and NDCG needs no
normalising: the ideal ranking puts the target first, which scores 1.

**Ties are averaged over every order they could be broken in.** If ``g`` items score
higher than the target and ``t`` others score the same, the target's rank is equally
likely to be any of ``g + 1 .. g + t + 1``, and each metric is its mean over those ranks
(McSherry & Najork, ECIR 2008). For scores ``[0.9, 0.5, 0.5, 0.5, 0.2, 0.1]`` with
target item 2, the rank is 2, 3 or 4, so Recall@3 is 2/3. Breaking ties in the target's
favour would instead give a model that scores every item the same a perfect 1.0, and
``torch.topk``'s tie order can differ between CPU, MPS and CUDA.

The rank comes from counting, not sorting, so the score matrix never leaves its device.
"""

from collections.abc import Sequence

import torch

DEFAULT_KS = (5, 10)
METRICS = ("recall", "ndcg", "mrr")


def accuracy_metrics(
    scores: torch.Tensor, targets: torch.Tensor, ks: Sequence[int] = DEFAULT_KS
) -> dict[str, torch.Tensor]:
    """Each user's Recall@K, NDCG@K and MRR@K, keyed ``"recall@5"``, ``"ndcg@10"``, ….

    ``scores`` is ``[B, n_items]`` (column j is item j) and ``targets`` is ``[B]`` item IDs.
    Values are float32 ``[B]`` tensors on the scores' device, one per user, left for the
    caller to aggregate. Scores may be ±inf (masked or never generated), but not NaN.
    """
    _check_inputs(scores, targets, ks)
    target_scores = scores.gather(1, targets.long()[:, None])
    n_greater = (scores > target_scores).sum(1)
    n_tied = (scores == target_scores).sum(1) - 1  # minus the target itself

    metrics = {}
    for k in ks:
        ranks = torch.arange(1, k + 1, device=scores.device, dtype=torch.float32)
        gains = {
            "recall": torch.ones_like(ranks),
            "ndcg": 1 / torch.log2(ranks + 1),
            "mrr": 1 / ranks,
        }
        # The mean gain over ranks lo + 1 .. hi, where gains past K are 0, is
        # (cumulative[hi] - cumulative[lo]) / (t + 1), with both clamped to K.
        lo = n_greater.clamp(max=k)
        hi = (n_greater + n_tied + 1).clamp(max=k)
        for name in METRICS:
            cumulative = torch.cat([ranks.new_zeros(1), gains[name].cumsum(0)])
            metrics[f"{name}@{k}"] = (cumulative[hi] - cumulative[lo]) / (n_tied + 1)
    return metrics


def _check_inputs(scores: torch.Tensor, targets: torch.Tensor, ks: Sequence[int]) -> None:
    if scores.ndim != 2 or not scores.is_floating_point():
        raise ValueError(
            f"scores must be a 2-D float tensor, got {scores.dtype} {tuple(scores.shape)}"
        )
    if (
        targets.shape != scores.shape[:1]
        or targets.is_floating_point()
        or targets.dtype == torch.bool
    ):
        raise ValueError(
            f"targets must be integer item IDs of shape [{scores.shape[0]}], "
            f"got {targets.dtype} {tuple(targets.shape)}"
        )
    if targets.device != scores.device:
        raise ValueError(f"targets are on {targets.device} but scores are on {scores.device}")
    if not ks or any(k < 1 for k in ks):
        raise ValueError(f"ks must be positive, got {ks}")
    # One combined check, so a GPU or MPS batch costs a single sync. A NaN would fail
    # every comparison, ranking a NaN target first; an out-of-range target would crash
    # CUDA with a device-side assert.
    n_items = scores.shape[1]
    if torch.isnan(scores).any() | (targets < 0).any() | (targets >= n_items).any():
        if torch.isnan(scores).any():
            raise ValueError("scores contain NaN")
        raise ValueError(f"targets must be item IDs in 0..{n_items - 1}")
