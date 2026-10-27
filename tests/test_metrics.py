import itertools
import math

import pytest
import torch

from generative_recsys.eval.metrics import accuracy_metrics

DEVICES = [
    "cpu",
    *(["mps"] if torch.backends.mps.is_available() else []),
    *(["cuda"] if torch.cuda.is_available() else []),
]


def metrics_for(scores, target, k):
    """One user's metrics at one K, as plain floats."""
    result = accuracy_metrics(torch.tensor([scores]), torch.tensor([target]), ks=(k,))
    return {name.split("@")[0]: value.item() for name, value in result.items()}


def expected(rank, k):
    """The metrics for a target at ``rank``, straight from their definitions."""
    if rank > k:
        return {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0}
    return {"recall": 1.0, "ndcg": 1 / math.log2(rank + 1), "mrr": 1 / rank}


def assert_metrics(actual, wanted):
    assert actual == pytest.approx(wanted)


SCORES = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]  # no ties: item j has rank j + 1


@pytest.mark.parametrize("k", [5, 10])
def test_target_ranked_first_scores_one(k):
    assert_metrics(metrics_for(SCORES, 0, k), {"recall": 1.0, "ndcg": 1.0, "mrr": 1.0})


def test_target_ranked_third():
    # Two items score higher, so r = 3: NDCG = 1 / log2(4) = 0.5, MRR = 1/3.
    assert_metrics(metrics_for(SCORES, 2, 5), {"recall": 1.0, "ndcg": 0.5, "mrr": 1 / 3})


def test_target_ranked_exactly_k_still_counts():
    assert_metrics(metrics_for(SCORES, 4, 5), expected(5, 5))


def test_target_ranked_k_plus_one_scores_zero():
    assert_metrics(metrics_for(SCORES, 5, 5), {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0})
    assert_metrics(metrics_for(SCORES, 2, 2), {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0})


def test_ties_average_over_the_ranks_the_target_could_take():
    # One item above the target and two tied with it: rank 2, 3 or 4, each with
    # probability 1/3. At K = 3, rank 4 contributes 0.
    actual = metrics_for([0.9, 0.5, 0.5, 0.5, 0.2, 0.1], 2, 3)
    wanted = {
        "recall": 2 / 3,
        "ndcg": (1 / math.log2(3) + 1 / math.log2(4)) / 3,  # 0.377, not NDCG at rank 3 (0.5)
        "mrr": (1 / 2 + 1 / 3) / 3,
    }
    assert_metrics(actual, wanted)


def test_ties_entirely_inside_the_top_k_change_nothing():
    # Ranks 2..4 are all within K = 5, so Recall is exactly 1.
    assert metrics_for([0.9, 0.5, 0.5, 0.5, 0.2, 0.1], 2, 5)["recall"] == pytest.approx(1.0)


def test_a_model_that_scores_every_item_the_same_is_random():
    n_items = 1000
    actual = metrics_for([0.0] * n_items, 123, 10)
    # Every rank 1..n_items is equally likely: the mean of each metric over all ranks.
    wanted = {
        name: sum(expected(r, 10)[name] for r in range(1, n_items + 1)) / n_items
        for name in expected(1, 10)
    }
    assert_metrics(actual, wanted)
    assert actual["recall"] == pytest.approx(10 / n_items)


def test_a_target_left_out_of_a_short_beam_gets_almost_nothing():
    # Phase 5 preview: beam search scores 5 items and leaves the rest at -inf. The target
    # isn't among the 5, so it ties with the other 994 unscored items: ranks 6..1000, of
    # which 6..10 are hits at K = 10. Breaking ties in its favour would give rank 6, a hit.
    scores = torch.full((1000,), -math.inf)
    scores[:5] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    actual = metrics_for(scores.tolist(), 500, 10)
    assert actual["recall"] == pytest.approx(5 / 995)


def brute_force(scores, target, k):
    """Each metric averaged over every tie-breaking order: all permutations of the items."""
    totals = dict.fromkeys(expected(1, k), 0.0)
    orders = list(itertools.permutations(range(len(scores))))
    for order in orders:
        position = {item: i for i, item in enumerate(order)}
        ranking = sorted(range(len(scores)), key=lambda item: (-scores[item], position[item]))
        for name, value in expected(ranking.index(target) + 1, k).items():
            totals[name] += value / len(orders)
    return totals


def test_matches_brute_force_on_random_tied_scores():
    generator = torch.Generator().manual_seed(0)
    # 6 items scored 0..2 have plenty of ties; K = 3 cuts through many of them.
    scores = torch.randint(0, 3, (20, 6), generator=generator).float()
    targets = torch.randint(0, 6, (20,), generator=generator)
    result = accuracy_metrics(scores, targets, ks=(3,))
    for user in range(20):
        wanted = brute_force(scores[user].tolist(), targets[user].item(), 3)
        assert_metrics({name: result[f"{name}@3"][user].item() for name in wanted}, wanted)


def test_returns_one_float32_value_per_user_per_metric_and_k():
    scores = torch.rand(7, 50)
    result = accuracy_metrics(scores, torch.arange(7))
    assert list(result) == ["recall@5", "ndcg@5", "mrr@5", "recall@10", "ndcg@10", "mrr@10"]
    for value in result.values():
        assert value.shape == (7,)
        assert value.dtype == torch.float32


def test_each_user_is_ranked_against_their_own_scores():
    scores = torch.tensor([SCORES, SCORES[::-1]])
    result = accuracy_metrics(scores, torch.tensor([0, 0]), ks=(5,))
    # User 0's target is first; user 1's is last (rank 6).
    assert result["recall@5"].tolist() == [1.0, 0.0]


@pytest.mark.parametrize("device", DEVICES)
def test_runs_on_the_scores_device_and_matches_cpu(device):
    generator = torch.Generator().manual_seed(0)
    scores = torch.randint(0, 20, (64, 300), generator=generator).float()  # ties included
    targets = torch.randint(0, 300, (64,), generator=generator)
    on_cpu = accuracy_metrics(scores, targets)
    on_device = accuracy_metrics(scores.to(device), targets.to(device))
    for name, value in on_device.items():
        assert value.device.type == device
        torch.testing.assert_close(value.cpu(), on_cpu[name])


@pytest.mark.parametrize("nan_at", [0, 3])  # the target's own score, and another item's
def test_nan_scores_raise(nan_at):
    scores = torch.tensor([SCORES])
    scores[0, nan_at] = math.nan
    with pytest.raises(ValueError, match="NaN"):
        accuracy_metrics(scores, torch.tensor([0]))


@pytest.mark.parametrize("target", [-1, 6])
def test_out_of_range_targets_raise(target):
    with pytest.raises(ValueError, match=r"0\.\.5"):
        accuracy_metrics(torch.tensor([SCORES]), torch.tensor([target]))


@pytest.mark.parametrize(
    ("scores", "targets"),
    [
        (torch.tensor(SCORES), torch.tensor([0])),  # scores not 2-D
        (torch.tensor([[1, 2, 3]]), torch.tensor([0])),  # integer scores
        (torch.tensor([SCORES]), torch.tensor([0, 1])),  # one target per user
        (torch.tensor([SCORES]), torch.tensor([0.0])),  # float targets
    ],
)
def test_malformed_inputs_raise(scores, targets):
    with pytest.raises(ValueError):
        accuracy_metrics(scores, targets)


def test_non_positive_k_raises():
    with pytest.raises(ValueError, match="ks"):
        accuracy_metrics(torch.tensor([SCORES]), torch.tensor([0]), ks=(0,))
