import itertools
import math

import numpy as np
import pytest
import torch

from generative_recsys.eval.catalog import (
    inclusion_probabilities,
    popularity_percentiles,
    tail_items,
)


def probabilities(scores, k, excluded=None):
    excluded = None if excluded is None else torch.tensor([excluded])
    return inclusion_probabilities(torch.tensor([scores]), k, excluded)[0].tolist()


def test_tied_items_share_the_slots_left():
    # Item 0 is certainly in; one slot is left for three tied items.
    assert probabilities([0.9, 0.5, 0.5, 0.5, 0.2], 2) == pytest.approx([1, 1 / 3, 1 / 3, 1 / 3, 0])


def test_without_ties_the_top_k_is_certain():
    assert probabilities([0.1, 0.9, 0.5, 0.7], 2) == [0, 1, 0, 1]


def test_excluded_items_never_take_a_slot_even_at_minus_infinity():
    # One scored item and three at -inf; item 1 is masked, so items 2 and 3 share the slot.
    inf = math.inf
    assert probabilities([3.0, -inf, -inf, -inf], 2, [False, True, False, False]) == pytest.approx(
        [1, 0, 0.5, 0.5]
    )


def brute_force(scores, k):
    """Each item's share of all tie-breaking orders in which it lands in the top k."""
    totals = [0.0] * len(scores)
    orders = list(itertools.permutations(range(len(scores))))
    for order in orders:
        position = {item: i for i, item in enumerate(order)}
        ranking = sorted(range(len(scores)), key=lambda item: (-scores[item], position[item]))
        for item in ranking[:k]:
            totals[item] += 1 / len(orders)
    return totals


def test_matches_brute_force_on_random_tied_scores():
    generator = torch.Generator().manual_seed(0)
    scores = torch.randint(0, 3, (20, 6), generator=generator).float()
    result = inclusion_probabilities(scores, 3)
    for user in range(20):
        assert result[user].tolist() == pytest.approx(brute_force(scores[user].tolist(), 3))
    assert result.sum(1).tolist() == pytest.approx([3.0] * 20)  # always exactly k slots


def test_popularity_percentiles_share_the_middle_of_tied_ranges():
    assert popularity_percentiles(np.array([0, 0, 5, 9])).tolist() == [0.25, 0.25, 0.625, 0.875]


def test_a_random_recommender_has_popularity_bias_one_half():
    counts = np.random.default_rng(0).integers(0, 20, size=1000)  # plenty of ties
    assert popularity_percentiles(counts).mean() == pytest.approx(0.5)


def test_the_head_holds_the_most_popular_fifth_of_interactions():
    assert tail_items(np.array([50, 30, 10, 10])).tolist() == [False, True, True, True]


def test_items_tied_with_the_last_head_item_are_in_the_head():
    # 40 alone reaches 20% of 100, and the other 40 ties with it: no item ID decides.
    assert tail_items(np.array([10, 40, 10, 40])).tolist() == [True, False, True, False]
