import pandas as pd
import pytest
import torch

from generative_recsys.data.split import Split
from generative_recsys.eval.catalog import popularity_percentiles, tail_items
from generative_recsys.eval.evaluator import evaluate

N_ITEMS = 20  # more than the largest default K
ACCELERATORS = [
    *(["mps"] if torch.backends.mps.is_available() else []),
    *(["cuda"] if torch.cuda.is_available() else []),
]

# (item, timestamp) per user. Test targets 3, 6, 5; validation targets 2, 5, 4.
SEQUENCES = [
    [(0, 1), (1, 2), (2, 3), (3, 3)],  # test target shares a timestamp with item 2
    [(4, 1), (5, 2), (6, 3)],
    [(7, 1), (0, 1), (1, 1), (4, 1), (5, 1)],  # everything on the same timestamp
]


def make_split(sequences=SEQUENCES, n_items=N_ITEMS):
    rows = [
        {"user_id": user, "item_id": item, "timestamp": timestamp, "position": position}
        for user, sequence in enumerate(sequences)
        for position, (item, timestamp) in enumerate(sequence)
    ]
    return Split.from_interactions(pd.DataFrame(rows), len(sequences), n_items)


@pytest.fixture
def split():
    return make_split()


class FixedScores:
    """Returns a fixed [n_users, n_items] table's rows, and records what it was given."""

    def __init__(self, table, max_len=10):
        self.table = torch.as_tensor(table, dtype=torch.float32)
        self.max_len = max_len
        self.histories = []

    def score(self, user_ids, histories):
        self.histories.append(histories.cpu())
        return self.table.to(user_ids.device)[user_ids]


class Popularity:
    """Every user gets the same row: a view, so writing into it would corrupt the counts."""

    max_len = 1

    def __init__(self, split):
        self.counts = torch.bincount(
            torch.tensor(split.train["item_id"].to_numpy()), minlength=split.n_items
        )

    def score(self, user_ids, histories):
        return self.counts.float().to(user_ids.device).expand(len(user_ids), -1)


def table_from(split, rows):
    """A score table where rows[u] maps item → score and every other item scores 0."""
    table = torch.zeros(split.n_users, split.n_items)
    for user, scores in enumerate(rows):
        for item, value in scores.items():
            table[user, item] = value
    return table


def history(split, user, which):
    return split.history(user, which).tolist()


def test_masking_uses_the_full_history_not_the_truncated_one(split):
    # max_len = 1 shows the model only one history item, but all of them are masked.
    # Each user's history items score 10 and the target 5, so only masking puts it first.
    rows = [
        {**dict.fromkeys(history(split, u, "test"), 10.0), int(split.test_targets[u]): 5.0}
        for u in range(split.n_users)
    ]
    model = FixedScores(table_from(split, rows), max_len=1)
    result = evaluate(model, split, "test", device="cpu", ks=(1,))
    assert result.per_user["recall@1"].tolist() == [1.0, 1.0, 1.0]
    assert all(h.shape[1] == 1 for h in model.histories)


def test_test_split_masks_the_validation_item(split):
    rows = [
        {int(split.valid_targets[u]): 10.0, int(split.test_targets[u]): 5.0}
        for u in range(split.n_users)
    ]
    result = evaluate(FixedScores(table_from(split, rows)), split, "test", device="cpu", ks=(1,))
    assert result.metrics["test/recall@1"] == 1.0


def test_validation_split_does_not_mask_the_test_item(split):
    # The test item comes after the validation target, so it's still a candidate.
    rows = [
        {int(split.test_targets[u]): 10.0, int(split.valid_targets[u]): 5.0}
        for u in range(split.n_users)
    ]
    result = evaluate(FixedScores(table_from(split, rows)), split, "valid", device="cpu", ks=(1,))
    assert result.metrics["valid/recall@1"] == 0.0


def test_scores_returned_as_a_view_are_not_written_into(split):
    model = Popularity(split)
    before = model.counts.clone()
    evaluate(model, split, "test", device="cpu")
    assert torch.equal(model.counts, before)


def test_expected_coverage_and_popularity_over_tied_scores(split):
    # Items 3 and 6 are in no test history. Every user scores both 1 and the rest 0, so
    # each lands in a user's top 1 with probability 1/2. Each is in nobody's top 1 with
    # probability (1/2)^3, so expected coverage is 2 * (1 - 1/8) / 8 items.
    rows = [{3: 1.0, 6: 1.0}] * split.n_users
    result = evaluate(FixedScores(table_from(split, rows)), split, "test", device="cpu", ks=(1,))
    assert result.metrics["test/coverage@1"] == pytest.approx(2 * (1 - 1 / 8) / N_ITEMS)

    counts = split.train["item_id"].value_counts().reindex(range(N_ITEMS), fill_value=0)
    percentile = popularity_percentiles(counts.to_numpy())
    tail = tail_items(counts.to_numpy())
    assert result.metrics["test/popularity_bias@1"] == pytest.approx(
        (percentile[3] + percentile[6]) / 2
    )
    assert result.metrics["test/tail_share@1"] == pytest.approx(
        (float(tail[3]) + float(tail[6])) / 2
    )


def test_slices_split_users_by_whether_the_target_shares_a_timestamp(split):
    # Only user 1's target comes strictly later. The model ranks it first, and ranks
    # item 6 (in neither of the others' histories) above the other users' targets.
    rows = [{6: 1.0}, {int(split.test_targets[1]): 1.0}, {6: 1.0}]
    result = evaluate(FixedScores(table_from(split, rows)), split, "test", device="cpu", ks=(1,))
    assert result.metrics["test/same_time/share"] == pytest.approx(2 / 3)
    assert result.metrics["test/later_time/recall@1"] == 1.0
    assert result.metrics["test/same_time/recall@1"] == 0.0
    assert result.metrics["test/recall@1"] == pytest.approx(1 / 3)


def test_returns_every_metric_for_all_users_and_both_slices(split):
    result = evaluate(Popularity(split), split, "valid", device="cpu")
    per_user = [
        f"{name}@{k}"
        for k in (5, 10)
        for name in ("recall", "ndcg", "mrr", "popularity_bias", "tail_share")
    ]
    assert sorted(result.per_user) == sorted(per_user)
    assert all(v.shape == (split.n_users,) for v in result.per_user.values())
    expected = {
        *(f"valid/{name}" for name in per_user),
        *(f"valid/{s}/{name}" for s in ("same_time", "later_time") for name in per_user),
        "valid/coverage@5",
        "valid/coverage@10",
        "valid/same_time/share",
    }
    assert set(result.metrics) == expected
    assert all(isinstance(v, float) for v in result.metrics.values())


def tied_split_and_scores():
    sequences = [
        [(i % N_ITEMS, 0), ((i + 1) % N_ITEMS, 1), ((i + 3) % N_ITEMS, 1)] for i in range(23)
    ]
    generator = torch.Generator().manual_seed(0)
    return make_split(sequences), torch.randint(0, 3, (23, N_ITEMS), generator=generator)


def test_results_do_not_depend_on_batch_size():
    split, table = tied_split_and_scores()
    one = evaluate(FixedScores(table), split, "test", device="cpu", batch_size=1, ks=(2, 3))
    many = evaluate(FixedScores(table), split, "test", device="cpu", batch_size=1024, ks=(2, 3))
    for name, values in one.per_user.items():  # equal up to float32 rounding in matmuls
        assert values.tolist() == pytest.approx(many.per_user[name].tolist())
    assert one.metrics == pytest.approx(many.metrics)


def test_modules_are_evaluated_in_eval_mode_and_restored(split):
    class Recording(torch.nn.Module):
        max_len = 2

        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.5)
            self.modes = []

        def score(self, user_ids, histories):
            self.modes.append(self.training)
            return torch.zeros(len(user_ids), N_ITEMS)

    model = Recording()
    model.train()
    evaluate(model, split, "test", device="cpu")
    assert model.modes == [False]
    assert model.training


def test_wrong_score_shape_raises(split):
    model = FixedScores(torch.zeros(split.n_users, N_ITEMS - 1))
    with pytest.raises(ValueError, match=r"\[3, 20\]"):
        evaluate(model, split, "test", device="cpu")


def test_a_target_in_its_own_history_raises():
    split = make_split([[(0, 1), (1, 2), (0, 3)]])  # preprocessing would have dropped the repeat
    with pytest.raises(ValueError, match="own history"):
        evaluate(FixedScores(torch.zeros(1, N_ITEMS)), split, "test", device="cpu")


def test_a_history_leaving_fewer_than_k_items_raises():
    split = make_split([[(0, 1), (1, 2), (2, 3), (3, 4)]], n_items=4)
    with pytest.raises(ValueError, match="fewer than 2"):
        evaluate(FixedScores(torch.zeros(1, 4)), split, "test", device="cpu", ks=(2,))


@pytest.mark.parametrize("device", ACCELERATORS)
def test_accelerators_match_cpu(device):
    split, table = tied_split_and_scores()
    on_cpu = evaluate(FixedScores(table), split, "test", device="cpu", ks=(2, 3))
    on_device = evaluate(FixedScores(table), split, "test", device=device, ks=(2, 3))
    assert on_device.metrics == pytest.approx(on_cpu.metrics)
