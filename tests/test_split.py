import json

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from generative_recsys.data.datasets import eval_dataset, train_dataset
from generative_recsys.data.preprocess import build
from generative_recsys.data.split import Split, load_split

# Users with 3, 6 and 10 interactions, built by preprocessing so the layout is the real one.
# Items repeat across users; same-day timestamps exercise the tie-break.
HISTORIES = {
    "u0": ["x", "y", "z"],
    "u1": ["a", "b", "c", "d", "e", "f"],
    "u2": ["j", "i", "h", "g", "f", "e", "d", "c", "b", "a"],
}


@pytest.fixture
def processed():
    rows = [
        {"user": user, "item": item, "timestamp": 100 * (pos // 2), "rating": 5.0}
        for user, items in HISTORIES.items()
        for pos, item in enumerate(items)
    ]
    return build(pd.DataFrame(rows), pd.DataFrame(columns=["item"]), k=1)


@pytest.fixture
def split(processed):
    stats = processed.stats
    return Split.from_interactions(processed.interactions, stats["n_users"], stats["n_items"])


def sequences(split):
    return [split.items[split.offsets[u] : split.offsets[u + 1]] for u in range(split.n_users)]


def examples(dataset):
    return [dataset[i] for i in range(len(dataset))]


def unpadded(history, pad_id):
    return history[history != pad_id].tolist()


def test_targets_are_the_last_two_items(split):
    for user, seq in enumerate(sequences(split)):
        assert split.valid_targets[user] == seq[-2]
        assert split.test_targets[user] == seq[-1]


def test_splits_put_back_together_give_each_full_sequence(split):
    train = split.train.groupby("user_id")["item_id"].apply(list)
    for user, seq in enumerate(sequences(split)):
        rebuilt = [*train[user], split.valid_targets[user], split.test_targets[user]]
        assert rebuilt == seq.tolist()


def test_no_target_appears_in_that_users_training_data(split):
    # Comparing item IDs is enough: preprocessing keeps one interaction per (user, item).
    for user, items in split.train.groupby("user_id")["item_id"]:
        assert split.valid_targets[user] not in set(items)
        assert split.test_targets[user] not in set(items)


def test_training_examples_never_reach_the_targets(split):
    targets = {u: {split.valid_targets[u], split.test_targets[u]} for u in range(split.n_users)}
    for example in examples(train_dataset(split, max_len=20)):
        user = int(example["user_id"])
        seen = {*unpadded(example["history"], split.pad_id), int(example["target"])}
        assert not seen & targets[user]


def test_training_examples_are_every_prefix(split):
    # u1's training items are a, b, c, d, so its examples are [a]→b, [a, b]→c, [a, b, c]→d.
    dataset = train_dataset(split, max_len=20)
    user = 1
    got = [
        (unpadded(e["history"], split.pad_id), int(e["target"]))
        for e in examples(dataset)
        if int(e["user_id"]) == user
    ]
    seq = sequences(split)[user].tolist()
    assert got == [(seq[:t], seq[t]) for t in range(1, len(seq) - 2)]
    assert len(dataset) == sum(len(s) - 3 for s in sequences(split))  # u0 has none


@pytest.mark.parametrize("which", ["valid", "test"])
def test_each_user_appears_once_per_eval_split(split, which):
    users = [int(e["user_id"]) for e in examples(eval_dataset(split, which, max_len=20))]
    assert users == list(range(split.n_users))


def test_test_history_is_training_history_plus_validation_item(split):
    for user in range(split.n_users):
        valid, test = split.history(user, "valid"), split.history(user, "test")
        assert test.tolist() == [*valid.tolist(), split.valid_targets[user]]


def test_truncation_keeps_the_most_recent_items_and_pads_on_the_left(split):
    dataset = eval_dataset(split, "test", max_len=4)
    full = [split.history(u, "test").tolist() for u in range(split.n_users)]

    assert dataset[0]["history"].tolist() == [split.pad_id, split.pad_id, *full[0]]
    assert dataset[2]["history"].tolist() == full[2][-4:]  # 9 items cut to the last 4


def test_padding_is_n_items_and_only_on_the_left(split):
    for dataset in (train_dataset(split, max_len=7), eval_dataset(split, "test", max_len=7)):
        for example in examples(dataset):
            history = example["history"]
            assert len(history) == 7
            is_pad = (history == split.pad_id).tolist()
            assert is_pad == sorted(is_pad, reverse=True)  # all pads before all items
            assert not all(is_pad)  # never an empty history
            assert 0 <= int(example["target"]) < split.n_items


def test_dataloader_stacks_examples_into_batches(split):
    batch = next(iter(DataLoader(train_dataset(split, max_len=5), batch_size=4)))
    assert batch["history"].shape == (4, 5)
    assert batch["user_id"].shape == batch["target"].shape == (4,)


def test_load_split_reads_the_processed_folder(tmp_path, processed, split):
    folder = tmp_path / "toy"
    folder.mkdir()
    processed.interactions.to_parquet(folder / "interactions.parquet", index=False)
    (folder / "stats.json").write_text(json.dumps(processed.stats))

    loaded = load_split("toy", tmp_path)
    assert np.array_equal(loaded.items, split.items)
    assert np.array_equal(loaded.offsets, split.offsets)
    pd.testing.assert_frame_equal(loaded.train, split.train)


def test_load_split_points_to_the_preprocess_script(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"scripts/preprocess\.py --dataset toy"):
        load_split("toy", tmp_path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda df: df.iloc[::-1], "sorted by user_id"),
        (lambda df: df.assign(position=df["position"].iloc[::-1].to_numpy()), "position"),
        (lambda df: df[df["user_id"] != 0].assign(user_id=lambda d: d["user_id"] - 1), "at least"),
    ],
)
def test_rejects_tables_not_in_the_processed_layout(processed, change, message):
    stats = processed.stats
    with pytest.raises(ValueError, match=message):
        Split.from_interactions(change(processed.interactions), stats["n_users"], stats["n_items"])


def test_max_len_must_be_positive(split):
    with pytest.raises(ValueError, match="max_len"):
        train_dataset(split, max_len=0)
