import gzip
import hashlib
import importlib.util
import json
import zipfile

import pandas as pd
import pytest

from generative_recsys.data.download import ChecksumError, RemoteFile
from generative_recsys.data.preprocess import (
    Expected,
    StatsMismatchError,
    build,
    kcore_filter,
    preprocess,
)
from generative_recsys.utils.paths import REPO_ROOT

# Raw fixtures: 6 users x 6 items, so every user and item has 6 interactions and survives
# the 5-core filter, plus one rare item ("Z"/99) that one user touched once and must be dropped.
USERS = ["U6", "U5", "U4", "U3", "U2", "U1"]  # deliberately unsorted
ITEMS = ["B6", "B5", "B4", "B3", "B2", "B1"]
DAY = 86_400


def registry_for(dataset, folder):
    """A registry whose pinned checksums are those of the fixture files."""
    files = []
    for path in sorted(folder.iterdir()):
        data = path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        files.append(RemoteFile(path.name, (path.as_uri(),), sha256, len(data)))
    return {dataset: tuple(files)}


@pytest.fixture
def beauty_raw(tmp_path):
    folder = tmp_path / "raw" / "beauty"
    folder.mkdir(parents=True)
    reviews = [
        # Every review on the same day: the order must come from the tie-break alone.
        {"reviewerID": u, "asin": i, "unixReviewTime": DAY, "overall": 5.0, "reviewText": "..."}
        for u in USERS
        for i in ITEMS
    ] + [{"reviewerID": "U1", "asin": "Z", "unixReviewTime": 0, "overall": 1.0}]
    with gzip.open(folder / "reviews_Beauty_5.json.gz", "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in reviews)

    meta = [
        {
            "asin": i,
            "title": f"Cleanser &amp; Toner {i}",
            "brand": "Acme",
            "price": 9.5,
            "categories": [["Beauty", "Skin Care"]],
            "related": {"also_bought": ["B2"]},
            "salesRank": {"Beauty": 7},
        }
        for i in ITEMS[:-1]
    ] + [
        {"asin": "B1", "title": "", "categories": [["Beauty"]]},  # sparse metadata
        {"asin": "UNUSED", "title": "never reviewed", "categories": [["Beauty"]]},
    ]
    with gzip.open(folder / "meta_Beauty.json.gz", "wt") as f:
        f.writelines(repr(m) + "\n" for m in meta)  # Python literals, as in the real file
    return tmp_path / "raw", registry_for("beauty", folder)


@pytest.fixture
def ml1m_raw(tmp_path):
    folder = tmp_path / "raw" / "ml-1m"
    folder.mkdir(parents=True)
    ratings = [f"{u}::{m}::4::{1000 + m}" for u in range(1, 7) for m in range(1, 7)]
    ratings.append("1::99::2::5000")
    movies = ["1::Café Society (1999)::Comedy|Drama"] + [
        f"{m}::Movie {m} (2000)::Action" for m in [*range(2, 7), 99]
    ]
    with zipfile.ZipFile(folder / "ml-1m.zip", "w") as z:
        z.writestr("ml-1m/ratings.dat", "\n".join(ratings) + "\n")
        z.writestr("ml-1m/movies.dat", ("\n".join(movies) + "\n").encode("latin-1"))
    return tmp_path / "raw", registry_for("ml-1m", folder)


def run(dataset, raw, out, **kwargs):
    raw_dir, registry = raw
    return preprocess(dataset, raw_dir, out, registry=registry, expected={}, **kwargs)


def read(out, dataset, table):
    return pd.read_parquet(out / dataset / f"{table}.parquet")


def test_beauty_end_to_end(tmp_path, beauty_raw):
    out = tmp_path / "processed"
    stats = run("beauty", beauty_raw, out)

    assert (stats["n_users"], stats["n_items"], stats["n_interactions"]) == (6, 6, 36)
    assert stats["kcore"]["rounds"] == 1
    assert stats["last_two_share_timestamp"] == 1.0
    assert sorted(p.name for p in (out / "beauty").iterdir()) == [
        "interactions.parquet",
        "items.parquet",
        "stats.json",
        "users.parquet",
    ]
    assert json.loads((out / "beauty" / "stats.json").read_text()) == stats

    interactions = read(out, "beauty", "interactions")
    assert list(interactions.columns) == ["user_id", "item_id", "timestamp", "rating", "position"]

    users, items = read(out, "beauty", "users"), read(out, "beauty", "items")
    assert users["raw_user_id"].tolist() == sorted(USERS)  # IDs follow sorted raw IDs
    assert items["raw_item_id"].tolist() == sorted(ITEMS)
    assert list(items.columns) == [
        "item_id",
        "raw_item_id",
        "title",
        "brand",
        "categories",
        "description",
        "price",
    ]  # no related / salesRank / imUrl
    assert items.loc[1, "title"] == "Cleanser & Toner B2"  # HTML entities decoded
    assert items.loc[0, ["title", "brand", "price"]].isna().all()  # B1: empty or missing
    assert items.loc[0, "categories"].tolist()[0].tolist() == ["Beauty"]


def test_ties_are_broken_by_raw_item_id(tmp_path, beauty_raw):
    out = tmp_path / "processed"
    run("beauty", beauty_raw, out)
    interactions = read(out, "beauty", "interactions")
    items = read(out, "beauty", "items").set_index("item_id")["raw_item_id"]

    # The raw file lists items B6..B1, all on the same day; the output must be B1..B6.
    for _, history in interactions.groupby("user_id"):
        assert items[history["item_id"]].tolist() == sorted(ITEMS)
        assert history["position"].tolist() == list(range(6))


def test_ml1m_end_to_end(tmp_path, ml1m_raw):
    out = tmp_path / "processed"
    stats = run("ml-1m", ml1m_raw, out)
    assert (stats["n_users"], stats["n_items"], stats["n_interactions"]) == (6, 6, 36)

    items = read(out, "ml-1m", "items")
    assert items["raw_item_id"].tolist() == [1, 2, 3, 4, 5, 6]  # numeric, and 99 is gone
    assert items.loc[0, "title"] == "Café Society (1999)"  # decoded as latin-1
    assert items.loc[0, "genres"].tolist() == ["Comedy", "Drama"]
    interactions = read(out, "ml-1m", "interactions")
    assert interactions.groupby("user_id")["timestamp"].is_monotonic_increasing.all()


@pytest.mark.parametrize("dataset", ["beauty", "ml-1m"])
def test_output_is_byte_identical_across_runs(tmp_path, beauty_raw, ml1m_raw, dataset):
    raw = {"beauty": beauty_raw, "ml-1m": ml1m_raw}[dataset]
    run(dataset, raw, tmp_path / "a")
    run(dataset, raw, tmp_path / "b")
    for path in sorted((tmp_path / "a" / dataset).iterdir()):
        assert path.read_bytes() == (tmp_path / "b" / dataset / path.name).read_bytes(), path.name


def test_rerun_replaces_previous_output(tmp_path, beauty_raw):
    out = tmp_path / "processed"
    run("beauty", beauty_raw, out)
    (out / "beauty" / "stale.txt").write_text("from an older version")
    run("beauty", beauty_raw, out)
    assert not (out / "beauty" / "stale.txt").exists()
    assert not (out / "beauty.tmp").exists()


def test_missing_raw_file_points_to_the_download_script(tmp_path, beauty_raw):
    raw_dir, _ = beauty_raw
    (raw_dir / "beauty" / "meta_Beauty.json.gz").unlink()
    with pytest.raises(FileNotFoundError, match=r"scripts/download\.py --dataset beauty"):
        run("beauty", beauty_raw, tmp_path / "processed")


def test_modified_raw_file_fails_the_checksum(tmp_path, beauty_raw):
    raw_dir, _ = beauty_raw
    path = raw_dir / "beauty" / "reviews_Beauty_5.json.gz"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ChecksumError):
        run("beauty", beauty_raw, tmp_path / "processed")


def test_unexpected_size_fails_before_writing(tmp_path, beauty_raw):
    raw_dir, registry = beauty_raw
    out = tmp_path / "processed"
    with pytest.raises(StatsMismatchError, match="n_users=6"):
        preprocess(
            "beauty", raw_dir, out, registry=registry, expected={"beauty": Expected(7, 6, 36)}
        )
    assert not out.exists()


def test_kcore_filter_repeats_until_stable():
    # With k=2: item z has one interaction, so it goes in round 1. That leaves user c with
    # only y, so c goes in round 2. A single pass would have kept c.
    df = pd.DataFrame(
        {"user": ["a", "a", "b", "b", "c", "c"], "item": ["x", "y", "x", "y", "y", "z"]}
    )
    filtered, rounds = kcore_filter(df, k=2)
    assert rounds == 2
    assert sorted(filtered["user"].unique()) == ["a", "b"]
    assert sorted(filtered["item"].unique()) == ["x", "y"]


def test_build_keeps_each_pairs_first_interaction():
    interactions = pd.DataFrame(
        {
            "user": ["a", "a", "a"],
            "item": ["x", "y", "x"],
            "timestamp": [30, 20, 10],
            "rating": [1.0, 2.0, 3.0],
        }
    )
    processed = build(interactions, pd.DataFrame(columns=["item"]), k=1)
    assert processed.stats["duplicates_removed"] == 1
    assert processed.interactions["timestamp"].tolist() == [10, 20]  # x at 10, then y at 20
    assert processed.interactions["rating"].tolist() == [3.0, 2.0]
    assert processed.stats["items_without_metadata"] == 2


def test_script_fails_cleanly_without_raw_files(tmp_path, capsys):
    path = REPO_ROOT / "scripts" / "preprocess.py"
    spec = importlib.util.spec_from_file_location("preprocess_script", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    argv = ["--dataset", "beauty", "--raw-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")]
    assert script.main(argv) == 1
    assert "scripts/download.py --dataset beauty" in capsys.readouterr().err
