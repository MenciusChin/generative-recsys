"""Turn a dataset's raw files into the processed layout every model reads.

Output, in ``data/processed/<dataset>/``:

    interactions.parquet  user_id, item_id, timestamp, rating, position: one row per
                          interaction, sorted by user and then chronologically
    users.parquet         user_id ↔ raw_user_id
    items.parquet         item_id ↔ raw_item_id, plus item metadata
    stats.json            counts, history lengths, raw and output checksums

Steps: read → keep each (user, item) pair's first interaction → k-core filter until
stable → contiguous IDs → sort each user's history. IDs are ``0..n-1``, assigned in
sorted raw-ID order, so an item's ID is its column in a ``[batch, n_items]`` score
matrix; padding is left to the sequence datasets. Ties in timestamp are broken by
item ID. The same raw files always produce byte-identical output.
"""

import ast
import gzip
import hashlib
import html
import io
import json
import shutil
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generative_recsys.data.download import DATASETS, RemoteFile, verify
from generative_recsys.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

K_CORE = 5  # part of the benchmark's definition, not a tunable setting


class StatsMismatchError(ValueError):
    """The processed dataset doesn't have the expected size."""


@dataclass(frozen=True)
class Expected:
    n_users: int
    n_items: int
    n_interactions: int


EXPECTED = {
    "beauty": Expected(n_users=22_363, n_items=12_101, n_interactions=198_502),  # as in TIGER
    "ml-1m": Expected(n_users=6_040, n_items=3_416, n_interactions=999_611),  # pinned 2026-10-26
}


@dataclass(frozen=True)
class Processed:
    interactions: pd.DataFrame
    users: pd.DataFrame
    items: pd.DataFrame
    stats: dict[str, Any]


# Readers return interactions (user, item, timestamp, rating) with raw IDs, and one
# metadata row per item (item, ...). Raw files are passed in by file name.


def read_beauty(raw: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    with gzip.open(raw["reviews_Beauty_5.json.gz"], "rt", encoding="utf-8") as f:
        reviews = [json.loads(line) for line in f]
    interactions = pd.DataFrame(
        {
            "user": [r["reviewerID"] for r in reviews],
            "item": [r["asin"] for r in reviews],
            "timestamp": [r["unixReviewTime"] for r in reviews],
            "rating": [r["overall"] for r in reviews],
        }
    )

    wanted = set(interactions["item"])
    rows = []
    with gzip.open(raw["meta_Beauty.json.gz"], "rt", encoding="utf-8") as f:
        for line in f:
            meta = ast.literal_eval(line)  # Python literals, not JSON. Never eval(): it runs code.
            if meta["asin"] not in wanted:
                continue
            price = meta.get("price")
            rows.append(
                {
                    "item": meta["asin"],
                    "title": _clean_text(meta.get("title")),
                    "brand": _clean_text(meta.get("brand")),
                    "categories": meta.get("categories") or None,  # list of category paths
                    "description": _clean_text(meta.get("description")),
                    "price": float(price) if price is not None else None,
                }
            )
    # `related` and `salesRank` are left out on purpose: both were computed from every
    # customer's behaviour up to the 2014 crawl, so they'd leak future interactions.
    columns = ["item", "title", "brand", "categories", "description", "price"]
    return interactions, pd.DataFrame(rows, columns=columns)


def read_ml1m(raw: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(raw["ml-1m.zip"]) as z:
        ratings = z.read("ml-1m/ratings.dat")
        movies = z.read("ml-1m/movies.dat").decode("latin-1")
    # "::" separators need pandas' slow Python parser; a tab lets the C parser read it.
    interactions = pd.read_csv(
        io.BytesIO(ratings.replace(b"::", b"\t")),
        sep="\t",
        names=["user", "item", "rating", "timestamp"],
    )[["user", "item", "timestamp", "rating"]]

    rows = []
    for line in movies.splitlines():
        movie_id, title, genres = line.split("::")
        rows.append({"item": int(movie_id), "title": title, "genres": genres.split("|")})
    return interactions, pd.DataFrame(rows, columns=["item", "title", "genres"])


READERS: dict[str, Callable[[Mapping[str, Path]], tuple[pd.DataFrame, pd.DataFrame]]] = {
    "beauty": read_beauty,
    "ml-1m": read_ml1m,
}


def preprocess(
    dataset: str,
    raw_dir: str | Path = RAW_DATA_DIR,
    out_dir: str | Path = PROCESSED_DATA_DIR,
    *,
    registry: Mapping[str, tuple[RemoteFile, ...]] = DATASETS,
    expected: Mapping[str, Expected] = EXPECTED,
) -> dict[str, Any]:
    """Build ``out_dir/<dataset>/`` from the verified raw files and return its stats.

    Raises ``FileNotFoundError`` if a raw file is missing (it never downloads),
    ``ChecksumError`` if one isn't the pinned file, and ``StatsMismatchError`` if the
    result doesn't have the expected size, in which case nothing is written.
    """
    if dataset not in READERS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(READERS)}")
    raw = _verified_raw_files(dataset, Path(raw_dir), registry[dataset])
    processed = build(*READERS[dataset](raw))
    stats = {
        "dataset": dataset,
        **processed.stats,
        "raw_sha256": {f.name: f.sha256 for f in registry[dataset]},
    }
    if dataset in expected:
        _check_expected(dataset, stats, expected[dataset])
    return _write(Path(out_dir) / dataset, processed, stats)


def build(interactions: pd.DataFrame, items: pd.DataFrame, k: int = K_CORE) -> Processed:
    """Filter, re-index and sort raw interactions; attach metadata to the surviving items."""
    n_raw = len(interactions)
    # Keep each (user, item) pair's earliest interaction.
    df = interactions.sort_values(["user", "timestamp", "item"]).drop_duplicates(["user", "item"])
    n_duplicates = n_raw - len(df)
    df, rounds = kcore_filter(df, k)

    raw_users = np.sort(df["user"].unique())
    raw_items = np.sort(df["item"].unique())
    df = df.assign(
        user_id=pd.Categorical(df["user"], categories=raw_users).codes.astype(np.int64),
        item_id=pd.Categorical(df["item"], categories=raw_items).codes.astype(np.int64),
    )
    # Item IDs follow raw-ID order, so sorting by item_id breaks timestamp ties by raw ID.
    df = df.sort_values(["user_id", "timestamp", "item_id"], ignore_index=True)
    df["position"] = df.groupby("user_id").cumcount().astype(np.int64)
    out = pd.DataFrame(
        {
            "user_id": df["user_id"],
            "item_id": df["item_id"],
            "timestamp": df["timestamp"].astype(np.int64),
            "rating": df["rating"].astype(np.float32),
            "position": df["position"],
        }
    )

    users = pd.DataFrame({"user_id": np.arange(len(raw_users)), "raw_user_id": raw_users})
    meta = items.drop_duplicates("item").rename(columns={"item": "raw_item_id"})
    item_table = pd.DataFrame({"item_id": np.arange(len(raw_items)), "raw_item_id": raw_items})
    item_table = item_table.merge(meta, on="raw_item_id", how="left", indicator=True)
    n_without_metadata = int((item_table.pop("_merge") == "left_only").sum())

    stats = {
        "n_users": len(users),
        "n_items": len(item_table),
        "n_interactions": len(out),
        "density": round(len(out) / (len(users) * len(item_table)), 6),
        "history_length": _length_stats(out),
        "last_two_share_timestamp": _last_two_tied(out),
        "kcore": {"k": k, "rounds": rounds, "interactions_removed": n_raw - n_duplicates - len(df)},
        "duplicates_removed": n_duplicates,
        "items_without_metadata": n_without_metadata,
    }
    return Processed(out, users, item_table, stats)


def kcore_filter(df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, int]:
    """Drop users and items with fewer than ``k`` interactions until none are left.

    One pass isn't enough: removing a rare item can leave a user below ``k``.
    Returns the filtered frame and the number of rounds that removed something.
    """
    rounds = 0
    while True:
        user_counts = df.groupby("user")["item"].transform("size")
        item_counts = df.groupby("item")["user"].transform("size")
        keep = (user_counts >= k) & (item_counts >= k)
        if keep.all():
            return df, rounds
        df = df[keep]
        rounds += 1


def _verified_raw_files(
    dataset: str, raw_dir: Path, files: tuple[RemoteFile, ...]
) -> dict[str, Path]:
    paths = {}
    for file in files:
        path = raw_dir / dataset / file.name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download it first: "
                f"uv run python scripts/download.py --dataset {dataset}"
            )
        verify(file, path)
        paths[file.name] = path
    return paths


def _check_expected(dataset: str, stats: Mapping[str, Any], expected: Expected) -> None:
    got = Expected(stats["n_users"], stats["n_items"], stats["n_interactions"])
    if got != expected:
        raise StatsMismatchError(f"{dataset}: expected {expected}, got {got}")


def _write(dest: Path, processed: Processed, stats: dict[str, Any]) -> dict[str, Any]:
    """Write into a temporary sibling, then swap it in, so a crash never leaves half a dataset."""
    tmp = dest.with_name(f"{dest.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    tables = {
        "interactions.parquet": processed.interactions,
        "users.parquet": processed.users,
        "items.parquet": processed.items,
    }
    for name, table in tables.items():
        table.to_parquet(tmp / name, index=False)
    stats = {**stats, "output_sha256": {name: _sha256(tmp / name) for name in tables}}
    (tmp / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return stats


def _clean_text(value: str | None) -> str | None:
    # Amazon text has HTML entities ("&amp;"); empty strings become nulls.
    return (html.unescape(value).strip() or None) if value else None


def _length_stats(interactions: pd.DataFrame) -> dict[str, float]:
    lengths = interactions.groupby("user_id").size()
    return {
        "min": int(lengths.min()),
        "median": float(lengths.median()),
        "mean": round(float(lengths.mean()), 2),
        "p90": float(lengths.quantile(0.9)),
        "p99": float(lengths.quantile(0.99)),
        "max": int(lengths.max()),
    }


def _last_two_tied(interactions: pd.DataFrame) -> float:
    # Share of users whose last two interactions share a timestamp. For them the
    # tie-break alone decides which item is the test target in a leave-one-out split.
    timestamps = interactions.groupby("user_id")["timestamp"]
    last, second = timestamps.nth(-1).to_numpy(), timestamps.nth(-2).to_numpy()
    return round(float((last == second).mean()), 4)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
