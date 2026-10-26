# Data

Datasets are never committed. `data/raw/` and `data/processed/` are gitignored, and both
datasets' terms forbid redistribution, so each machine downloads them from the original source.

## Downloading

```bash
uv run python scripts/download.py --dataset beauty
uv run python scripts/download.py --dataset ml-1m
```

Files land in `data/raw/<dataset>/` under their upstream names and stay compressed.
Each file's size and SHA-256 are pinned in `src/generative_recsys/data/download.py`:

- **A file that's already present and valid is skipped,** so re-running is safe.
- **A download is written to `<name>.part`** and renamed only once it verifies. An interrupted
  download leaves nothing under the real name.
- **A file that doesn't match fails loudly** and the script exits non-zero. A downloaded file
  that doesn't match is discarded and the next source is tried. A file already present that
  doesn't match is left alone: delete it, or re-run with `--force`.

**Downloading is a one-time setup step, not a run.** Preprocessing, training and evaluation
only read local files. If one is missing, they fail and point here instead of downloading it.

## Datasets

### Amazon Beauty 2014, 5-core (benchmark)

| File | Bytes | SHA-256 |
|---|---|---|
| `reviews_Beauty_5.json.gz` | 44,786,479 | `6b4314ec0d74fba455d4ada65cf1313194665797c0f2d33a7c9a516da3968c20` |
| `meta_Beauty.json.gz` | 97,828,609 | `de77f42de70131ef35cb8e3274d4a290e511203c2cd0396b6440d541da055095` |

- **Sources,** tried in order:
  - the UCSD McAuley Lab copy, `https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/`
  - the original host, `https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/`.
    The [dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) still links
    here, but it refused connections when the checksums were pinned (2026-10-23).
- **Reviews:** one JSON object per line, with `reviewerID`, `asin`, `overall` (stars),
  `unixReviewTime` and the review text.
  - 198,502 reviews by 22,363 users of 12,101 items, matching the figures in the TIGER paper.
  - Timestamps are whole days (always midnight UTC), so one user's reviews can tie.
- **Metadata:** one product per line: title, description, brand, categories, price, sales rank,
  image URL and related items. 259,204 products, including all 12,101 in the 5-core set.
  - The lines are **Python dict literals, not JSON** (`{'asin': '…'}`). Parse them with
    `ast.literal_eval`. **Never use `eval`**, the dataset page's own suggestion: it would run
    any code in the file.
- **Checksums:** no checksum is published for these files. The pinned values are trust on
  first use: they guarantee every later download matches the 2026-10-23 one. The evidence that
  this one is the original is that its counts match the published figures exactly.
- **Terms:** no formal licence. The dataset page asks users to cite:
  - R. He and J. McAuley. *Ups and downs: Modeling the visual evolution of fashion trends with
    one-class collaborative filtering.* WWW 2016.
  - J. McAuley, C. Targett, Q. Shi and A. van den Hengel. *Image-based recommendations on styles
    and substitutes.* SIGIR 2015.

  The reviews are Amazon users' content. Use them for research only, don't redistribute them, and
  don't quote review text at length in write-ups.

### MovieLens-1M (smoke tests)

| File | Bytes | SHA-256 |
|---|---|---|
| `ml-1m.zip` | 5,917,549 | `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20` |

- **Source:** `https://files.grouplens.org/datasets/movielens/ml-1m.zip`
- **Contents:** `ml-1m/ratings.dat`, `movies.dat` and `users.dat`, with fields separated by `::`.
  - 1,000,209 ratings by 6,040 users of 3,706 movies (2000–2003).
  - Timestamps are in seconds. `movies.dat` is latin-1, not UTF-8.
  - The zip is read in place, without being extracted.
- **Checksum:** GroupLens publishes MD5 `c4d9eecfca2ab87c1945afe126590906`
  ([`ml-1m.zip.md5`](https://files.grouplens.org/datasets/movielens/ml-1m.zip.md5)). The pinned
  SHA-256 is of a download that matched it.
- **Terms** ([README](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)):
  - research use only
  - no endorsement by the University of Minnesota or GroupLens may be stated or implied
  - no redistribution without permission
  - no commercial use
  - cite F. M. Harper and J. A. Konstan. *The MovieLens Datasets: History and Context.*
    ACM TiiS 5(4), 2015.

## If a download fails

1. **Check whether the source is up.** For example:

   ```bash
   curl -sI https://files.grouplens.org/datasets/movielens/ml-1m.zip
   ```

   A `200` with the `Content-Length` from the table means it's fine. Retry later, or try from
   another network.
2. **If the source is gone,** find another copy and add its URL to that file's `urls` in
   `download.py`. Don't change the pinned checksum: it's what decides whether the new copy is
   the same file. If no copy matches, the data has changed, and that needs a deliberate decision
   recorded in an issue, because every result depends on it.

## Raw columns

The upstream definitions are in the "Format" section of the
[Amazon dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) and in the
ML-1M README (`unzip -p data/raw/ml-1m/ml-1m.zip ml-1m/README`). "Filled in" counts are for
the 12,101 items in the Beauty 5-core set.

**Beauty reviews** (`reviews_Beauty_5.json.gz`)

| Column | Meaning | Kept as |
|---|---|---|
| `reviewerID` | User ID | `raw_user_id` |
| `asin` | Amazon product ID | `raw_item_id` |
| `unixReviewTime` | Review date, unix seconds (always midnight UTC) | `timestamp` |
| `overall` | Star rating, 1–5 | `rating` |
| `reviewTime` | The same date as text, e.g. `"09 13, 2009"` | dropped |
| `reviewText`, `summary` | Review body and headline | dropped |
| `reviewerName` | Display name | dropped |
| `helpful` | `[helpful votes, total votes]` | dropped |

**Beauty metadata** (`meta_Beauty.json.gz`)

| Column | Meaning | Filled in | Kept? |
|---|---|---|---|
| `asin` | Product ID | 12,101 | yes |
| `title` | Product name | 12,094 | yes |
| `brand` | Brand | 10,003 | yes |
| `categories` | Category paths, e.g. `[['Beauty', 'Makeup', 'Face', 'Concealers']]` | 12,101 | yes |
| `description` | Product description | 11,163 | yes |
| `price` | Price in USD at crawl time (2014) | 11,516 | yes |
| `imUrl` | Image URL | 12,094 | no: no phase uses images |
| `salesRank` | `{category: rank}` at crawl time | 11,883 | no: leaks the future (see below) |
| `related` | `also_bought`, `also_viewed`, `bought_together`, `buy_after_viewing`: product IDs | 12,019 | no: leaks the future |

**ML-1M** (fields separated by `::`)

| File | Columns | Kept |
|---|---|---|
| `ratings.dat` | `UserID`, `MovieID`, `Rating` (1–5 whole stars), `Timestamp` (unix seconds) | all |
| `movies.dat` | `MovieID`, `Title` (with year, e.g. `Toy Story (1995)`), `Genres` (`\|`-separated) | all |
| `users.dat` | `UserID`, `Gender`, `Age` (bracket code), `Occupation` (code 0–20), `Zip-code` | none |

## Preprocessing

```bash
uv run python scripts/preprocess.py --dataset beauty   # ~20 s
uv run python scripts/preprocess.py --dataset ml-1m    # ~5 s
```

This reads the raw files, after checking them against the pinned checksums, and writes
`data/processed/<dataset>/`. Steps:

1. **Keep each (user, item) pair's first interaction.** Neither dataset has duplicates today.
2. **Filter to the 5-core, repeatedly until stable.** Every remaining user and item has at least
   5 interactions. One pass isn't enough: removing a rare item can leave a user with 4, and that
   user then has to go too.
   - Beauty is already a 5-core: 0 rounds.
   - ML-1M needs 1 round, which removes 290 movies with fewer than 5 ratings. A few users drop
     below their original 20 ratings as a result; the shortest history left is 18.
3. **Assign contiguous IDs.** Users become `0..U-1` and items `0..N-1`, in sorted order of their
   raw IDs, so the numbering doesn't depend on row order.
   - An item's ID is its column in a `[batch, n_items]` score matrix.
   - There is no padding ID: that's for the sequence datasets and models to add, e.g. `n_items`.
4. **Order each user's history by `(timestamp, item ID)`.**
   - **The tie-break matters.** For 43.5% of Beauty users (42.1% of ML-1M users), the last two
     interactions share a timestamp, so the tie-break alone decides which is the test item in a
     leave-one-out split.
   - Since item IDs follow raw-ID order and the Beauty file is sorted by product ID, this is the
     raw file's order: what a stable time sort of the file, as public preprocessing scripts do,
     produces.
5. **Write the tables and `stats.json`, then check the counts.** The script fails, writing nothing,
   if the counts aren't the expected ones.

The output is byte-identical across runs with the same raw files and locked dependencies.
`stats.json` records the SHA-256 of every input and output.

| File | Columns |
|---|---|
| `interactions.parquet` | `user_id`, `item_id`, `timestamp`, `rating`, `position`: one row per interaction, sorted by user and then chronologically. `position` is the index in that user's history, from 0. |
| `users.parquet` | `user_id`, `raw_user_id` |
| `items.parquet` | `item_id`, `raw_item_id`, then Beauty: `title`, `brand`, `categories`, `description`, `price`; ML-1M: `title`, `genres`. Missing values are nulls, and HTML entities are decoded. |
| `stats.json` | Counts, density, history-length percentiles, share of users whose last two interactions tie, k-core rounds, input and output SHA-256s |

| | Users | Items | Interactions | Density | History length (median / p90 / max) |
|---|---|---|---|---|---|
| Beauty | 22,363 | 12,101 | 198,502 | 0.073% | 6 / 14 / 204 |
| ML-1M | 6,040 | 3,416 | 999,611 | 4.84% | 96 / 400 / 2,277 |

**Why `related` and `salesRank` are dropped.** Both were computed at the 2014 crawl, from every
customer's behaviour, including interactions that come after our test items. For example, if
`also_bought` links a cleanser to a toner because many people later bought both, then an item
representation built from it lets a model "predict" the toner with information from the future.

## Splits

Covered by [#12](https://github.com/MenciusChin/generative-recsys/issues/12).
