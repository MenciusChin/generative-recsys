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

## Preprocessing, splits and reproducibility checks

Preprocessing is covered by [#11](https://github.com/MenciusChin/generative-recsys/issues/11),
and splits by [#12](https://github.com/MenciusChin/generative-recsys/issues/12).
