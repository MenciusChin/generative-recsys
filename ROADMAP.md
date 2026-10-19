# Roadmap

A production-style generative recommender, built in the order the field developed it:
classical baselines → sequential models → Semantic IDs → generative retrieval → hybrid serving.
The reference design is TIGER ([Rajput et al., 2023](https://arxiv.org/abs/2305.05065)).

This file holds the stable plan: phases, decisions and exit criteria. Day-to-day tasks live in
[GitHub Issues](https://github.com/MenciusChin/generative-recsys/issues), grouped by milestone (one per phase).

## Principles

- **Evaluation first.** No model result counts until the data split and metrics are tested and trusted.
- **Reproducible results.** Every number in the results table comes from a logged run with its
  config and git commit, and can be regenerated from scratch.
- **Honest comparisons.** Generative retrieval is compared against strong baselines (SASRec,
  two-tower + ANN) on the same split, metrics and hardware, including where it loses.
- **Systems, not just accuracy.** Each model is also measured on what it costs to serve.
- **Device-agnostic.** The same code runs on CUDA, Apple MPS or CPU; only the config profile changes.

## Decisions

| Area | Decision |
|---|---|
| Goal | Portfolio / learning: clean code, per-phase write-ups, parity with published numbers |
| Python & env | Python 3.12, `uv`, committed `uv.lock` |
| Package | `generative_recsys` (in `src/`) |
| Framework | Plain PyTorch with a thin custom trainer; HF `transformers` only for T5 backbone configs |
| Compute | Develop on Mac (MPS), with CUDA where available; device picked by `get_device()` (CUDA → MPS → CPU) |
| Datasets | Amazon Beauty 2014 5-core (benchmark), MovieLens-1M (smoke tests) |
| Data handling | `pandas`, processed data stored as parquet |
| Config | OmegaConf + dataclass schemas, CLI overrides, `local` / `cuda` profiles |
| Experiment tracking | Local JSON run logs (`runs/<id>/`); W&B as an opt-in flag from Phase 2 |
| ANN | `faiss-cpu`: exact search, HNSW and IVF |
| Workflow | `main` always green; PRs only; squash merges; tag per finished phase |
| Branch names | `phase-<N>/<issue#>-<slug>`, `fix/<issue#>-<slug>`, `chore/<slug>`, `docs/<slug>`; checked in CI |
| Tracking work | GitHub Issues + Milestones (one milestone per phase) |

## Research questions

The questions from the [README](README.md), and the phase that answers each one:

| Question | Phase |
|---|---|
| When do semantic IDs beat arbitrary item IDs? | 4, 5 |
| How do codebook size and ID depth affect quality? | 4, 5 |
| How does generative retrieval compare with two-tower and sequential baselines? | 5 |
| What happens when the catalog changes? | 5 |
| When is a hybrid generative + classical retrieval system preferable? | 6 |
| How do batching, caching and constrained decoding affect online latency? | 6 |

## Evaluation protocol

Fixed in Phase 1, then shared by every model.

**Data and split**
- Interactions sorted by timestamp per user, 5-core filtering.
- **Leave-one-out split:** last item → test, second-to-last → validation, the rest → training.
- Fixed seeds; headline results are the mean of ≥3 seeds.
- *Open:* whether items already in the user's history are masked during ranking. We'll decide in
  Phase 1 and apply the same rule to every model.

**Recommendation quality**
- **Full ranking** over the whole item catalog; no sampled negatives.
- Per-user accuracy: **Recall@K**, **NDCG@K**, **MRR@K** for K ∈ {5, 10}. Model selection uses validation NDCG@10.
- Catalog-level: **coverage@K** (share of the catalog that is ever recommended) and **popularity bias**
  (mean popularity percentile of recommended items).

**Systems**
- **p50 / p95 latency** and **throughput**, at several batch sizes
- **GPU memory** (or unified memory on MPS)
- All measured on the same hardware for every model, with the hardware recorded alongside the results.

## Phases

### Phase 0 — Scaffolding · `v0.0-scaffold`

Project skeleton, tooling and CI. No data handling or models yet.

- Migrate the existing `pyproject.toml` to `uv` and Python 3.12; complete the `src/generative_recsys` layout
- Makefile (`lint`, `format`, `test`), pre-commit, ruff
- Utilities: `get_device()`, `seed_everything()`, config loading, local run logger
- GitHub Actions CI: install, lint, tests on CPU, branch-name check

**Done when:** a fresh clone runs `uv sync && make test` cleanly on the Mac, and CI passes on `main`.

### Phase 1 — Data and evaluation harness · `v0.1-eval`

- Download scripts with checksums for Amazon Beauty (reviews + item metadata) and ML-1M; sources documented in `data/README.md`
- Preprocessing to parquet: 5-core filtering, ID remapping, per-user chronological sequences, dataset stats
- Leave-one-out split and sequence datasets, with tests that check for leakage
- Full-ranking accuracy metrics plus coverage and popularity bias, tested against hand-computed cases
- Popularity baseline end to end, and a script that builds `results/` tables from run logs

**Done when:** popularity numbers on both datasets are reproducible from a clean checkout, and the
dataset stats match published figures (Beauty: ~22.4k users, ~12.1k items, ~198k interactions).

### Phase 2 — Classical retrieval · `v0.2-classical`

- ItemKNN and BPR matrix factorization
- **Two-tower** retrieval with in-batch negatives
- ANN candidate retrieval with FAISS (exact, HNSW, IVF): recall lost vs. latency gained
- Thin trainer: training loop, checkpoints, early stopping on validation NDCG@10; opt-in W&B logging
- First comparison of accuracy, coverage and serving cost

**Done when:** all classical baselines appear in the results table, with ANN recall and latency reported
next to exact search.

### Phase 3 — Sequential recommendation · `v0.3-sequential`

- **SASRec**: the main sequential baseline and the bar generative retrieval has to clear
- Sanity checks: causal masking (no peeking at future items), padding, position embeddings
- Sequence-length ablations
- Comparison against the classical baselines

**Done when:** SASRec on Beauty lands near the SASRec numbers reported in the TIGER paper.

### Phase 4 — Semantic IDs · `v0.4-semantic-ids`

- Item text embeddings from title, brand, categories and description, using a sentence encoder
- **RQ-VAE** (3 levels × 256 codes) plus a collision-resolving 4th token, as in TIGER; **RQ-KMeans** as a simpler comparison
- Three ID types to compare downstream: **random**, **semantic** (from content) and **collaborative**
  (quantized from two-tower or SASRec item embeddings)
- Diagnostics: codebook utilization, collision rate, reconstruction loss, and whether items that share
  a prefix are actually similar

**Done when:** every item has a unique ID under each scheme, no codebook level has collapsed, and the
diagnostics are written up in `docs/`.

### Phase 5 — Generative retrieval · `v0.5-generative`

- Small T5-style encoder-decoder that reads a user's history as ID tokens and generates the next item's ID
- Beam search **constrained by a trie** of valid IDs, so every output is a real item
- Decoding experiments: beam width, sampling vs. beam search, constrained vs. unconstrained
- Ablations: random vs. semantic vs. collaborative IDs, codebook size, ID depth
- **Catalog-update failure analysis:** add items after training and measure whether they are ever retrieved,
  and what retraining or re-indexing it takes to fix that

**Done when:** results match or beat SASRec on Beauty (TIGER reports about +20% NDCG@10 over SASRec),
and the ablations explain where the gains come from.

### Phase 6 — Hybrid retrieval and serving · `v0.6-serving`

- **Hybrid retrieval:** merge candidates from generative retrieval and two-tower + ANN; when does each one win?
- FastAPI inference service for generative, two-tower and hybrid retrieval
- Batched constrained beam search with KV cache, plus caching of user encodings and frequent prefixes
- Docker image, load tests, latency and throughput benchmark across batch sizes
- Final architecture and experiment report

**Done when:** the report compares p50/p95 latency, throughput, memory and recall for all three
approaches on the same hardware, and answers every research question above.

### Phase 7 — Stretch

Pick based on what Phases 1–6 show:

- Initialize the backbone from a small pretrained LLM, with ID tokens added to its vocabulary
- A ranking stage behind retrieval
- Better cold-start handling via content-only Semantic IDs
- Additional Amazon categories (Sports, Toys)

## Results

The generated table will live in [`results/`](results/) and be summarized in the README as phases complete.
