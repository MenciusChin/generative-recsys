# Generative Recommender Systems Lab

A systems-oriented implementation path from classical recommendation to **generative retrieval**.

The goal is not just to reproduce a paper. This repo will build a recommendation stack incrementally, keep strong baselines, measure each modeling choice, and eventually expose the system through an online serving layer.

## Learning path

```text
Popularity / MF
      ↓
Two-Tower Retrieval
      ↓
Sequential Recommendation (SASRec)
      ↓
Semantic IDs / RQ-VAE
      ↓
Generative Retrieval
      ↓
Constrained Decoding
      ↓
Hybrid Retrieval + Serving
```

## What I want to understand

- When do semantic IDs beat arbitrary item IDs?
- How do codebook size and ID depth affect quality?
- How does generative retrieval compare with two-tower and sequential baselines?
- What happens when the catalog changes?
- When is a hybrid generative + classical retrieval system preferable?
- How do batching, caching, and constrained decoding affect online latency?

## Planned evaluation

**Recommendation quality**
- Recall@K
- NDCG@K
- MRR@K
- Coverage / popularity bias

**Systems**
- throughput
- p50 / p95 latency
- GPU memory
- batch-size sensitivity

## Repository layout

```text
src/generative_recsys/
tests/
data/
configs/
notebooks/
```

See [ROADMAP.md](ROADMAP.md) for milestones.

## Status

**Phase 0 — project scaffolding.**

The first implementation target is a reproducible dataset + evaluation pipeline followed by simple popularity and matrix-factorization baselines.
