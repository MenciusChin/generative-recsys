# Roadmap

## Phase 0 — Foundations

- [ ] Choose the first public dataset
- [ ] Define train / validation / test splitting
- [ ] Build reproducible evaluation utilities
- [ ] Add experiment configuration and deterministic seeds
- [ ] Establish Recall@K / NDCG@K baselines

## Phase 1 — Classical retrieval

- [ ] Popularity baseline
- [ ] Matrix factorization baseline
- [ ] Two-tower retrieval
- [ ] ANN candidate retrieval
- [ ] Compare accuracy, coverage, and serving cost

## Phase 2 — Sequential recommendation

- [ ] Implement SASRec-style sequential recommender
- [ ] Position / masking sanity checks
- [ ] Sequence-length ablations
- [ ] Compare against classical retrieval

## Phase 3 — Semantic IDs

- [ ] Item embedding pipeline
- [ ] Residual quantization / RQ-VAE
- [ ] Semantic ID construction
- [ ] Codebook utilization analysis
- [ ] Compare random vs semantic vs collaborative IDs

## Phase 4 — Generative retrieval

- [ ] Autoregressive item-ID generation
- [ ] TIGER-style experiments
- [ ] Valid-ID constrained decoding
- [ ] Beam-search / sampling experiments
- [ ] Catalog-update failure analysis

## Phase 5 — Hybrid + serving

- [ ] Classical + generative hybrid retrieval
- [ ] Online inference API
- [ ] Batching and caching
- [ ] Latency / throughput benchmark
- [ ] Final architecture + experiment report
