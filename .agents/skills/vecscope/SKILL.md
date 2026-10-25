---
name: vecscope
description: Audit an embedding matrix or vector index before trusting retrieval results. Use when building or debugging RAG, semantic search, or recommendation retrieval, when "the same chunks always come back", when similarity scores all look alike, or before choosing ANN index parameters.
---

# vecscope

Run `vecscope audit <embeddings> --json` (or the MCP tool `audit_embeddings`; `pip install vecscope` if missing) and read `verdict`. For files above 100k rows a seeded sample is audited; pass `--sample 0` when ANN recall must be scored on every row.

- `fail` on **anisotropy** or **effective_rank**: center/whiten the embeddings or switch models before tuning retrieval; cosine scores are not discriminative. `details.mean_random_cosine_after_centering` shows what centering alone would achieve; if it is near 0, centering is enough and the model can stay.
- `fail` on **hubness**: use mutual k-NN, local scaling, or re-rank; raw cosine top-k will over-return hub items.
- `fail` on **collapse**: deduplicate the corpus first; top-k is wasted on copies.
- `fail` on **ann_recall**: raise `ef_search`/`nprobe` or rebuild the index; the model is not the problem.
- `fail` on **rogue_dimensions**: standardize each dimension (z-score) or drop `details.rogue_dims` before indexing; cosine is being decided by a few coordinates.
- `fail` on **norms**: L2-normalize before indexing and drop zero vectors (usually empty or failed inputs). Check **metric_mismatch** next.
- `fail` on **metric_mismatch**: the index metric (cosine vs inner product) changes the results; pick one deliberately and normalize if cosine is intended.
- `fail` on **distance_concentration**: nearest neighbors are barely closer than random items; reduce dimensionality, change the model, or add a re-ranker. Expect poor ANN recall too.
- `fail` on **outliers**: inspect `details.sample_ids`; they are usually junk, empty or foreign-language items. Remove or re-embed them.
- `fail` on **ood_queries** (needs `--queries`): the queries do not live where the corpus lives. Check that queries use the same model and the model's query prefix/instruction; inspect `details.ood_sample_ids` for topics the corpus does not cover.
- `fail` on **query_hubness**: a few corpus items answer most queries; re-rank or apply hubness reduction before trusting top-k.
- `fail` on **connectivity**: the k-NN graph has islands; raise HNSW `M`/`ef_construction` or add bridging content, otherwise whole regions are unreachable.

Every finding carries `healthy_range`, `warn_at` and `fail_at`; quote them when explaining a verdict.

Pass `--clusters 10` (or `clusters=10`) to get a per-cluster map with sample ids when you need to tell the user *where* a problem lives. Run `vecscope sweep <embeddings>` (MCP: `sweep_index`) before choosing HNSW/IVF parameters; take the recommended configuration and quote its recall and qps. The sweep uses the GPU automatically when the faiss build has one (IVF only; HNSW is CPU-only); mention the device from the result when comparing qps.

Report the numbers verbatim to the user. Do not guess at embedding quality without this evidence.
