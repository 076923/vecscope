---
name: vecscope
description: Audit an embedding matrix or vector index before trusting retrieval results. Use when building or debugging RAG, semantic search, or recommendation retrieval, when "the same chunks always come back", when similarity scores all look alike, before choosing ANN index parameters, or when queries seem to miss documents that are clearly in the corpus.
---

# vecscope

`vecscope` turns an embedding matrix (and optionally the ANN neighbor lists and the query
embeddings) into deterministic evidence: per-finding measurements, `ok` / `warn` / `fail`
verdicts, thresholds, likely causes, a suggested first change, and how to verify it. It
diagnoses the **vector search implementation** (model geometry, corpus hygiene, metric,
index). It does not measure relevance; judging whether the right documents come back needs
labeled queries.

## Run it

```bash
pip install vecscope                       # or "vecscope[mcp]" / "vecscope[index]"
vecscope audit <embeddings> --json         # MCP: audit_embeddings
vecscope audit <embeddings> --queries <query embeddings> --json   # adds query-side findings
vecscope audit <embeddings> --clusters 10 --json                  # adds a per-cluster map
vecscope sweep <embeddings>                # MCP: sweep_index; picks HNSW/IVF parameters
```

MCP only: `describe_findings` lists every finding with thresholds, causes and actions; pass
`verbose=false` to `audit_embeddings` to keep the response short when most findings are ok.

Inputs: one vector per row in `.npy/.npz/.parquet/.jsonl/.json/.csv`. Above 100k rows a
seeded sample is audited and `sampled: true` is set; pass `--sample 0` to audit every row,
and always do so when `--ann` is used (row ids must line up).

## Read it

- `verdict` is the worst finding. Each finding has `value`, `healthy_range`, `warn_at`,
  `fail_at`, `explanation`, `details`, `likely_causes`, `suggested_action`, `verify_by`.
- Quote the numbers and ranges verbatim. Do not describe embedding quality without them.
- Thresholds are defaults calibrated on eight public models; they are a starting point, not
  a universal bar. `warn` on anisotropy or hubness is common for good models. Override with
  `--thresholds file.json` (MCP: `thresholds`) when the user's domain has its own baseline.

## Act on it: diagnose, change one thing, re-verify

1. **Report the evidence**: the failing findings with values and ranges, plus the sampled flag.
2. **Separate the causes**: use `likely_causes` and the other findings to decide whether the
   problem is the model (anisotropy, effective_rank, rogue_dimensions, distance_concentration),
   the corpus (collapse, outliers, connectivity, antihubs), the metric (norms, metric_mismatch),
   the index (ann_recall) or the query pipeline (ood_queries, query_hubness). `--clusters`
   shows where in the corpus a problem lives.
3. **Try the smallest change first**: `suggested_action` names it. Any transform applied to
   the corpus (centering, standardizing, normalizing, truncating) must be applied to the
   queries too, and the index rebuilt.
4. **Re-audit and compare**: `verify_by` says which number should move. Then compare retrieval
   on real queries before and after (recall@k against exact search, and relevance if labels
   exist). A finding turning `ok` is not proof that search got better.
5. **Adopt only confirmed improvements**; otherwise revert and try the next cause.

Finding-specific notes:

- **anisotropy**: `details.mean_random_cosine_after_centering` shows what centering alone
  would achieve. Near 0 means the model can stay; centering changes rankings, so step 4 matters.
- **hubness / antihubs**: hubness often follows anisotropy; fix anisotropy first, then re-rank
  with mutual k-NN or local scaling if hubs remain.
- **collapse / outliers / connectivity**: `details.sample_ids` are row ids to show the user.
- **norms / metric_mismatch**: decide the index metric deliberately; normalize if cosine is intended.
- **ann_recall**: raise `ef_search` / `nprobe` before rebuilding; `vecscope sweep` picks values
  and reports qps per device (GPU is used for IVF when the faiss build has one; HNSW is CPU-only).
- **ood_queries**: check that queries use the same model and the model's query prefix before
  suspecting the corpus; `details.ood_sample_ids` lists the farthest queries.
