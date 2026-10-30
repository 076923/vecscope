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
vecscope compare <embeddings A> <embeddings B>   # MCP: compare_embeddings; same rows, two spaces
```

In CI: commit a `--json` run as the baseline and run `vecscope audit <embeddings> --baseline
baseline.json` on every rebuild; it exits 1 when a verdict got worse or when `--fail-on`
(default `fail`) trips on its own. Report the `regressions` list, not every `warn`; a `fail`
that was already in the baseline is not a regression. Verdict bands hide slow drift, so each
change row also carries `worsened` and `worsening` (positive = worse, in the metric's units);
`--tolerances tol.json` (`{"collapse": 0.01, "ann_recall": 0.02}`) makes a worsening beyond
the budget a `value_regressions` entry and exit 1 even when the verdict held. Add
`--require-comparable` to a gate: it exits 2 when the baseline was made with different n,
dim, k, sampling, thresholds or near-dup cutoff, instead of printing a diff that measures the
settings rather than the corpus. Pick tolerances from measured retrieval quality (README,
"Calibrating thresholds to your domain"), not from what makes the pipeline green. Two rules
can fail a run and the `gate` block (JSON) or line (text) names which one did: the absolute
thresholds (`--fail-on`) or the baseline (`verdict_regressions`, `value_regressions`). When
the user's healthy corpus sits at `warn`/`fail` on a finding by the public defaults, gate
with `--fail-on none --baseline` so only the change is judged, and say that is the policy.

MCP server on a shared machine: start it with `--allow-dir DIR` (or `VECSCOPE_ALLOWED_DIRS`)
so the tools can only read embeddings under those directories; a refused path says so.

MCP only: `describe_findings` lists every finding with thresholds, causes and actions; pass
`verbose=false` to `audit_embeddings` to keep the response short when most findings are ok.

Inputs: one vector per row in `.npy/.npz/.parquet/.jsonl/.json/.csv`. Above 100k rows a
seeded sample is audited and `sampled: true` is set; pass `--sample 0` to audit every row,
and always do so when `--ann` is used (row ids must line up). When a sampled verdict sits
near a threshold, re-run with `--sample-repeats 3` (MCP: `sample_repeats`): findings then
carry `value_range` and `verdict_stable`, and `unstable_findings` names the borderline ones.
A sampled value does not mean the same thing for every finding, and the explanation says
which: `hubness` and `antihubs` on a sample are a lower bound (hubness grows with density;
compare only runs with the same sample size), the `collapse` share is about sample fraction
times the full share (a pair counts only when both rows were drawn; the verdict uses the
estimated full share in `details.estimated_full_share`), `outliers` are over-counted on
small samples, and `connectivity` sees only the sample's own islands. The response lists
`inconclusive_findings` (hubness, antihubs and connectivity verdicts other than `fail`,
outlier verdicts other than `ok`): never report those as healthy; say they need `--sample 0`.
anisotropy, effective_rank, rogue_dimensions, norms, metric_mismatch and
distance_concentration are stable estimates. `value_range` is the spread between repeated
samples, not an interval that contains the full-corpus value.

## Read it

- `verdict` is the worst finding. Each finding has `value`, `healthy_range`, `warn_at`,
  `fail_at`, `explanation`, `details`, `likely_causes`, `suggested_action`, `verify_by`.
- Quote the numbers and ranges verbatim. Do not describe embedding quality without them.
- Thresholds are defaults calibrated on seven public models and re-checked on 19 models
  against labeled retrieval data; they are a starting point, not a universal bar. `warn` on
  hubness is common for good models; `ood_queries` measures query-to-corpus shift, which
  short queries against long passages always show, so read it with the model's query
  prefix in mind and never as a quality score. Override with
  `--thresholds file.json` (MCP: `thresholds`) when the user's domain has its own baseline,
  and derive that baseline from labeled queries split into a tuning and a held-out part; a
  threshold chosen to make the current corpus pass predicts nothing.
- A `fail` does not prove retrieval is bad and all-`ok` does not prove it is good: the audit
  measures the vector space and the index, not relevance. Say so when reporting.

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
5. **Adopt only confirmed improvements**; otherwise revert and try the next cause. Measure
   latency and memory alongside quality: a transform that fixes a finding can cost more at
   serving time than the finding did.

Finding-specific notes:

- **anisotropy**: `details.mean_random_cosine_after_centering` shows what centering alone
  would achieve. Near 0 means the model can stay; centering changes rankings, so step 4 matters.
  On labeled data anisotropy by itself did not lower nDCG@10 (the best public models sit at
  0.58 to 0.82, below the 0.85 warn threshold); it breaks absolute score thresholds, not
  ranking. Suspect the model only when `distance_concentration`, `effective_rank` or
  `rogue_dimensions` move with it.
- **hubness / antihubs**: hubness often follows anisotropy; fix anisotropy first, then re-rank
  with mutual k-NN or local scaling if hubs remain.
- **collapse**: read `details.largest_near_duplicate_groups` before blaming the model. Many
  small groups (pairs, triples) are duplicate items: the same image or text under several ids;
  deduplicate or collapse them at serving time. A few huge groups are an embedding failure
  (a constant vector) or boilerplate. `details.sample_pairs` lists row pairs with their cosine.
  With a second embedding of the same rows, `compare` settles it: duplicates both spaces flag
  with the same partner belong to the items. Quote `compare`'s `neighbor_scope` with its
  numbers: `sample` means the neighbors were recomputed inside the sampled rows and the
  agreement is about the sample's own graph (it reads higher than the full corpus); pass
  full-corpus neighbor lists (`--neighbors-a/-b`, MCP `neighbors_a_path`/`neighbors_b_path`)
  when the question is about the production ranking.
- **outliers / connectivity**: `details.sample_ids` are row ids to show the user.
- **norms / metric_mismatch**: decide the index metric deliberately; normalize if cosine is intended.
- **ann_recall**: if `details.rows_with_fewer_than_k_ids_ratio` is above 0 the ANN file was
  queried with k and included self, which caps recall; re-export with k + 1 before judging the
  index. Then raise `ef_search` / `nprobe` before rebuilding; `vecscope sweep` picks values
  and reports qps per device (GPU is used for IVF when the faiss build has one; HNSW is CPU-only).
  Report the recommendation's `margin`; when it is `tight`, propose the `safer_alternative`
  instead, and relay `caveats`: HNSW recall falls as the corpus grows, so a sweep on a sample
  is optimistic for HNSW (IVF can move either way because nlist follows the row count), and
  qps only ranks settings on this machine. Each point also carries
  `latency_ms_p50/p95/p99` (single-query, this process) and `index_bytes`; use them to
  weigh a setting's memory and tail latency against its recall, and pass real query traffic
  as `--queries` whenever a log exists, because corpus-drawn queries can pick a different
  setting. Confirm the pick in the serving environment before fixing it.
- **ood_queries**: check that queries use the same model and the model's query prefix before
  suspecting the corpus; `details.ood_sample_ids` lists the farthest queries. It measures
  distribution shift, not failure: good models score 0 to 0.65 on short-query benchmarks,
  and queries from a closely related model can score 1.0 and still retrieve well, so confirm
  with real queries (step 4). A wrong-language model can score 0 when both sides are encoded
  equally badly; `query_hubness` and `hubness` catch that case.
