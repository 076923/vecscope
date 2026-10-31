---
name: vecscope
description: Audit an embedding matrix or vector index before trusting retrieval results. Use when building or debugging RAG, semantic search, or recommendation retrieval, when "the same chunks always come back", when similarity scores all look alike, before choosing ANN index parameters, or when queries seem to miss documents that are clearly in the corpus.
---

# vecscope

`vecscope` turns an embedding matrix (and optionally the ANN neighbor lists and the query
embeddings) into deterministic evidence: per-finding measurements, `ok` / `warn` / `fail`
verdicts, thresholds, likely causes, a suggested first change, and how to verify it.

It diagnoses the **vector search implementation** (model geometry, corpus hygiene, metric,
index). It does not measure relevance; judging whether the right documents come back needs
labeled queries.

## Run it

```bash
pip install vecscope                       # or "vecscope[mcp]" / "vecscope[index]"
vecscope audit <embeddings> --json         # MCP: audit_embeddings
vecscope audit <embeddings> --queries <query embeddings> --json   # adds query-side findings
vecscope audit <embeddings> --queries <today's queries> --baseline good.json --json   # + query_shift
vecscope audit <embeddings> --queries <q> --qrels <qrels.tsv> --json   # + nDCG/recall/MRR on labels
vecscope audit <index.faiss | qdrant://host:6333/coll | ...> --ann-from-source --sample 0 --json
vecscope audit <embeddings> --sample 0 --knn-cache knn.npz --json   # appended rows only
vecscope audit <embeddings> --clusters 10 --json                  # adds a per-cluster map
vecscope sweep <embeddings>                # MCP: sweep_index; picks HNSW/IVF parameters
vecscope compare <embeddings A> <embeddings B>   # MCP: compare_embeddings; same rows, two spaces
vecscope calibrate fit_*.json --min-quality 0.45 --holdout hold_*.json --out th.json  # own thresholds
```

In CI:

- Commit a `--json` run as the baseline and run `vecscope audit <embeddings> --baseline
  baseline.json` on every rebuild; it exits 1 when a verdict got worse or when `--fail-on`
  (default `fail`) trips on its own.
- Report the `regressions` list, not every `warn`; a `fail` that was already in the baseline
  is not a regression.
- Verdict bands hide slow drift, so each change row also carries `worsened` and `worsening`
  (positive = worse, in the metric's units).
- `--tolerances tol.json` (`{"collapse": 0.01, "ann_recall": 0.02}`) makes a worsening beyond
  the budget a `value_regressions` entry and exit 1 even when the verdict held.
- Add `--require-comparable` to a gate: it exits 2 when the baseline was made with different
  n, dim, k, sampling, thresholds or near-dup cutoff, instead of printing a diff that measures
  the settings rather than the corpus.
- Pick tolerances from measured retrieval quality (README, "Calibrating thresholds to your
  domain"), not from what makes the pipeline green.
- Two rules can fail a run and the `gate` block (JSON) or line (text) names which one did:
  the absolute thresholds (`--fail-on`) or the baseline (`verdict_regressions`,
  `value_regressions`).
- When the user's healthy corpus sits at `warn`/`fail` on a finding by the public defaults,
  gate with `--fail-on none --baseline` so only the change is judged, and say that is the
  policy.

MCP server file access:

- By default it reads only under the directory it was started in; `--allow-dir DIR` (or
  `VECSCOPE_ALLOWED_DIRS`) names the directories, and a refused path says so.
- `--allow-any` lifts the restriction; do not suggest it on a shared machine.
- Network vector stores are refused unless the server runs with `--allow-network`.

Vector stores:

- The path can be a faiss index file or a store URI (`qdrant://`, `chroma://`, `milvus://`,
  `weaviate://`, or `+local` forms with a directory).
- Rows come in ascending id order and findings carry `sample_source_ids` (the store's ids)
  next to `sample_ids`; report those to the user.
- `--ann-from-source` (MCP `ann_from_source`) scores the store's own index the way `--ann`
  does.
- Never put an API key in the URI; use `api_key_env=NAME`.

Large corpora that grow by appending: `--knn-cache knn.npz` with `--sample 0` reuses the
exact neighbor lists and computes only the new rows; `knn_cache.status` says whether the cache
was reused, extended or rebuilt.

MCP only:

- `describe_findings` lists every finding with thresholds, causes and actions.
- Every `audit_embeddings` response carries an `audit_id`; after a change, re-audit with
  `baseline=<that id>` and read `baseline_diff` (`improvements`, `regressions`,
  `retrieval_changes`) and the "vs baseline" part of `summary` instead of comparing two
  responses by hand.
- A baseline audited on the same corpus with queries also gives `query_shift`; after a
  transform it does not (`query_reference_note`).
- Pass `verbose=false` to keep the response short when most findings are ok.

Inputs: one vector per row in `.npy/.npz/.parquet/.jsonl/.json/.csv`.

- Above 100k rows a seeded sample is audited and `sampled: true` is set; pass `--sample 0` to
  audit every row, and always do so when `--ann` is used (row ids must line up).
- When a sampled verdict sits near a threshold, re-run with `--sample-repeats 3` (MCP:
  `sample_repeats`): findings then carry `value_range` and `verdict_stable`, and
  `unstable_findings` names the borderline ones.
- A sampled value does not mean the same thing for every finding, and the explanation says
  which:
  - `hubness` and `antihubs` on a sample are a lower bound (hubness grows with density;
    compare only runs with the same sample size).
  - The `collapse` share is about sample fraction times the full share (a pair counts only
    when both rows were drawn; the verdict uses the estimated full share in
    `details.estimated_full_share`).
  - Exact copies are counted over every row even when sampling
    (`details.full_corpus_exact_copy_share`); near-duplicates that are not exact copies are
    only seen in the sample.
  - `outliers` are over-counted on small samples.
  - `connectivity` sees only the sample's own islands.
- The response lists `inconclusive_findings` (hubness, antihubs and connectivity verdicts
  other than `fail`, outlier verdicts other than `ok`): never report those as healthy; say
  they need `--sample 0`.
- anisotropy, effective_rank, rogue_dimensions, norms, metric_mismatch and
  distance_concentration are stable estimates.
- `value_range` is the spread between repeated samples, not an interval that contains the
  full-corpus value.

## Read it

- `verdict` is the worst finding. Each finding has `value`, `healthy_range`, `warn_at`,
  `fail_at`, `explanation`, `details`, `likely_causes`, `suggested_action`, `verify_by`.
- Quote the numbers and ranges verbatim. Do not describe embedding quality without them.
- Thresholds are defaults calibrated on seven public models and re-checked on 19 models
  against labeled retrieval data; they are a starting point, not a universal bar.
  - `warn` on hubness is common for good models.
  - `query_gap` is a distance ratio that short queries against long passages keep at 1 to 2.5
    for good pipelines, so read it as "same space or not", never as a quality score.
  - `ood_queries` is deprecated: report `query_gap` instead.
  - Override with `--thresholds file.json` (MCP: `thresholds`) when the user's domain has its
    own baseline, and derive that baseline from labeled queries split into a tuning and a
    held-out part; a threshold chosen to make the current corpus pass predicts nothing.
  - `vecscope calibrate` proposes that file from audits of several pipeline states run with
    `--qrels`. Use only the pairs it proposes; a finding it marks as not tracking quality (or
    tracking it the other way) should not gate on that data. Quote its holdout line.
- A `fail` does not prove retrieval is bad and all-`ok` does not prove it is good: the audit
  measures the vector space and the index, not relevance. Say so when reporting.

## Act on it: diagnose, change one thing, re-verify

1. **Report the evidence**: the failing findings with values and ranges, plus the sampled flag.
   - Start from `plan`: the non-ok findings grouped by shared cause, in fix order, each with
     `first_change` and `re_measure`.
   - Work through it one step at a time and re-audit after each.
   - Each finding in a step has `placed_because`. "this group's cause covers it" is the
     group's own finding; a reason naming another finding ("anisotropy is not ok; ...") is a
     grouping by co-occurrence. Say so when you report it, and let the re-measure numbers
     confirm the shared cause before you call it one.
   - A step whose findings are all `inconclusive` (`only_inconclusive`) needs `--sample 0`
     before anything is changed.
2. **Separate the causes**: use `likely_causes` and the other findings to decide whether the
   problem is
   - the model (anisotropy, effective_rank, rogue_dimensions, distance_concentration),
   - the corpus (collapse, outliers, connectivity, antihubs),
   - the metric (norms, metric_mismatch),
   - the index (ann_recall)
   - or the query pipeline (query_gap, query_shift, query_hubness).

   `--clusters` shows where in the corpus a problem lives.
3. **Try the smallest change first**: `suggested_action` names it. Any transform applied to
   the corpus (centering, standardizing, normalizing, truncating) must be applied to the
   queries too, and the index rebuilt.
4. **Re-audit and compare**: `verify_by` says which number should move.
   - Then compare retrieval on real queries before and after: when labels exist pass
     `--qrels` (MCP `qrels_path`) on both runs and `--baseline` on the second.
   - Quote `retrieval_changes` next to the finding changes, and only when
     `retrieval_comparable` is true.
   - Pass the production top-k as `--ranked` to see the index's effect (`overlap_with_exact`).
   - A finding turning `ok` is not proof that search got better, and a change that fixes a
     finding but lowers `ndcg_at_k` is not an improvement.
5. **Adopt only confirmed improvements**; otherwise revert and try the next cause. Measure
   latency and memory alongside quality: a transform that fixes a finding can cost more at
   serving time than the finding did.

Finding-specific notes:

- **anisotropy**: `details.mean_random_cosine_after_centering` shows what centering alone
  would achieve.
  - Near 0 means the model can stay; centering changes rankings, so step 4 matters.
  - On labeled data anisotropy by itself did not lower nDCG@10 (the best public models sit at
    0.58 to 0.82, below the 0.85 warn threshold); it breaks absolute score thresholds, not
    ranking.
  - Suspect the model only when `distance_concentration`, `effective_rank` or
    `rogue_dimensions` move with it.
- **hubness / antihubs**: hubness often follows anisotropy.
  - Fix anisotropy first, then re-rank with local scaling against the queries if hubs remain
    (CSLS: 2 cos(q, d) minus d's mean cosine to its 10 nearest queries).
  - On 21 real pairs it lowered query hub skewness in 19 and raised nDCG@10 on all 4 with
    `query_hubness` not ok (+0.002 to +0.022).
  - Where `query_hubness` was ok it moved nDCG@10 by −0.015 to +0.018, so keep it only if
    labeled queries confirm it.
  - Scaling by corpus neighbors instead raised hub skewness and lowered nDCG@10 on 16 of 21;
    do not use it.
- **collapse**: read `details.largest_near_duplicate_groups` before blaming the model.
  - Many small groups (pairs, triples) are duplicate items: the same image or text under
    several ids; deduplicate or collapse them at serving time.
  - A few huge groups are an embedding failure (a constant vector) or boilerplate.
  - `details.sample_pairs` lists row pairs with their cosine.
  - With a second embedding of the same rows, `compare` settles it: duplicates both spaces
    flag with the same partner belong to the items.
  - Quote `compare`'s `neighbor_scope` with its numbers: `sample` means the neighbors were
    recomputed inside the sampled rows and the agreement is about the sample's own graph (it
    reads higher than the full corpus).
  - Pass full-corpus neighbor lists (`--neighbors-a/-b`, MCP
    `neighbors_a_path`/`neighbors_b_path`) when the question is about the production ranking.
- **outliers / connectivity**: `details.sample_ids` are row ids to show the user.
- **norms / metric_mismatch**: decide the index metric deliberately; normalize if cosine is intended.
- **ann_recall**: if `details.rows_with_fewer_than_k_ids_ratio` is above 0 the ANN file was
  queried with k and included self, which caps recall; re-export with k + 1 before judging the
  index.
  - Then raise `ef_search` / `nprobe` before rebuilding; `vecscope sweep` picks values and
    reports qps per device (GPU is used for IVF when the faiss build has one; HNSW is
    CPU-only).
  - Report the recommendation's `margin`; when it is `tight`, propose the `safer_alternative`
    instead.
  - Relay `caveats`: HNSW recall falls as the corpus grows, so a sweep on a sample is
    optimistic for HNSW (IVF can move either way because nlist follows the row count), and
    qps only ranks settings on this machine.
  - Each point also carries `latency_ms_p50/p95/p99` (single-query, this process) and
    `index_bytes`; use them to weigh a setting's memory and tail latency against its recall.
  - Pass real query traffic as `--queries` whenever a log exists, because corpus-drawn
    queries can pick a different setting.
  - Confirm the pick in the serving environment before fixing it.
- **query_hubness**: skewness grows with corpus size.
  - Good English models measured 1.5 to 3.4 on corpora up to 26k documents and 4.0 to 4.8 on
    58k (FiQA), so a `warn` just above 4 on a large corpus is not by itself a query fault.
  - Compare against a baseline of the same corpus and read it with `hubness`.
  - Values of 11 or more meant a model that does not cover the language.
- **query_gap**: a `fail` most often means the queries were embedded by a different model or
  model version than the corpus.
  - Check that before suspecting the corpus, unless `collapse` is also not ok: near-copies
    shrink the corpus distance the gap divides by, so the plan lists it with the duplicates
    and it is read again after deduplicating.
  - Models fine-tuned from the same base (bge-large and mxbai-embed-large, bge-base and
    gte-base, bge-m3 and its Korean fine-tunes) share a space and can be swapped with little
    nDCG loss, so a low `query_gap` with a mismatched model name is not proof of a fault, and
    confirm with real queries (step 4).
  - It does not detect a missing query prefix (bge, e5): on SciFact and NFCorpus dropping the
    prefix moved nDCG@10 by -0.014 to +0.007 and the distances in either direction, so check
    the prefix in code.
  - A wrong-language model scores normally here when both sides are encoded equally badly;
    `query_hubness` (and `hubness`) catch that case.
- **query_shift**: only present with a reference (CLI `--baseline` from a run with
  `--queries`, MCP `query_baseline_path`).
  - It answers "did today's queries move away from the corpus compared with a run known to be
    healthy"; `details.shifted_sample_ids` are the farthest moved queries.
  - Most queries moved: diff the query pipeline.
  - A few moved: check their topics against the corpus.
  - The reference must come from the same corpus and sample size, otherwise the distances are
    not comparable.
- **ood_queries** (deprecated): same question as `query_gap` with a cutoff that also counts
  the normal query/passage distance.
  - It false-alarmed on 4 of 18 pipelines that reproduce published nDCG.
  - Quote `query_gap` instead.
