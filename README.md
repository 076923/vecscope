<div align="center">

**English** · [한국어](https://github.com/076923/vecscope/blob/main/docs/ko/README.md)

# vecscope

### Embedding space evidence for agents.

**Before an agent trusts a vector index, let it see the space it is searching.**

[![PyPI version](https://img.shields.io/pypi/v/vecscope?style=flat-square&color=3775a9)](https://pypi.org/project/vecscope/)
[![Python](https://img.shields.io/pypi/pyversions/vecscope?style=flat-square)](https://pypi.org/project/vecscope/)
[![CI](https://img.shields.io/github/actions/workflow/status/076923/vecscope/ci.yml?style=flat-square&label=CI)](https://github.com/076923/vecscope/actions/workflows/ci.yml)
[![MIT license](https://img.shields.io/badge/license-MIT-f4c430?style=flat-square)](https://github.com/076923/vecscope/blob/main/LICENSE)

[Quick start](#quick-start) · [What it measures](#what-vecscope-measures) · [Queries and clusters](#queries-and-clusters) · [Index sweep](#index-sweep) · [Compare two embeddings](#compare-two-embeddings) · [Retrieval quality and the fix order](#retrieval-quality-and-the-fix-order) · [Vector stores](#read-from-a-vector-store) · [Use with an agent](#use-with-an-agent) · [Use in CI](#use-in-ci) · [Input formats](#input-formats) · [Real embeddings](#real-embeddings) · [FAQ](#faq)

<code>pip install vecscope</code>

</div>

---

Agents build RAG pipelines, semantic search and recommendation retrieval every day, but they
cannot see the embedding space they are searching.

When retrieval goes wrong it rarely looks like a bug. It looks like:

- "the same three chunks come back for everything"
- "every score is 0.8-something"
- "top-k is full of copies"
- "brute force finds it and HNSW does not"

Each of those failures has a known, deterministic diagnostic in the literature. `vecscope`
turns them into one command and one JSON:

- measurements
- `ok` / `warn` / `fail` verdicts
- an explanation an agent can act on

It runs locally on NumPy, needs no model or network, and reports the same numbers every time.

`vecscope` diagnoses the **vector search implementation**: the geometry the model produced,
the hygiene of the corpus, the distance metric, the index and the query pipeline.

It does not measure relevance. Whether the right documents come back for a question needs
labeled queries, and a clean audit is evidence that search is not sabotaging retrieval, not
proof that retrieval is good.

## Quick start

### 1. Install

```bash
pip install vecscope
```

### 2. Get a file of embeddings

`vecscope` reads a file with one embedding vector per row: the `(n, dim)` array a model's
`encode` returns, saved as `.npy`, `.parquet`, `.jsonl`, `.json` or `.csv`
([all formats](#input-formats)).

```python
import numpy as np
embeddings = model.encode(texts)          # shape (n, dim), e.g. (3000, 128)
np.save("embeddings.npy", embeddings)
```

No embeddings at hand? Generate three synthetic matrices (healthy, anisotropic, collapsed):

```bash
python -c "import urllib.request as u; u.urlretrieve('https://raw.githubusercontent.com/076923/vecscope/main/examples/make_examples.py', 'make_examples.py')"
python make_examples.py
```

### 3. Run the audit

```bash
vecscope audit embeddings.npy
```

```text
vecscope x.y.z  n=3000 dim=128 k=10  verdict=FAIL
  [ok  ] hubness                skewness=1.1661  (ok < 2, warn >= 2, fail >= 4)
                               k-occurrence skewness 1.17; top 1% of items take 3.9% of all neighbor slots (max N_k=51, expected 10).
  [WARN] antihubs               antihub_excess=0.0523  (ok < 0.05, warn >= 0.05, fail >= 0.2)
                               5.2% of items are never anyone's neighbor (random data at k=10 would give 0.0%); the excess is unreachable through k-NN retrieval.
                               try: inspect a sample of never-retrieved items; drop empty inputs, then apply the hubness fix
  [FAIL] anisotropy             mean_random_cosine=0.8732  (ok < 0.5, warn >= 0.5, fail >= 0.85)
                               random pairs have mean cosine 0.873 (isotropic ~0); mean unit vector has norm 0.93; centering would bring the cosine to -0.000.
                               try: subtract the corpus mean vector and re-normalize, applying the same transform to queries; details.mean_random_cosine_after_centering shows the expected result
  [ok  ] effective_rank         effective_rank_ratio=0.5407  (ok >= 0.1, warn < 0.1, fail < 0.05)
                               effective rank 69.2 of 128 attainable (128 dims; 54.1%); low values mean most dimensions carry nothing.
  [ok  ] collapse               near_duplicate_ratio=0.0000  (ok < 0.02, warn >= 0.02, fail >= 0.1)
                               0.0% of items have a neighbor with cosine >= 0.99 or an exact copy (exact duplicates 0.0%); no row has a near-duplicate.
  [ok  ] rogue_dimensions       top1_dim_variance_ratio=1.3070  (ok < 10, warn >= 10, fail >= 50)
                               dimension 94 alone holds 1.0% of variance, 1.3x the uniform share of 0.78%; top 3 dims hold 3.0%. Rogue dimensions dominate cosine.
  [ok  ] norms                  norm_cv=0.0000  (ok < 0.1, warn >= 0.1, fail >= 0.3)
                               vector norms vary with CV 0.000 (median 1.000, range 1.000-1.000); 0.00% are zero vectors. Unequal norms bias inner-product search toward long vectors.
  [ok  ] metric_mismatch        topk_overlap=1.0000  (ok >= 0.9, warn < 0.9, fail < 0.7)
                               cosine and inner-product top-10 share 100.0% of neighbors; 100.0% of items get identical lists. Low overlap means the index metric changes the results.
  [ok  ] distance_concentration contrast_over_noise=2.2784  (ok >= 1.4, warn < 1.4, fail < 1.1)
                               on 3000 rows a random item is 3.27x farther than the nearest neighbor (random distance 0.127); isotropic noise at 128-d and the same rows would give 1.44x, so the contrast is 2.28x that of noise. On all rows the contrast is 3.27x (median NN distance 0.039). Near 1.0 means distances carry no signal.
  [ok  ] outliers               outlier_ratio=0.0030  (ok < 0.01, warn >= 0.01, fail >= 0.05)
                               0.30% of items have no neighbor within cosine distance 0.050 (median 0.039); see details.sample_ids. Typically junk, empty or foreign-language content.
  [ok  ] connectivity           island_ratio=0.0000  (ok < 0.01, warn >= 0.01, fail >= 0.1)
                               0.00% of items sit in 0 small island(s) of the k-NN graph (fewer than 15 items; 60 components in total). Graph-based ANN (HNSW) attaches islands to the wrong region or never reaches them.
```

### 4. Read the result

- The first line gives the overall `verdict`: the worst verdict among the findings.
- Each finding shows the measured value, the range that would be `ok`, and a sentence saying
  what the number means for retrieval. Findings that are not `ok` add a `try:` line with the
  smallest change to test first.
- The exit code is `1` on `FAIL`, so the same command works as a CI gate.
- Add `--json` to get the same evidence as JSON for an agent or a script.

## What vecscope measures

| Finding | Metric | ok | warn | fail | What it means when it fails |
| --- | --- | --- | --- | --- | --- |
| `hubness` | skewness of the k-occurrence distribution | < 2.0 | 2.0 to 4.0 | ≥ 4.0 | a few items appear in everyone's k-NN list and will top every query |
| `antihubs` | share of items never retrieved as a neighbor, above the random baseline e^-k | < 5% | 5 to 20% | ≥ 20% | part of the corpus is unreachable through k-NN |
| `anisotropy` | mean cosine of random pairs | < 0.85 | 0.85 to 0.95 | ≥ 0.95 | vectors crowd into a narrow cone, so scores compress into a band; good retrievers sit well below this without losing nDCG, so only extreme values are flagged |
| `effective_rank` | participation ratio of the covariance spectrum / attainable rank (dim, or rows − 1 when rows do not exceed dims) | ≥ 0.10 | 0.05 to 0.10 | < 0.05 | most dimensions carry nothing |
| `collapse` | share of rows with a neighbor at cosine ≥ 0.99 or an exact copy (zero vectors included); details give the duplicate groups and sample pairs | < 2% | 2 to 10% | ≥ 10% | top-k is spent on copies of the same item |
| `ann_recall` | recall@k of ANN neighbors vs exact k-NN, counted by distance so that exact ties (duplicates) are interchangeable; the tie slack is 1e-6 + 2e-6 x distance, two to three times float32 rounding | ≥ 0.95 | 0.85 to 0.95 | < 0.85 | the index, not the model, is losing results |
| `rogue_dimensions` | variance of the largest dimension as a multiple of the uniform 1/dim share | < 10x | 10 to 50x | ≥ 50x | a few coordinates decide every cosine score |
| `norms` | coefficient of variation of vector norms | < 0.10 | 0.10 to 0.30 | ≥ 0.30 | inner-product search favors long vectors; zero vectors are reported too |
| `metric_mismatch` | top-k overlap between cosine and inner product | ≥ 0.90 | 0.70 to 0.90 | < 0.70 | the index metric changes which results come back |
| `distance_concentration` | relative contrast (random distance / nearest-neighbor distance) divided by that of isotropic noise at the same dimension and row count | ≥ 1.4 | 1.1 to 1.4 | < 1.1 | the nearest neighbor is barely closer than a random item; 1.0 is pure noise |
| `outliers` | share of items with an abnormally far nearest neighbor | < 1% | 1 to 5% | ≥ 5% | junk, empty or foreign items; ids are listed in `details` |
| `connectivity` | share of items in small k-NN graph islands | < 1% | 1 to 10% | ≥ 10% | self-contained groups too small to be clusters; HNSW attaches them badly or never reaches them |
| `query_gap` | median query-to-corpus nearest distance divided by the median corpus-to-corpus nearest distance | < 2.6 | 2.6 to 3.0 | ≥ 3.0 | queries and corpus do not share a space, typically queries embedded by a different model (5 to 10). Pipelines that reproduce published nDCG@10 sit at 0.94 to 2.52; a missing query prefix does not reliably move it |
| `query_shift` | share of queries farther from the corpus than the median + 3 MAD of a reference query set (needs a baseline) | < 20% | 20 to 30% | ≥ 30% | the query pipeline moved since the reference: another model or version, or traffic on topics the corpus does not cover. Healthy query sets score 0 to 9% |
| `ood_queries` | deprecated, read `query_gap`: share of queries farther from the corpus than 99% of corpus items are from each other | < 50% | 50 to 80% | ≥ 80% | same question as `query_gap`, but it counts the normal distance between short queries and long passages too (good models score 0 to 65% on BEIR) and false-alarmed on 4 of 18 reproduced pipelines |
| `query_hubness` | skewness of corpus k-occurrence over query results | < 4.0 | 4.0 to 10.0 | ≥ 10.0 | the same few items answer most queries; the value tracks nDCG@10 across models (Spearman −0.5 to −0.9), wrong-language models score 11 to 30. It grows with corpus size: good English models score 1.5 to 3.4 up to 26k documents and 4.0 to 4.8 on 58k, so on large corpora compare against a baseline |

Every finding in the JSON carries its `healthy_range`, `warn_at` and `fail_at`, plus a `details`
block with the raw numbers. For `anisotropy` that block includes the cosine the same vectors
would have after centering, so the agent can see whether centering alone fixes the problem.

Some findings appear only when you pass their inputs:

- `ann_recall`: an ANN neighbor file with `--ann`
- `query_gap`, `query_hubness` and `ood_queries`: query embeddings with `--queries`
- `query_shift`: query embeddings, and `--baseline` also pointing at an earlier audit run with
  queries

Each finding also carries `likely_causes`, `suggested_action` and `verify_by`. They are
hypotheses, not prescriptions:

- apply the smallest change
- re-audit
- compare retrieval on real queries before adopting it

The [agent skill](https://github.com/076923/vecscope/blob/main/.agents/skills/vecscope/SKILL.md) spells out
that loop.

Thresholds are defaults calibrated on [seven public models](#real-embeddings) and re-checked
against 19 models on nine labeled datasets in two languages ([verification](https://github.com/076923/vecscope/blob/main/docs/verification.md)).
They are a starting point for diagnosis, not a universal quality bar.

Embedding spaces differ by model, language, domain and dimensionality, so a `warn` on a good
model is normal and a clean audit does not guarantee good search.

Override any pair per run:

```bash
vecscope audit vectors.npy --thresholds my_thresholds.json
# my_thresholds.json: {"anisotropy.mean_random_cosine": [0.4, 0.7], "hubness.skewness": [2.5, 5]}
```

The keys are listed in `src/vecscope/evidence.py` (`THRESHOLDS`) and echoed in every JSON
result under `thresholds`.

## Queries and clusters

Pass the query embeddings too and `vecscope` checks whether queries live where the corpus
lives.

- A different model on the query side shows up as `query_gap`.
- The same few chunks answering everything shows up as `query_hubness`, which is also what
  an embedding model that does not cover the corpus language raises.
- With a baseline from a run whose queries were known to be healthy, `query_shift` reports
  how many of today's queries moved away from the corpus compared with then. This also
  catches a partial change (some traffic on another model or on new topics).

```bash
vecscope audit corpus.npy --queries queries.npy
vecscope audit corpus.npy --queries good_queries.npy --json > baseline.json
vecscope audit corpus.npy --queries todays_queries.npy --baseline baseline.json   # adds query_shift
```

A missing query prefix (bge, e5) is not something these findings detect: on SciFact and
NFCorpus it changed nDCG@10 by -0.014 to +0.007 and moved the query distances in either
direction ([verification](https://github.com/076923/vecscope/blob/main/docs/verification.md)).
Check the prefix in the code that embeds queries.

Add `--clusters 10` to append a per-cluster map:

- size
- tightness
- near-duplicate share
- how much of the hub traffic each cluster takes
- sample ids the agent can look up

It carries no verdicts; it tells the agent *where* a problem lives.

```text
  clusters (3):
     id   size  share  cos2ctr  neardup  hubshare  sample_ids
      2   1113  37.1%    0.217     0.0%     37.0%  [1282, 1611, 496, 283, 2041]
      0   1023  34.1%    0.232     0.0%     34.2%  [1855, 2932, 1786, 1025, 2476]
      1    864  28.8%    0.222     0.0%     28.9%  [2505, 647, 2740, 2499, 2965]
```

## Index sweep

`vecscope sweep` builds HNSW and IVF indexes with a small parameter grid and reports, for each
setting on this corpus:

- recall@k against exact k-NN
- build time
- queries per second
- single-query latency percentiles (p50/p95/p99)
- the serialized index size

It then recommends the fastest setting that reaches the target recall (default 0.95) and
reports its `margin` over the target.

A margin under 0.02 marks the recommendation `tight` and names the fastest setting with real
headroom, because recall at fixed parameters falls as the corpus grows: a setting that just
clears the target on a 50k-row sample usually misses it on the full corpus.

It needs the `faiss-cpu` extra.

```bash
pip install "vecscope[index]"
vecscope sweep embeddings.npy -k 10 --target-recall 0.95
```

```text
  index params                                        recall  build s       qps  p50 ms  p95 ms     MiB  dev
 *hnsw  M=8, ef_construction=100, ef_search=16         0.987     0.35    237954   0.034   0.051     0.6  cpu
 *hnsw  M=16, ef_construction=100, ef_search=32        0.998     1.23    165139   0.084   0.108     0.8  cpu
 *ivf   nlist=76, nprobe=64                            0.999     0.14     33251   0.087   0.095     0.4  cpu
  recommendation: hnsw (M=8, ef_construction=100, ef_search=16)  recall=0.987  qps=237954  p95=0.051ms  index=0.6MiB  margin=+0.037  (fastest configuration with recall@5 >= 0.95)
  note: recall is measured on 200 queries drawn from the corpus itself; real query traffic can differ, pass it as queries when available
  note: HNSW recall at fixed parameters falls as the corpus grows, so a sweep on a sample is optimistic for HNSW; IVF nlist is set from the row count (4*sqrt(n)), so its recall at a given nprobe can move either way with size. Re-run at production size before fixing parameters
  note: queries_per_second is the median of 3 runs on this machine with one query batch; it ranks settings against each other and is not a serving throughput
  note: latency_ms_p50/p95/p99 time 200 queries one at a time in this process, with no network, concurrency or serving stack; compare settings with them, do not quote them as service latency
  note: index_bytes is the serialized index (vectors plus graph or lists); the process needs more while building, and a product-quantized index would be smaller
```

Queries default to a seeded sample of the corpus; pass `--queries` to time real queries,
and do so whenever a query log exists: the setting that is fastest at the target recall on
corpus-drawn queries can differ from the one real traffic needs.

- Each setting is timed three times and the median kept (`--timing-repeats`).
- `--latency-queries` (default 200, 0 to skip) are also searched one at a time for the
  percentiles.
- `index_bytes` is the size of the serialized index, so a setting can be judged on memory as
  well as speed.

The notes at the end are part of the JSON as `caveats`. The MCP tool is `sweep_index`, with
the same options as parameters (`n_queries`, `target_recall`, `device`, `margin`,
`timing_repeats`, `latency_queries`).

The sweep answers one question: which parameters reach the target recall on this corpus
and these queries, and what they cost. The numbers are relative to this machine and this
process; confirm the chosen setting in the serving environment with its own latency,
throughput and memory measurements before fixing it.

With a GPU build of faiss (`pip install faiss-gpu-cu12` on Linux) the IVF index and the exact
baseline run on the GPU automatically; the output header names the device. HNSW has no GPU
implementation in faiss and stays on the CPU. Force a device with `--device cpu|gpu`.

## Compare two embeddings

When two models embedded the same items, or one model did before and after a change,
`vecscope compare` tells how far the two spaces agree on who is near whom. It reports:

- how often the nearest neighbor and the top-k lists coincide
- the near-duplicate share in each space and in both
- whether both spaces pick the same duplicate partner
- the rows that disagree most

It carries no verdicts. Its job is to separate item problems from model problems:

- a 10% collapse that both spaces flag with the same partners is a property of the items
  (the same image under several ids, say)
- duplicates that only one space flags point at that model

```bash
vecscope compare clip.npy lens.npy -k 10          # rows must be the same items in the same order
```

```text
vecscope x.y.z  compare  n=3000 dim=128/128 k=10  neighbors=recomputed/full
  nn1_agreement=0.7050  topk_overlap=0.8537  topk_overlap_p10=0.7000  rows_with_no_overlap=0.0000
  near_dup A=0.0400  B=0.0400  both=0.0400  only_A=0.0000  only_B=0.0000  same_nearest_among_both=1.0000
    the two spaces share the nearest neighbor for 70.5% of rows and 85.4% of their top-10 lists
    on average (0.0% of rows share nothing). Near-duplicates: 4.0% in A, 4.0% in B, 4.0% in both,
    and 100.0% of those agree on the partner. Duplicates flagged by both spaces with the same
    partner are a property of the items; duplicates flagged by one space only point at that model.
  largest disagreements (row: nearest in A / nearest in B, top-k overlap):
    1769: 1340 / 1340  0.50
    1810: 1229 / 1229  0.50
```

Read `neighbor_scope` before quoting the agreement.

Above `--sample` rows (default 100k) a seeded subset is compared, and by default the
neighbors are recomputed exactly *inside that subset*: an item whose true neighbors fell
outside the sample is compared on substitutes, which agree more often than the real ones.

On a 2,000-row synthetic pair of spaces, a 500-row sample reports a top-10 overlap of 0.75
where the full corpus gives 0.57. Such a result is marked `neighbor_scope: sample`, and the
explanation says so.

To compare full-corpus rankings on a sample of rows, pass neighbor lists computed on the full
corpus in each space, exact or from the production index (an `(n x >= k)` id file, self ids
and negative fillers are dropped):

```bash
vecscope compare clip.npy lens.npy -k 10 --neighbors-a clip_knn.npy --neighbors-b lens_knn.npy
```

The result is then `neighbors=supplied/full`: the rows may be sampled, their neighbors are
not, and the agreement is between the lists you supplied, exact or approximate as they were
computed. The MCP tool is `compare_embeddings` (`neighbors_a_path`, `neighbors_b_path`).

## Retrieval quality and the fix order

The findings describe the vector space; whether the right documents come back needs labels.

Pass relevance judgments for the queries and the audit adds a `retrieval` block:

- nDCG@k, recall@k, MRR@k and hit rate of exact cosine search over every row (also when the
  findings are sampled)
- `duplicate_result_share`, the share of result slots taken by a near-duplicate of a
  higher-ranked result

`--ranked` scores the top-k your production index returned next to exact search, with
`overlap_with_exact`.

```bash
vecscope audit corpus.npy --queries queries.npy --qrels qrels.tsv
vecscope audit corpus.npy --queries queries.npy --qrels qrels.tsv --ranked prod_topk.npy
```

`qrels` holds one of:

- `query row, corpus row, grade` per line (`.tsv`/`.csv`, a header is skipped)
- `{"query row": {"corpus row": grade}}` in `.json`
- one object per line in `.jsonl`

BEIR's `query-id`/`corpus-id`/`score` names work. Rows are positions in the query and corpus
files.

With `--baseline`, the diff lists `retrieval_changes` next to the finding changes, so a
change is judged by the finding it was meant to fix *and* by retrieval:

```text
  retrieval vs baseline:
    exact   duplicate_result_share   0.0512 -> 0.0000  (-0.0512)
    exact   hit_rate_at_k            1.0000 -> 1.0000  (+0.0000)
    exact   mrr_at_k                 1.0000 -> 1.0000  (+0.0000)
    exact   ndcg_at_k                1.0000 -> 1.0000  (+0.0000)
    exact   recall_at_k              1.0000 -> 1.0000  (+0.0000)
```

`retrieval_comparable` is false when the two runs used different qrels or k.

Every audit also carries a `plan`: the non-ok findings grouped by the cause they likely
share, in the order to fix them. Each step lists:

- its findings with values
- the common cause to check
- the smallest change that tests it
- the numbers to re-measure
- why it comes at that position

```text
  plan (fix in this order; each step is a hypothesis to test):
    1. metric: metric_mismatch=0.0329 fail
       cause: vector norms vary, so cosine and inner product rank differently
       first: decide the index metric; L2-normalize if cosine is intended
       check: norm_cv near 0 and topk_overlap 1.0 on re-audit
    2. shared_direction: anisotropy=0.926 warn, antihubs=0.061 warn, collapse=0.108 fail
       cause: a large component shared by all vectors (a mean offset or a few high-variance coordinates) that compresses scores and creates hubs
       first: subtract the corpus mean and re-normalize (anisotropy.details shows the expected cosine after centering); if rogue_dimensions fails, z-score dimensions instead. Apply the same transform to queries and rebuild the index
       check: anisotropy and rogue_dimensions below warn_at, hubness.skewness down; retrieval ndcg_at_k not lower than before
```

That is a synthetic corpus with a shared offset on unnormalized vectors (2,000 rows, 32 dims);
the diff above is the same corpus after both steps, against the normalized one. Its 80 labeled
queries are easy enough that nDCG stays at 1.0, so only the duplicate share shows the change.

The order follows dependencies between fixes:

- broken inputs (zero vectors, copies) first, because they distort every statistic
- then a dominant coordinate (a rogue dimension decides every distance and with it the norms,
  outliers and hubs, so z-scoring comes before normalizing)
- then the metric
- then transforms such as centering, which rebuild the neighbor graph
- then the model
- then corpus structure and the index
- the query side last, measured against the corrected corpus

A near-duplicate share caused by a shared direction (every cosine above the cutoff, no exact
copies) is grouped with the direction, not with deduplication. A `query_gap` next to
duplicates is listed with them: near-copies shrink the corpus distance the gap divides by, so
it is read again after deduplicating before the queries are blamed. Islands next to a
near-duplicate group smaller than the island size go with the duplicates too, since the
copies may be the islands.

The rules are fixed and listed in `src/vecscope/plan.py`; a step is a hypothesis to test with
its re-measure line, not a diagnosis. Each finding in a step carries `placed_because`: either
"this group's cause covers it", or the other finding whose verdict put it there (for example
"rogue_dimensions is not ok; one coordinate moves this finding too"). The second kind is a
grouping by co-occurrence, so check it with the step's re-measure numbers.

What the fixes did to retrieval on real embeddings is in the
[verification](https://github.com/076923/vecscope/blob/main/docs/verification.md#interventions).

## Read from a vector store

The path can name a faiss index file or a vector database instead of an embedding file.
Rows are read in ascending id order, so two reads of unchanged contents give the same matrix,
and the evidence records the source with a fingerprint of what was read. Row ids in
`details.sample_ids` come with `sample_source_ids`, the store's own ids.

```bash
pip install "vecscope[qdrant]"     # or [chroma], [milvus], [weaviate], [index] for faiss, [stores] for all
vecscope audit index.faiss --ann-from-source --sample 0
vecscope audit "qdrant://localhost:6333/docs?vector=dense" --save-ids ids.json
vecscope audit "qdrant+local:///data/qdrant?collection=docs"
vecscope audit "chroma+local:///data/chroma?collection=docs"
vecscope audit "milvus+local:///data/milvus.db?collection=docs"
vecscope audit "milvus://milvus.internal:19530/docs?api_key_env=MILVUS_TOKEN"
vecscope audit "weaviate://localhost:8080/Docs?grpc_port=50051"
```

`--ann-from-source` queries the store's own index for every stored vector and scores it as
`--ann`, so the recall of the index in production is measured without exporting neighbor
lists.

Options go in the query string:

- `vector=` (named vector, or the Milvus field)
- `https=1`
- `pk=` (Milvus primary key)
- `api_key_env=NAME` (the key is read from that environment variable; keys in the URI itself
  are not used)

Reading is read-only, except that a Milvus collection that is not loaded is loaded for the
read and released again. The MCP server reads local stores under `--allow-dir`, and network
stores only when started with `--allow-network`.

## Large corpora: incremental audits

An audit of every row is quadratic in the rows. When a corpus grows by appending,
`--knn-cache` keeps the exact neighbor lists between runs and computes only what the new
rows change: the new rows against all rows and the old rows against the new ones.

```bash
vecscope audit day1.npy --sample 0 --knn-cache knn.npz    # created: all rows
vecscope audit day2.npy --sample 0 --knn-cache knn.npz    # extended: only the appended rows
```

The cache holds a SHA-256 fingerprint of the rows it was computed on. It is:

- reused for the same rows
- extended when its rows are an unchanged prefix of the input
- rebuilt otherwise (changed or reordered rows, another `dim`)

The neighbor lists equal a full computation except among exactly tied distances, and the
findings agree to float32 rounding. In Python,
`vecscope.incremental.cached_knn_for_audit(x, k, path)` returns the lists for
`audit(x, knn=...)`.

On 100,000 rows x 768 dims (the machine of the [performance](#performance) table):

- appending 5,000 rows took 11 s to extend the cache and 4 s to audit from it, against 94 s
  for a full audit
- appending 20,000 rows took 43 s plus 4 s against 123 s

Every verdict matched the full audit and the largest relative value difference was 7e-5.
Creating the cache costs about what a full audit does (83 s here).

## Use with an agent

`vecscope` ships an MCP server with four read-only tools: `audit_embeddings`, `sweep_index`,
`compare_embeddings` and `describe_findings`. Register it with any MCP client:

```bash
pip install "vecscope[mcp]"
claude mcp add vecscope -- vecscope-mcp --allow-dir /data/embeddings     # Claude Code
```

```json
{ "mcpServers": { "vecscope": { "command": "vecscope-mcp", "args": ["--allow-dir", "/data/embeddings"] } } }
```

The JSON form goes in `.cursor/mcp.json` for Cursor and in `.mcp.json` for a Claude Code
project. Codex reads TOML from `~/.codex/config.toml`:

```toml
[mcp_servers.vecscope]
command = "vecscope-mcp"
args = ["--allow-dir", "/data/embeddings"]
```

The diagnose, change and re-verify loop is tested over the MCP stdio protocol
(`tests/test_agent_workflow.py`), and was run end to end by Claude Code in headless mode
(`claude -p` with the server above).

It audited a corpus with a shared offset, normalized, centered and re-audited as the plan
said, and stopped at the remaining step it was not asked to fix. Cursor and Codex use the
same protocol and were not run.

Then ask:

```text
Audit the embeddings in data/chunks.npy before we tune the retriever, and tell me
which finding to fix first.
```

The tools are read-only, but they read whatever path the agent names. On a machine that holds
data the agent should not see, restrict the server to the directories with embeddings; any
path outside them (after resolving `..` and symlinks) is refused before it is opened:

```json
{ "mcpServers": { "vecscope": { "command": "vecscope-mcp", "args": ["--allow-dir", "/data/embeddings"] } } }
```

What the server may read:

- By default, only the directory it was started in and below.
- `--allow-dir DIR` names the directories instead. It can be repeated; the environment
  variable `VECSCOPE_ALLOWED_DIRS` (paths separated by `:` on Unix, `;` on Windows) does the
  same.
- `--allow-any` (or `VECSCOPE_ALLOW_ANY=1`) lifts the restriction, so every file the server
  process can read is reachable. Use it only on your own machine.
- Network vector stores need `--allow-network` on top; read-only does not make reaching
  them harmless, so grant it only where the server's network access is what you intend.

Each file is read through one handle that is checked after opening: it must be a regular
file, and the same file the path resolves to inside the allowed directories. A symlink
swapped between the check and the read, a FIFO or a directory is refused. Local database
directories (`qdrant+local://` and the like) are checked by path only, because their client
opens the files itself.

The agent skill in [`.agents/skills/vecscope/SKILL.md`](https://github.com/076923/vecscope/blob/main/.agents/skills/vecscope/SKILL.md) tells
the agent when to run the audit and what to do about each verdict: center or swap the model,
deduplicate, re-rank, or rebuild the index. The server also carries those instructions, so an
agent that only has the MCP connection follows the same loop.

- `sweep_index` runs the [index sweep](#index-sweep).
- `compare_embeddings` runs the [comparison](#compare-two-embeddings).
- `describe_findings` lists every finding with its thresholds, causes and actions.

Responses start with a one-line `summary` (the non-ok findings, retrieval nDCG, the first
plan step and, on a re-audit, the comparison with the baseline) and an `audit_id`.

Pass that id as `baseline` when re-auditing after a change: the response then carries
`baseline_diff` (verdict and value changes, `regressions`, `improvements`, retrieval
changes), as `--baseline` does in the CLI, without the agent saving the earlier response. The
server keeps its last 32 audits in memory, and a saved response or `--json` file works too.

`verbose=false` drops the details, explanations and actions of findings that are ok. Agents
without MCP can run the CLI with `--json` and read the same evidence.

If the server stops connecting ("vecscope-mcp not found"), the Python environment it was
installed into has usually been replaced; this happens on hosted notebooks and clusters that
rebuild their environment.

Check with `pip show vecscope` in the interpreter the client uses, reinstall there, and
prefer registering the server through that interpreter so the error names the right
environment:

```json
{ "mcpServers": { "vecscope": { "command": "/path/to/env/bin/python", "args": ["-m", "vecscope.mcp_server"] } } }
```

## Use as a library

```python
import numpy as np
from vecscope import audit

embeddings = np.load("embeddings.npy")    # any (n, dim) float array

ev = audit(embeddings, k=10)
ev.verdict                                # "ok" | "warn" | "fail"
ev.findings[0].value                      # measured value of the first finding
ev.to_dict()                              # the full evidence, same as --json
```

## Use in CI

The JSON output is stable across runs and carries `schema_version`, so it can be committed
as a baseline and diffed on every index rebuild. `--baseline` prints each finding's change
and exits 1 when any verdict got worse (or when `--fail-on` trips), so the pipeline fails on
a regression and not on a `warn` that was already there.

```bash
vecscope audit embeddings.npy --json > baseline.json          # once, reviewed and committed
vecscope audit embeddings.npy --baseline baseline.json        # on every rebuild
```

```text
  baseline baseline.json: verdict ok -> fail  (n, dim, k, sampling, thresholds or near-dup cutoff differ: values and verdicts are not directly comparable)
    hubness                1.0454 -> 0.8594  (-0.1860)  ok -> ok
    antihubs               0.0383 -> 0.0105  (-0.0278)  ok -> ok
    anisotropy             0.0105 -> 0.0127  (+0.0022)  ok -> ok
    collapse               0.0000 -> 0.8000  (+0.8000)  ok -> fail  WORSE
    ...
  regressions: collapse
```

The example diffs `healthy.npy` against `collapsed.npy`, which have different row counts,
hence the note on the first line; a rebuild of the same corpus with the same settings prints
no note.

With `--json` the diff is attached as `baseline_diff`; the library call is
`diff_evidence(new, old)`. A baseline written by an older `schema_version` is rejected with
a message to re-create it.

Audit every row (`--sample 0`) for the baseline and the check, so the two runs see the same
rows. When both runs pass `--queries`, the baseline's query distances also give
`query_shift` (see [Queries and clusters](#queries-and-clusters)); it shows up under
`only_in_new` in the diff because the baseline itself had no reference.

A verdict is a coarse signal: a value can drift toward the threshold for months without
crossing it. Every change row therefore carries:

- `worsened`: the value moved in the bad direction
- `worsening`: by how much, in the metric's units, positive = worse

`--tolerances` turns that drift into a gate independent of the verdict bands:

```bash
# tolerances.json: {"collapse": 0.01, "ann_recall": 0.02, "hubness": 0.3}
vecscope audit embeddings.npy --baseline baseline.json --tolerances tolerances.json --require-comparable
```

- A finding whose value worsens by more than its tolerance is listed under
  `value_regressions` (text: `VALUE WORSE by 0.0300 > 0.01`) and exits 1 even when its
  verdict is unchanged.
- Findings without a tolerance are judged by verdict alone.
- A value that improves is never a regression.

The tolerance is a budget per rebuild against the committed baseline, so re-commit the
baseline when a change was reviewed and accepted.

The run also reports `comparable`: false when n, dim, k, sampling, thresholds or the
near-duplicate cutoff differ from the baseline. `--require-comparable` makes that an error
(exit 2) instead of a note, which is the right policy for a gate, since a diff across
different settings measures the settings, not the corpus. The library call is
`diff_evidence(new, old, tolerances={...})`.

Two different rules can fail a run, and the output keeps them apart:

- The absolute rule is the verdict against the thresholds (`--fail-on fail` by default,
  `--fail-on warn` to be strict).
- The baseline rules are `regressions` (a verdict got worse) and `value_regressions` (a value
  worsened beyond its tolerance).

The defaults are calibrated on public models, so a domain whose healthy corpus sits at `warn`
or `fail` on some finding should not be blocked by the absolute rule at all:
`--fail-on none --baseline baseline.json` judges only the change against the reviewed
baseline.

Every run ends with a `gate` block (JSON) or line (text) that names the rule that decided:

```text
  gate: FAIL; verdict fail reaches --fail-on fail; verdict worse than baseline: collapse
  gate: pass (baseline)
```

Treat the gate as a trend tracker first. Which tolerance is "too much" depends on how the
findings relate to retrieval quality on your own queries.

The [retrieval quality study](#do-the-findings-predict-retrieval-quality) shows that relation
for public datasets, and [calibrating thresholds](#calibrating-thresholds-to-your-domain) says
how to measure it on yours before blocking merges on it.

## CLI options

```bash
vecscope audit vectors.npy -k 20                       # neighbors per item (default 10)
vecscope audit vectors.parquet --json                  # JSON evidence
vecscope audit vectors.npy --ann hnsw_neighbors.npy    # score ANN recall against exact k-NN
vecscope audit vectors.npy --queries queries.npy       # add query-side findings
vecscope audit vectors.npy --queries q.npy --qrels qrels.tsv   # add nDCG/recall/MRR of exact search
vecscope audit vectors.npy --queries q.npy --qrels qrels.tsv --ranked topk.npy   # and of the production ranking
vecscope audit index.faiss --ann-from-source --sample 0     # read a faiss index or store; score its own neighbors
vecscope audit "qdrant://host:6333/docs" --save-ids ids.json   # vector store URI; write ids in row order
vecscope audit vectors.npy --sample 0 --knn-cache knn.npz   # reuse / extend exact neighbor lists between runs
vecscope audit vectors.npy --clusters 10               # append a per-cluster report (at most n/10 clusters)
vecscope audit vectors.npy --thresholds th.json         # override (warn, fail) pairs
vecscope audit vectors.npy --fail-on warn              # exit 1 on WARN as well
vecscope audit vectors.npy --fail-on none --baseline old.json   # ignore the absolute thresholds; fail only on regressions
vecscope audit vectors.npy --baseline old.json         # diff against an earlier --json run; exit 1 on regression
vecscope audit vectors.npy --baseline old.json --tolerances tol.json   # also exit 1 when a value worsens beyond its tolerance
vecscope audit vectors.npy --baseline old.json --require-comparable    # exit 2 when the baseline used different settings
vecscope audit vectors.npy --near-dup 0.98             # cosine threshold for near-duplicates
vecscope audit vectors.npy --sample 0                  # audit every row (default samples 100k)
vecscope audit vectors.npy --sample-repeats 3          # re-audit 3 samples; report value ranges and verdict stability
vecscope sweep vectors.npy --target-recall 0.95        # HNSW/IVF parameter sweep (faiss extra)
vecscope sweep vectors.npy --device gpu                # IVF on the GPU (faiss GPU build)
vecscope sweep vectors.npy --margin 0.05               # headroom under which the pick is marked tight (default 0.02)
vecscope sweep vectors.npy --n-queries 2000 --timing-repeats 5   # queries to time (default 1000), timing runs per setting (default 3)
vecscope sweep vectors.npy --latency-queries 0         # skip the single-query latency percentiles (default 200 queries)
vecscope compare a.npy b.npy -k 10                     # agreement between two embeddings of the same rows
vecscope compare a.npy b.npy --neighbors-a a_knn.npy --neighbors-b b_knn.npy   # full-corpus neighbors on a sample of rows
```

## Input formats

The embedding file holds one vector per row. Every format is converted to a 2-D `float32` array
of shape `(n, dim)` before the audit. Values must be finite: NaN, Inf and magnitudes above 1e16
(whose squares overflow `float32`) are rejected with a message. In the examples `embeddings` is
that array, for instance the output of a model's `encode`:

```python
embeddings = model.encode(texts)        # shape (n, dim), e.g. (3000, 128)
embeddings[0]                           # the vector of the first item
# array([ 0.0312, -0.1187,  0.0459, ...], dtype=float32)
```

| Extension | Expected layout | Example |
| --- | --- | --- |
| `.npy` | a 2-D numeric array | `np.save("vectors.npy", embeddings)` |
| `.npz` | the first array in the archive is used | `np.savez("vectors.npz", emb=embeddings)` |
| `.parquet`, `.pq` | a list-of-float column whose name contains `emb` or `vec`; other columns (ids, text) are ignored | `pd.DataFrame({"id": ids, "embedding": list(embeddings)}).to_parquet("vectors.parquet")` |
| `.jsonl` | one JSON object per line; the first list-valued field is the vector | `{"id": 7, "text": "...", "embedding": [0.1, -0.3, ...]}` |
| `.json` | a JSON array of objects (as above) or an array of lists | `[[0.1, -0.3, ...], [0.2, 0.5, ...]]` |
| `.csv`, `.tsv`, `.txt` | one vector per line, numbers only, no header or id column | `0.1,-0.3,0.8,...` |

The optional `--ann` file (or `ann_path` in MCP) is an integer array of shape `(n, k)`: row `i`
holds the ids of the `k` neighbors an ANN index returned for item `i`, in the same row order as
the embedding file. Any of the formats above works.

Ids refer to row positions, so the file must come from an index built on exactly these rows,
so every row has to be audited: above the `--sample` size (100k rows by default) pass
`--sample 0`, or the audit refuses.

Ids that point at the item itself are ignored, so an index that returns self first should be
queried with k + 1.

The optional `--queries` file (`queries_path` in MCP) holds query embeddings in any of the
same formats, with the same `dim` as the corpus.

The reference for `query_shift` is an earlier `--json` output of an audit run with
`--queries` (`--baseline`, or in MCP `baseline` or `query_baseline_path`).

`--baseline` and `baseline` use it only when it was audited on the same corpus or an
unchanged prefix of it (checked with `corpus_fingerprint`, a SHA-256 of the rows): after
normalizing or centering, the earlier query distances are in another space, so
`query_shift` is skipped and `query_reference_note` says why.

## Performance

Exact k-NN is quadratic in the number of rows: doubling the rows roughly quadruples the time,
and a full audit (`--sample 0`) of 100k rows takes minutes on a laptop.

The k-NN pass works in row blocks sized to keep its distance matrix near 1 GiB whatever the
row count; the other passes need a few float32 copies of the matrix, so the peak grows with
rows x dims at roughly four times the matrix size.

Measured on an 8-vCPU AMD EPYC 7R13 with 8 BLAS threads (NumPy only, single process,
clustered synthetic corpus, k=10); memory is the peak the audit itself allocates, the input
matrix comes on top. A three-run median in separate processes agreed with these single runs
within 2%, and the OS peak memory within 0.04 GiB:

| rows | dim | seconds | peak memory (input excluded) |
|---|---|---|---|
| 10,000 | 384 | 1.3 | 0.40 GiB |
| 10,000 | 768 | 2.0 | 0.41 GiB |
| 10,000 | 1,536 | 3.6 | 0.44 GiB |
| 50,000 | 384 | 17 | 1.63 GiB |
| 50,000 | 768 | 23 | 1.70 GiB |
| 50,000 | 1,536 | 36 | 1.84 GiB |
| 100,000 | 384 | 65 | 1.71 GiB |
| 100,000 | 768 | 86 | 1.85 GiB |
| 100,000 | 1,536 | 130 | 3.45 GiB |
| 200,000 | 768 | 338 | 3.46 GiB |
| 200,000, default 100k sample | 768 | 85 | 2.14 GiB |

Time grows close to n² (3.5 to 3.9x per doubling from 50k up); memory is flat from 50k rows
at 384 and 768 dims because the k-NN block shrinks with n, and steps up at 1,536 dims and at
200k rows. The default 100k-row sample keeps any corpus near the 100k rows.

`--queries` adds one more pass over the corpus per query batch, and `metric_mismatch` adds a
second k-NN pass only when vector norms vary. A sampled audit is a statement about the
sample: use `--sample 0` when you need every island and outlier counted.

`python examples/benchmark_scale.py --rows 50000 100000 --dim 768` re-measures the table on
your machine (wall-clock and the peak NumPy allocates inside the audit) at the sizes you
care about, and `examples/benchmark_sweep.py` does the same for the index sweep; both take
`--corpus` (clustered, isotropic, anisotropic, duplicates, lowrank, zipf) and record the
machine in their JSON.

Blocking bounds memory, not work: the exact pass still costs about n² × dim multiply-adds,
so a million rows at 768 dims is roughly 100 times the 100k-row figure. For corpora that
size, audit the default sample (or several with `--sample-repeats`) and reserve `--sample 0`
for the findings that need every row. Which findings those are:

| finding | what a sample gives | needs every row |
|---|---|---|
| anisotropy, effective_rank, rogue_dimensions, norms, metric_mismatch, distance_concentration | a distribution estimate: within 2.3% of the full value on 200k x 384 with 100k, 50k and 20k samples, and the verdict holds | no |
| hubness, antihubs | a *lower bound*: k-occurrence concentrates with density, so the same corpus scores less hubness on a smaller sample (skewness 2.6 `warn` on all 200k rows, 1.15 on a quarter, 0.16 on a tenth). Verdicts other than `fail` are marked inconclusive | to measure the corpus's own hubness |
| collapse | the duplicate *share* seen inside the sample: a pair counts only when both rows were drawn, so the sampled share is 0.96 to 1.03 x (sample fraction x full share); the verdict uses the estimated full share, at least the exact copies counted over every row | to count or list every near-duplicate |
| outliers | an estimate that over-counts on small samples (3.9x at a tenth); verdicts other than `ok` are marked inconclusive | to find every outlier |
| connectivity | the islands of the sample's own k-NN graph, which miss most islands of the corpus (37% of rows in islands scored 0.08% on a quarter sample); verdicts other than `fail` are marked inconclusive | yes, to judge connectivity at all |
| ann_recall | not available on a sample: ids must line up with the ANN file | yes (`--sample 0`) |
| query_gap, query_shift, query_hubness, ood_queries | reference distances from the sample; every query is scored. query_gap is a ratio of two distances that both grow on a smaller sample; compare query_shift only against a baseline with the same sample size | no |

A sampled audit therefore:

- lists `inconclusive_findings` (the verdicts the table marks inconclusive)
- flags them `INCONCLUSIVE (sampled)` in the text output and names them in the gate line
- takes the collapse verdict from `details.estimated_full_share`: the sampled share divided
  by the sample fraction, or the share of rows with an exact copy counted over every row
  (`details.full_corpus_exact_copy_share`) when that is larger. Counting exact copies over
  the whole corpus is cheap (one digest per row), so rare copies the sample missed still
  count; near-duplicates that are not exact copies are only seen in the sample
- repeats each caveat in the finding's explanation next to the number

The 200k-row figures come from the
[verification](https://github.com/076923/vecscope/blob/main/docs/verification.md);
`tests/test_regression_injection.py` checks the same properties at 6k rows on every run.

Every number is exact for the rows it saw; vecscope never substitutes an approximate
neighbor search for the exact one. The only approximation is the choice of rows.

## How vecscope works

`vecscope` computes exact k-NN once over the matrix and derives every diagnostic from it:

- the k-occurrence distribution gives hubness and anti-hubs
- nearest-neighbor cosines give collapse, outliers and distance concentration
- the k-NN graph gives connectivity
- random pairs and the covariance spectrum give anisotropy, effective rank and rogue
  dimensions
- vector norms give the norm spread
- a second inner-product pass (skipped when norms are uniform) gives metric mismatch
- the ANN neighbor file, if present, is compared against the exact neighbors for recall

Query embeddings, if present, get their own exact search against the corpus for the
query-side findings.

Everything is deterministic: fixed seeds, exact neighbors, no learned components, no network.
Two runs on the same file give the same JSON.

Exact k-NN is quadratic in the number of rows, so inputs above 100,000 rows are audited on a
seeded random sample unless you pass `--sample 0`.

## Real embeddings

Thresholds are calibrated on public, pre-computed embeddings from seven models over eight
datasets: OpenAI ada-002 and text-embedding-3-large, Cohere embed-v3 (Korean and English
Wikipedia), all-MiniLM-L6-v2, bge-m3, mxbai-embed-large-v1 and gte-base-en-v1.5.

The table shows the measured value for the first 40k rows of each dataset; cells without a
label are `ok`. `python examples/real_models.py` reproduces it, no model download needed.

| finding | Cohere v3 (ko wiki) | Cohere v3 (en wiki) | ada-002 (dbpedia) | 3-large (dbpedia) | MiniLM-L6 (wiki) | bge-m3 (wiki) | mxbai-large (wiki) | gte-base (arxiv) |
|---|---|---|---|---|---|---|---|---|
| hubness | 1.47 | 1.16 | 1.56 | 1.21 | 1.59 | 1.85 | 1.43 | 2.28<br>warn |
| antihubs | 0.009 | 0.005 | 0.013 | 0.005 | 0.022 | 0.026 | 0.019 | 0.013 |
| anisotropy | 0.333 | 0.228 | 0.692 | 0.086 | 0.026 | 0.307 | 0.330 | 0.210 |
| effective_rank | 0.158 | 0.178 | 0.090<br>warn | 0.148 | 0.285 | 0.142 | 0.087<br>warn | 0.136 |
| collapse | 0.002 | 0.002 | 0.000 | 0.000 | 0.001 | 0.000 | 0.017 | 0.000 |
| rogue_dimensions | 3.2 | 3.2 | 3.0 | 2.5 | 1.7 | 1.8 | 2.0 | 1.8 |
| norms | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 |
| metric_mismatch | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.952 |
| distance_concentration | 1.72 | 1.74 | 1.74 | 1.43 | 1.78 | 1.84 | 2.21 | 1.62 |
| outliers | 0.001 | 0.001 | 0.000 | 0.000 | 0.001 | 0.002 | 0.003 | 0.000 |
| connectivity | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Re-running the audit on the first 40k rows of each set reproduces all 88 verdicts. The only
`warn`s are hubness on gte-base and `effective_rank` on ada-002 and mxbai-large.

ada-002's well-known anisotropy (0.692) is below the `warn` threshold of 0.85: its mean unit
vector has norm 0.83, so unrelated documents start from a cosine floor near 0.7. Its
successor is isotropic on the same corpus (norm 0.29). Centering brings every model here to
a random-pair cosine of 0.00, which is why the finding reports that number.

Hubness skewness of 1.2 to 1.9 is the normal range for text embeddings, so the `warn`
threshold sits at 2.0; gte-base on arXiv abstracts sits just above it. The anisotropy
thresholds (0.85, 0.95) come from the retrieval-quality study below.

## Do the findings predict retrieval quality?

Partly, and in a specific way. [docs/retrieval_quality.md](https://github.com/076923/vecscope/blob/main/docs/retrieval_quality.md) measures
every finding against nDCG@10 on two labeled BEIR datasets (SciFact, NFCorpus) with exact
cosine search, for eight public models and for controlled defects injected into the best one.
`python examples/retrieval_quality.py` reproduces it.

- **Within one pipeline the findings track damage.** Injecting a rogue dimension, cutting the
  rank, adding noise or duplicating rows moves the watched finding to `fail` (from `ok`, or
  from `warn` where the corpus already carried a few duplicates) while nDCG@10 falls, and the
  drop is monotone in the defect level.

  Queries embedded with an unrelated model score `query_gap` 8 to 10 (`fail`, nDCG@10 near
  0); bge-base queries on the gte-base corpus, which share a space, score 2.9 to 3.9 and keep
  nDCG@10.

  Heavy near-duplication also raises `query_gap`, because the copies shrink the corpus
  distance it divides by; the plan lists it with the duplicates so it is read again after
  deduplicating.
- **Across good models the geometry findings do not rank the models.** The best models on both
  datasets (gte-base, bge, e5) have the *highest* anisotropy (0.58 to 0.82, counting the 19
  models of the [verification](https://github.com/076923/vecscope/blob/main/docs/verification.md)) and ordinary
  effective rank.

  What does separate the weak models is `norms` with `metric_mismatch` (unnormalized outputs)
  and `hubness`, at Spearman |rho| of 0.4 to 0.8 with nDCG@10.
- **Anisotropy alone does not hurt ranking.** Adding a shared offset until the random-pair
  cosine reaches 0.98 leaves nDCG@10 unchanged. It compresses scores, so absolute cosine
  thresholds and score calibration break; `distance_concentration` is the finding that moves
  when the signal itself is lost.

The `anisotropy`, `query_hubness`, `query_gap` and `query_shift` thresholds come from the
[verification](https://github.com/076923/vecscope/blob/main/docs/verification.md): 19
models and injected query faults on three labeled datasets, re-checked on three more.

What it does not cover is how well `warn` and `fail` predict a quality drop in other domains
or on production query logs. Until that is measured for your domain, read a verdict as a
reason to look and a baseline regression as a reason to compare retrieval, not as a quality
score.

### Calibrating thresholds to your domain

The defaults are calibrated on public models and public datasets; a `fail` on your corpus is
a reason to look, not proof that retrieval is bad, and all-`ok` is not proof that the right
documents come back.

Before using a threshold or a tolerance as a blocking gate, measure what it predicts on your
own data:

1. Collect labeled queries from the domain (a few hundred with relevant document ids is
   enough) and split them: one part to pick thresholds, a held-out part to check them.
2. Run `vecscope audit --json` on each candidate state of the pipeline (models, chunkers,
   transforms, index settings) and compute retrieval quality (nDCG@k, Recall@k, MRR) with the
   same queries on the same state. `examples/retrieval_quality.py` does both for the public
   datasets and can be pointed at your own.
3. Relate each finding's value to the quality metric across those states. A finding that does
   not move with quality in your domain should not gate it, whatever its default says.
4. Set `--thresholds` and `--tolerances` from the states where quality was acceptable, using
   only the first split.
5. Confirm on the held-out split that the thresholds separate the acceptable states from the
   rest. A threshold tuned and checked on the same queries will look better than it is.

`vecscope calibrate` does steps 3 to 5 from the audits of step 2. Audit each state with the
fitting queries, and each again with the held-out queries:

```bash
vecscope audit state_a.npy --queries q_fit.npy --qrels qrels_fit.tsv --json > fit_a.json
vecscope audit state_a.npy --queries q_hold.npy --qrels qrels_hold.tsv --json > hold_a.json
# ... the same for every state, then:
vecscope calibrate fit_*.json --min-quality 0.45 --holdout hold_*.json --out th.json
vecscope audit vectors.npy --thresholds th.json
```

For each finding it reports:

- the Spearman correlation with the retrieval metric across the states (`--metric`, nDCG@k by
  default), and whether the sign is the one the finding implies;
- the worst value among acceptable states and the best among the others;
- a (warn, fail) pair only when the finding tracks quality and separates the two groups,
  placed a third and two thirds of the way across the gap.

A finding that does not move with quality on your data, or moves the other way, gets no
pair: keep its default as a reason to look. The holdout line counts the unacceptable states
the proposal catches and the false alarms on acceptable ones. It needs at least three states
on both sides of `--min-quality` together; five or more make the correlations worth reading.

The same procedure applies to a `suggested_action`: it is a hypothesis about the cause, and
centering, standardizing or truncating vectors changes rankings.

- Keep the original vectors and retrieval results as the baseline.
- Apply the transform to a copy.
- Measure quality, latency and memory on both.
- Adopt the change only when the labeled queries improve, not when the finding turns `ok`.

## FAQ

<details>
<summary><strong>Does it send my vectors anywhere?</strong></summary>

No. `vecscope` is NumPy only and runs entirely on your machine. It never loads a model or opens
a network connection.

</details>

<details>
<summary><strong>How large a matrix can it handle?</strong></summary>

Exact k-NN is quadratic in the number of rows. By default `vecscope` audits a seeded random
sample of 100,000 rows when the input is larger, and says so in the output (`sampled: true`,
`n_total`).

A sampled result describes the sample, not the whole corpus: rare clusters, small islands and
a handful of outliers can be missed.

Row ids in `details` (`sample_ids`, `sample_pairs`, the cluster report) always refer to rows
of the input file, also when sampled. Use `--sample 0` to audit everything; see
[performance](#performance) for what that costs.

To see how much the sample itself moves a number, pass `--sample-repeats 3` (MCP:
`sample_repeats`). vecscope then audits three independent seeded samples and attaches
`value_range` (minimum to maximum across samples) and `verdict_stable` to every finding, and
lists `unstable_findings` at the top level.

A finding whose verdict flips between samples should be read as borderline, not as a property
of the corpus. The cost is three audits.

`value_range` is the spread between samples, not an interval that contains the full-corpus
value: hubness is a lower bound on every sample, and even a very stable finding can sit a
little outside the range of three samples.

Not every finding means the same thing on a sample: hubness is understated, outliers
overcounted, islands mostly invisible. The [performance](#performance) table says what each
finding can stand for and which verdicts a sampled audit marks `inconclusive`.

</details>

<details>
<summary><strong>My production model fails the anisotropy check. Is the model broken?</strong></summary>

Not necessarily. Many popular text embedding models are strongly anisotropic by construction.

The finding tells you that raw cosine scores are compressed into a narrow band; centering or
whitening the vectors, or re-ranking, usually restores discrimination. The `details` block
reports the cosine after centering so you can see whether that is enough. The verdict is
evidence for the agent, not a grade for the model.

</details>

<details>
<summary><strong>Which agents can use it?</strong></summary>

Any agent that supports local MCP servers, and any agent that can run a shell command and read
JSON. The skill file describes both paths.

</details>

## Documentation

- [Agent skill](https://github.com/076923/vecscope/blob/main/.agents/skills/vecscope/SKILL.md): when to run the audit and what to do with each verdict
- [Retrieval-quality study](https://github.com/076923/vecscope/blob/main/docs/retrieval_quality.md): every finding against nDCG@10 on labeled data
- [Verification](https://github.com/076923/vecscope/blob/main/docs/verification.md): exactness, recall scoring, the sampling claims, the query findings, interventions, agent integration, and 19 models on 9 labeled datasets
- [Examples](https://github.com/076923/vecscope/tree/main/examples/): synthetic matrices, the calibration script and the retrieval-quality study
- [Changelog](https://github.com/076923/vecscope/blob/main/CHANGELOG.md)

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](https://github.com/076923/vecscope/blob/main/CONTRIBUTING.md) for how to run
the tests and what a good change looks like. Security reports go through
[SECURITY.md](https://github.com/076923/vecscope/blob/main/SECURITY.md).

## License

[MIT](https://github.com/076923/vecscope/blob/main/LICENSE)
