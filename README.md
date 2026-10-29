<div align="center">

**English** · [한국어](https://github.com/076923/vecscope/blob/main/docs/readme/README_ko.md)

# vecscope

### Embedding space evidence for agents.

**Before an agent trusts a vector index, let it see the space it is searching.**

[![PyPI version](https://img.shields.io/pypi/v/vecscope?style=flat-square&color=3775a9)](https://pypi.org/project/vecscope/)
[![Python](https://img.shields.io/pypi/pyversions/vecscope?style=flat-square)](https://pypi.org/project/vecscope/)
[![CI](https://img.shields.io/github/actions/workflow/status/076923/vecscope/ci.yml?style=flat-square&label=CI)](https://github.com/076923/vecscope/actions/workflows/ci.yml)
[![MIT license](https://img.shields.io/badge/license-MIT-f4c430?style=flat-square)](https://github.com/076923/vecscope/blob/main/LICENSE)

[Quick start](#quick-start) · [What it measures](#what-vecscope-measures) · [Queries and clusters](#queries-and-clusters) · [Index sweep](#index-sweep) · [Compare two embeddings](#compare-two-embeddings) · [Use with an agent](#use-with-an-agent) · [Use in CI](#use-in-ci) · [Input formats](#input-formats) · [Real embeddings](#real-embeddings) · [FAQ](#faq)

<code>pip install vecscope</code>

</div>

---

Agents build RAG pipelines, semantic search and recommendation retrieval every day, but they
cannot see the embedding space they are searching. When retrieval goes wrong it rarely looks
like a bug. It looks like "the same three chunks come back for everything", "every score is
0.8-something", "top-k is full of copies", or "brute force finds it and HNSW does not".

Each of those failures has a known, deterministic diagnostic in the literature. `vecscope`
turns them into one command and one JSON: measurements, `ok` / `warn` / `fail` verdicts, and
an explanation an agent can act on. It runs locally on NumPy, needs no model or network, and
reports the same numbers every time.

`vecscope` diagnoses the **vector search implementation**: the geometry the model produced,
the hygiene of the corpus, the distance metric, the index and the query pipeline. It does not
measure relevance. Whether the right documents come back for a question needs labeled queries,
and a clean audit is evidence that search is not sabotaging retrieval, not proof that retrieval
is good.

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
vecscope 0.4.0  n=3000 dim=128 k=10  verdict=FAIL
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
| `anisotropy` | mean cosine of random pairs | < 0.50 | 0.50 to 0.85 | ≥ 0.85 | vectors crowd into a narrow cone, so scores compress into a band |
| `effective_rank` | participation ratio of the covariance spectrum / attainable rank (dim, or rows − 1 when rows do not exceed dims) | ≥ 0.10 | 0.05 to 0.10 | < 0.05 | most dimensions carry nothing |
| `collapse` | share of rows with a neighbor at cosine ≥ 0.99 or an exact copy (zero vectors included); details give the duplicate groups and sample pairs | < 2% | 2 to 10% | ≥ 10% | top-k is spent on copies of the same item |
| `ann_recall` | recall@k of ANN neighbors vs exact k-NN | ≥ 0.95 | 0.85 to 0.95 | < 0.85 | the index, not the model, is losing results |
| `rogue_dimensions` | variance of the largest dimension as a multiple of the uniform 1/dim share | < 10x | 10 to 50x | ≥ 50x | a few coordinates decide every cosine score |
| `norms` | coefficient of variation of vector norms | < 0.10 | 0.10 to 0.30 | ≥ 0.30 | inner-product search favors long vectors; zero vectors are reported too |
| `metric_mismatch` | top-k overlap between cosine and inner product | ≥ 0.90 | 0.70 to 0.90 | < 0.70 | the index metric changes which results come back |
| `distance_concentration` | relative contrast (random distance / nearest-neighbor distance) divided by that of isotropic noise at the same dimension and row count | ≥ 1.4 | 1.1 to 1.4 | < 1.1 | the nearest neighbor is barely closer than a random item; 1.0 is pure noise |
| `outliers` | share of items with an abnormally far nearest neighbor | < 1% | 1 to 5% | ≥ 5% | junk, empty or foreign items; ids are listed in `details` |
| `connectivity` | share of items in small k-NN graph islands | < 1% | 1 to 10% | ≥ 10% | self-contained groups too small to be clusters; HNSW attaches them badly or never reaches them |
| `ood_queries` | share of queries farther from the corpus than 99% of corpus items are from each other | < 5% | 5 to 20% | ≥ 20% | queries and corpus do not share a space: wrong model, missing query prefix, uncovered topics |
| `query_hubness` | skewness of corpus k-occurrence over query results | < 2.0 | 2.0 to 4.0 | ≥ 4.0 | the same few items answer most queries |

Every finding in the JSON carries its `healthy_range`, `warn_at` and `fail_at`, plus a `details`
block with the raw numbers. For `anisotropy` that block includes the cosine the same vectors
would have after centering, so the agent can see whether centering alone fixes the problem.
`ann_recall` appears only when you pass an ANN neighbor file with `--ann`; `ood_queries` and
`query_hubness` only when you pass query embeddings with `--queries`.

Each finding also carries `likely_causes`, `suggested_action` and `verify_by`. They are
hypotheses, not prescriptions: apply the smallest change, re-audit, and compare retrieval on
real queries before adopting it. The [agent skill](https://github.com/076923/vecscope/blob/main/.agents/skills/vecscope/SKILL.md) spells out
that loop.

Thresholds are defaults calibrated on [seven public models](#real-embeddings); they are a
starting point for diagnosis, not a universal quality bar. Embedding spaces differ by model,
language, domain and dimensionality, so a `warn` on a good model is normal and a clean audit
does not guarantee good search. Override any pair per run:

```bash
vecscope audit vectors.npy --thresholds my_thresholds.json
# my_thresholds.json: {"anisotropy.mean_random_cosine": [0.4, 0.7], "hubness.skewness": [2.5, 5]}
```

The keys are listed in `src/vecscope/evidence.py` (`THRESHOLDS`) and echoed in every JSON
result under `thresholds`.

## Queries and clusters

Pass the query embeddings too and `vecscope` checks whether queries live where the corpus
lives. Asymmetric models used without their query prefix, a different model on the query side,
or questions about topics the corpus does not cover all show up as `ood_queries`; the same few
chunks answering everything shows up as `query_hubness`.

```bash
vecscope audit corpus.npy --queries queries.npy
```

Add `--clusters 10` to append a per-cluster map: size, tightness, near-duplicate share, how
much of the hub traffic each cluster takes, and sample ids the agent can look up. It carries
no verdicts; it tells the agent *where* a problem lives.

```text
  clusters (3):
     id   size  share  cos2ctr  neardup  hubshare  sample_ids
      2   1113  37.1%    0.217     0.0%     37.0%  [1282, 1611, 496, 283, 2041]
      0   1023  34.1%    0.232     0.0%     34.2%  [1855, 2932, 1786, 1025, 2476]
      1    864  28.8%    0.222     0.0%     28.9%  [2505, 647, 2740, 2499, 2965]
```

## Index sweep

`vecscope sweep` builds HNSW and IVF indexes with a small parameter grid and reports, for each
setting, recall@k against exact k-NN, build time and queries per second on this corpus. It then
recommends the fastest setting that reaches the target recall (default 0.95) and reports its
`margin` over the target. A margin under 0.02 marks the recommendation `tight` and names the
fastest setting with real headroom, because recall at fixed parameters falls as the corpus
grows: a setting that just clears the target on a 50k-row sample usually misses it on the
full corpus. It needs the `faiss-cpu` extra.

```bash
pip install "vecscope[index]"
vecscope sweep embeddings.npy -k 10 --target-recall 0.95
```

```text
  index params                                       recall  build s       qps  dev
 *hnsw  M=8, ef_construction=100, ef_search=16        0.988     0.49    588131  cpu
 *hnsw  M=16, ef_construction=100, ef_search=32       1.000     0.48    238146  cpu
 *ivf   nlist=76, nprobe=16                           1.000     0.14    182966  cpu
  recommendation: hnsw (M=8, ef_construction=100, ef_search=16)  recall=0.988  qps=588131  margin=+0.038  (fastest configuration with recall@10 >= 0.95)
  note: recall is measured on 1000 queries drawn from the corpus itself; real query traffic can differ, pass it as queries when available
  note: recall at fixed parameters falls as the corpus grows; a sweep on a sample is optimistic for the full corpus, re-run at production size before fixing parameters
  note: queries_per_second is the median of 3 runs on this machine with one query batch; it ranks settings against each other and is not a serving throughput
```

Queries default to a seeded sample of the corpus; pass `--queries` to time real queries. Each
setting is timed three times and the median kept (`--timing-repeats`); the notes at the end
are part of the JSON as `caveats`. The MCP tool is `sweep_index`, with the same options as
parameters (`n_queries`, `target_recall`, `device`, `margin`, `timing_repeats`).

With a GPU build of faiss (`pip install faiss-gpu-cu12` on Linux) the IVF index and the exact
baseline run on the GPU automatically; the output header names the device. HNSW has no GPU
implementation in faiss and stays on the CPU. Force a device with `--device cpu|gpu`.

## Compare two embeddings

When two models embedded the same items, or one model did before and after a change,
`vecscope compare` tells how far the two spaces agree on who is near whom. It reports how
often the nearest neighbor and the top-k lists coincide, the near-duplicate share in each
space and in both, whether both spaces pick the same duplicate partner, and the rows that
disagree most. It carries no verdicts. Its job is to separate item problems from model
problems: a 10% collapse that both spaces flag with the same partners is a property of the
items (the same image under several ids, say); duplicates that only one space flags point at
that model.

```bash
vecscope compare clip.npy lens.npy -k 10          # rows must be the same items in the same order
```

```text
vecscope 0.4.0  compare  n=3000 dim=128/128 k=10
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

The MCP tool is `compare_embeddings`.

## Use with an agent

`vecscope` ships an MCP server with four read-only tools: `audit_embeddings`, `sweep_index`,
`compare_embeddings` and `describe_findings`. Register it with any MCP client (Claude Code,
Cursor, Codex, ...):

```bash
pip install "vecscope[mcp]"
```

```json
{ "mcpServers": { "vecscope": { "command": "vecscope-mcp" } } }
```

Then ask:

```text
Audit the embeddings in data/chunks.npy before we tune the retriever, and tell me
which finding to fix first.
```

The agent skill in [`.agents/skills/vecscope/SKILL.md`](https://github.com/076923/vecscope/blob/main/.agents/skills/vecscope/SKILL.md) tells
the agent when to run the audit and what to do about each verdict: center or swap the model,
deduplicate, re-rank, or rebuild the index. The server also carries those instructions, so an
agent that only has the MCP connection follows the same loop. `sweep_index` runs the
[index sweep](#index-sweep), `compare_embeddings` the [comparison](#compare-two-embeddings);
`describe_findings` lists every finding with its thresholds, causes and actions. Responses
start with a one-line `summary`; pass `verbose=false` to drop details for findings that are
ok. Agents without MCP can run the CLI with `--json` and read the same evidence.

If the server stops connecting ("vecscope-mcp not found"), the Python environment it was
installed into has usually been replaced; this happens on hosted notebooks and clusters that
rebuild their environment. Check with `pip show vecscope` in the interpreter the client uses,
reinstall there, and prefer registering the server through that interpreter so the error
names the right environment:

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
no note. With `--json` the diff is attached as `baseline_diff`; the library call is
`diff_evidence(new, old)`. A baseline written by an older `schema_version` is rejected with
a message to re-create it. Audit every row (`--sample 0`) for the baseline and the check, so
the two runs see the same rows.

## CLI options

```bash
vecscope audit vectors.npy -k 20                       # neighbors per item (default 10)
vecscope audit vectors.parquet --json                  # JSON evidence
vecscope audit vectors.npy --ann hnsw_neighbors.npy    # score ANN recall against exact k-NN
vecscope audit vectors.npy --queries queries.npy       # add query-side findings
vecscope audit vectors.npy --clusters 10               # append a per-cluster report (at most n/10 clusters)
vecscope audit vectors.npy --thresholds th.json         # override (warn, fail) pairs
vecscope audit vectors.npy --fail-on warn              # exit 1 on WARN as well
vecscope audit vectors.npy --baseline old.json         # diff against an earlier --json run; exit 1 on regression
vecscope audit vectors.npy --near-dup 0.98             # cosine threshold for near-duplicates
vecscope audit vectors.npy --sample 0                  # audit every row (default samples 100k)
vecscope audit vectors.npy --sample-repeats 3          # re-audit 3 samples; report value ranges and verdict stability
vecscope sweep vectors.npy --target-recall 0.95        # HNSW/IVF parameter sweep (faiss extra)
vecscope sweep vectors.npy --device gpu                # IVF on the GPU (faiss GPU build)
vecscope sweep vectors.npy --margin 0.05               # headroom under which the pick is marked tight (default 0.02)
vecscope sweep vectors.npy --n-queries 2000 --timing-repeats 5   # queries to time (default 1000), timing runs per setting (default 3)
vecscope compare a.npy b.npy -k 10                     # agreement between two embeddings of the same rows
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
the embedding file. Any of the formats above works. Ids refer to row positions, so the file
must come from an index built on exactly these rows, so every row has to be audited: above the
`--sample` size (100k rows by default) pass `--sample 0`, or the audit refuses. Ids that point
at the item itself are ignored, so an index that
returns self first should be queried with k + 1.

The optional `--queries` file (`queries_path` in MCP) holds query embeddings in any of the
same formats, with the same `dim` as the corpus.

## Performance

Exact k-NN is quadratic in the number of rows: doubling the rows roughly quadruples the time,
and a full audit (`--sample 0`) of 100k rows takes minutes on a laptop. The k-NN pass works in
row blocks sized to keep its distance matrix near 1 GiB whatever the row count; the other
passes need a few float32 copies of the matrix, so the peak grows with rows x dims at roughly
four times the matrix size. The peak tracked by Python for the audit itself (NumPy only, single
process; the process needs the input matrix on top of that):

| rows | dim | peak memory |
|---|---|---|
| 10,000 | 384 | 0.4 GiB |
| 10,000 | 1,536 | 0.4 GiB |
| 50,000 | 384 | 1.6 GiB |
| 50,000 | 1,536 | 1.8 GiB |
| 100,000 | 768 | 1.9 GiB |

The default 100k-row sample keeps any corpus inside the last row. `--queries` adds one more
pass over the corpus per query batch, and
`metric_mismatch` adds a second k-NN pass only when vector norms vary. A sampled audit is a
statement about the sample: use `--sample 0` when you need every island and outlier counted.

## How vecscope works

`vecscope` computes exact k-NN once over the matrix and derives every diagnostic from it:
the k-occurrence distribution gives hubness and anti-hubs, nearest-neighbor cosines give
collapse, outliers and distance concentration, the k-NN graph gives connectivity, random pairs
and the covariance spectrum give anisotropy, effective rank and rogue dimensions, vector norms
give the norm spread, a second inner-product pass (skipped when norms are uniform) gives
metric mismatch, and the ANN neighbor file, if present, is compared against the exact
neighbors for recall. Query embeddings, if present, get their own exact search against the
corpus for the query-side findings.

Everything is deterministic: fixed seeds, exact neighbors, no learned components, no network.
Two runs on the same file give the same JSON. Exact k-NN is quadratic in the number of rows,
so inputs above 100,000 rows are audited on a seeded random sample unless you pass `--sample 0`.

## Real embeddings

Thresholds are calibrated on public, pre-computed embeddings from seven models over eight
datasets: OpenAI ada-002 and text-embedding-3-large, Cohere embed-v3 (Korean and English
Wikipedia), all-MiniLM-L6-v2, bge-m3, mxbai-embed-large-v1 and gte-base-en-v1.5. The table
shows the measured value for the first 40k rows of each dataset; cells without a label are
`ok`. `python examples/real_models.py`
reproduces it, no model download needed.

| finding | Cohere v3 (ko wiki) | Cohere v3 (en wiki) | ada-002 (dbpedia) | 3-large (dbpedia) | MiniLM-L6 (wiki) | bge-m3 (wiki) | mxbai-large (wiki) | gte-base (arxiv) |
|---|---|---|---|---|---|---|---|---|
| hubness | 1.47 | 1.16 | 1.56 | 1.21 | 1.59 | 1.85 | 1.43 | 2.28<br>warn |
| antihubs | 0.009 | 0.005 | 0.013 | 0.005 | 0.022 | 0.026 | 0.019 | 0.013 |
| anisotropy | 0.333 | 0.228 | 0.692<br>warn | 0.086 | 0.026 | 0.307 | 0.330 | 0.210 |
| effective_rank | 0.158 | 0.178 | 0.090<br>warn | 0.148 | 0.285 | 0.143 | 0.087<br>warn | 0.136 |
| collapse | 0.002 | 0.002 | 0.000 | 0.000 | 0.001 | 0.000 | 0.017 | 0.000 |
| rogue_dimensions | 3.2 | 3.2 | 3.0 | 2.5 | 1.7 | 1.8 | 2.0 | 1.8 |
| norms | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 |
| metric_mismatch | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.952 |
| distance_concentration | 1.73 | 1.74 | 1.74 | 1.43 | 1.78 | 1.84 | 2.21 | 1.62 |
| outliers | 0.001 | 0.001 | 0.000 | 0.000 | 0.001 | 0.002 | 0.003 | 0.000 |
| connectivity | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Besides hubness on gte-base and `effective_rank` on ada-002 and mxbai-large, the only `warn`
is ada-002's well-known anisotropy: its mean unit vector has norm 0.83, so unrelated documents
start from a cosine floor near 0.7. Its successor is
isotropic on the same corpus (norm 0.29). Centering brings every model here to a random-pair
cosine of 0.00, which is why the finding reports that number. Hubness skewness of 1.2 to 1.9
is the normal range for text embeddings, so the `warn` threshold sits at 2.0; gte-base on
arXiv abstracts sits just above it. The anisotropy thresholds (0.5, 0.85) come from the
retrieval-quality study below, where the best models sit between 0.58 and 0.81.

## Do the findings predict retrieval quality?

Partly, and in a specific way. [docs/retrieval_quality.md](https://github.com/076923/vecscope/blob/main/docs/retrieval_quality.md) measures
every finding against nDCG@10 on two labeled BEIR datasets (SciFact, NFCorpus) with exact
cosine search, for eight public models and for controlled defects injected into the best one.
`python examples/retrieval_quality.py` reproduces it.

- **Within one pipeline the findings track damage.** Injecting a rogue dimension, cutting the
  rank, adding noise or duplicating rows moves the watched finding to `fail` (from `ok`, or
  from `warn` where the corpus already carried a few duplicates) while nDCG@10 falls, and the
  drop is monotone in the defect level. Queries embedded with the wrong
  model score `ood_queries` = 1.0.
- **Across good models the geometry findings do not rank the models.** The best models on both
  datasets (gte-base, bge, e5) have the *highest* anisotropy (0.58 to 0.81) and ordinary
  effective rank. What does separate the weak models is `norms` with `metric_mismatch`
  (unnormalized outputs) and `hubness`, at Spearman |rho| of 0.4 to 0.8 with nDCG@10.
- **Anisotropy alone does not hurt ranking.** Adding a shared offset until the random-pair
  cosine reaches 0.98 leaves nDCG@10 unchanged. It compresses scores, so absolute cosine
  thresholds and score calibration break; `distance_concentration` is the finding that moves
  when the signal itself is lost.

The `anisotropy` and `distance_concentration` thresholds were recalibrated from this study.

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
`n_total`). A sampled result describes the sample, not the whole corpus: rare clusters, small
islands and a handful of outliers can be missed. Row ids in `details` (`sample_ids`,
`sample_pairs`, the cluster report) always refer to rows of the input file, also when sampled.
Use `--sample 0` to audit everything; see [performance](#performance) for what that costs.

To see how much the sample itself moves a number, pass `--sample-repeats 3` (MCP:
`sample_repeats`). vecscope then audits three independent seeded samples and attaches
`value_range` (minimum to maximum across samples) and `verdict_stable` to every finding, and
lists `unstable_findings` at the top level. A finding whose verdict flips between samples
should be read as borderline, not as a property of the corpus. The cost is three audits.

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
- [Examples](https://github.com/076923/vecscope/tree/main/examples/): synthetic matrices, the calibration script and the retrieval-quality study
- [Changelog](https://github.com/076923/vecscope/blob/main/CHANGELOG.md)

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](https://github.com/076923/vecscope/blob/main/CONTRIBUTING.md) for how to run
the tests and what a good change looks like. Security reports go through
[SECURITY.md](https://github.com/076923/vecscope/blob/main/SECURITY.md).

## License

[MIT](https://github.com/076923/vecscope/blob/main/LICENSE)
