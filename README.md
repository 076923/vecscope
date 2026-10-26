<div align="center">

**English** · [한국어](docs/readme/README_ko.md)

# vecscope

### Embedding space evidence for agents.

**Before an agent trusts a vector index, let it see the space it is searching.**

[![PyPI version](https://img.shields.io/pypi/v/vecscope?style=flat-square&color=3775a9)](https://pypi.org/project/vecscope/)
[![Python](https://img.shields.io/pypi/pyversions/vecscope?style=flat-square)](https://pypi.org/project/vecscope/)
[![CI](https://img.shields.io/github/actions/workflow/status/076923/vecscope/ci.yml?style=flat-square&label=CI)](https://github.com/076923/vecscope/actions/workflows/ci.yml)
[![MIT license](https://img.shields.io/badge/license-MIT-f4c430?style=flat-square)](LICENSE)

[Quick start](#quick-start) · [What it measures](#what-vecscope-measures) · [Queries and clusters](#queries-and-clusters) · [Index sweep](#index-sweep) · [Use with an agent](#use-with-an-agent) · [Input formats](#input-formats) · [Real embeddings](#real-embeddings) · [FAQ](#faq)

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
vecscope 0.2.0  n=3000 dim=128 k=10  verdict=FAIL
  [FAIL] hubness          skewness=6.9023  (ok < 2, warn >= 2, fail >= 4)
                         k-occurrence skewness 6.90; top 1% of items take 16.3% of all neighbor slots (max N_k=356, expected 10).
                         try: re-rank top-k with mutual k-NN or local scaling; if anisotropy also fails, center first
  [FAIL] antihubs         antihub_excess=0.2180  (ok < 0.05, warn >= 0.05, fail >= 0.2)
                         21.8% of items are never anyone's neighbor (random data at k=10 would give 0.0%); the excess is unreachable through k-NN retrieval.
                         try: inspect a sample of never-retrieved items; drop empty inputs, then apply the hubness fix
  [FAIL] anisotropy       mean_random_cosine=0.9782  (ok < 0.3, warn >= 0.3, fail >= 0.6)
                         random pairs have mean cosine 0.978 (isotropic ~0); mean unit vector has norm 0.99; centering would bring the cosine to -0.000.
                         try: subtract the corpus mean vector and re-normalize, applying the same transform to queries; details.mean_random_cosine_after_centering shows the expected result
  [ok  ] effective_rank   effective_rank_ratio=0.9521  (ok >= 0.1, warn < 0.1, fail < 0.05)
                         effective rank 121.9 of 128 dims (95.2%); low values mean most dimensions carry nothing.
  [ok  ] collapse         near_duplicate_ratio=0.0000  (ok < 0.02, warn >= 0.02, fail >= 0.1)
                         0.0% of items have a neighbor with cosine >= 0.99 (exact duplicates 0.0%).
  [ok  ] rogue_dimensions top1_dim_variance_ratio=1.0762  (ok < 10, warn >= 10, fail >= 50)
                         dimension 7 alone holds 0.8% of variance, 1.1x the uniform share of 0.78%; top 3 dims hold 2.5%. Rogue dimensions dominate cosine.
  [ok  ] norms            norm_cv=0.0000  (ok < 0.1, warn >= 0.1, fail >= 0.3)
                         vector norms vary with CV 0.000 (median 1.000, range 1.000-1.000); 0.00% are zero vectors. Unequal norms bias inner-product search toward long vectors.
  [ok  ] metric_mismatch  topk_overlap=1.0000  (ok >= 0.9, warn < 0.9, fail < 0.7)
                         cosine and inner-product top-10 share 100.0% of neighbors; 100.0% of items get identical lists. Low overlap means the index metric changes the results.
  [ok  ] distance_concentration relative_contrast=1.5260  (ok >= 1.2, warn < 1.2, fail < 1.05)
                         a random item is only 1.53x farther than the nearest neighbor (median NN distance 0.014, random 0.022). Near 1.0 means distances carry no signal.
  [ok  ] outliers         outlier_ratio=0.0037  (ok < 0.01, warn >= 0.01, fail >= 0.05)
                         0.37% of items have no neighbor within cosine distance 0.018 (median 0.014); see details.sample_ids. Typically junk, empty or foreign-language content.
  [ok  ] connectivity     island_ratio=0.0000  (ok < 0.01, warn >= 0.01, fail >= 0.1)
                         0.00% of items sit in 0 small island(s) of the k-NN graph (fewer than 15 items; 1 components in total). Graph-based ANN (HNSW) attaches islands to the wrong region or never reaches them.
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
| `anisotropy` | mean cosine of random pairs | < 0.30 | 0.30 to 0.60 | ≥ 0.60 | vectors crowd into a narrow cone, so every pair looks similar |
| `effective_rank` | participation ratio / dim | ≥ 0.10 | 0.05 to 0.10 | < 0.05 | most dimensions carry nothing |
| `collapse` | near-duplicate ratio at cosine ≥ 0.99 | < 2% | 2 to 10% | ≥ 10% | top-k is spent on copies of the same item |
| `ann_recall` | recall@k of ANN neighbors vs exact k-NN | ≥ 0.95 | 0.85 to 0.95 | < 0.85 | the index, not the model, is losing results |
| `rogue_dimensions` | variance of the largest dimension as a multiple of the uniform 1/dim share | < 10x | 10 to 50x | ≥ 50x | a few coordinates decide every cosine score |
| `norms` | coefficient of variation of vector norms | < 0.10 | 0.10 to 0.30 | ≥ 0.30 | inner-product search favors long vectors; zero vectors are reported too |
| `metric_mismatch` | top-k overlap between cosine and inner product | ≥ 0.90 | 0.70 to 0.90 | < 0.70 | the index metric changes which results come back |
| `distance_concentration` | random distance / nearest-neighbor distance | ≥ 1.2 | 1.05 to 1.2 | < 1.05 | the nearest neighbor is barely closer than a random item |
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
real queries before adopting it. The [agent skill](.agents/skills/vecscope/SKILL.md) spells out
that loop.

Thresholds are defaults calibrated on [eight public models](#real-embeddings); they are a
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
      1   1014  33.8%    0.111     0.0%     34.5%  [1533, 2934, 800, 1585, 824]
      2   1001  33.4%    0.109     0.0%     33.4%  [2550, 1171, 4, 1496, 668]
      0    985  32.8%    0.111     0.0%     32.0%  [1910, 2516, 2859, 124, 2392]
```

## Index sweep

`vecscope sweep` builds HNSW and IVF indexes with a small parameter grid and reports, for each
setting, recall@k against exact k-NN, build time and queries per second on this corpus. It then
recommends the fastest setting that reaches the target recall (default 0.95). It needs the
`faiss-cpu` extra.

```bash
pip install "vecscope[index]"
vecscope sweep embeddings.npy -k 10 --target-recall 0.95
```

```text
  index params                                       recall  build s       qps
 *hnsw  M=16, ef_construction=100, ef_search=32       0.994     0.31    141532
  ivf   nlist=76, nprobe=32                           0.772     0.02     70925
 *ivf   nlist=76, nprobe=64                           0.975     0.02     41427
  recommendation: hnsw (M=32, ef_construction=100, ef_search=16)  recall=0.985  qps=128425
```

Queries default to a seeded sample of the corpus; pass `--queries` to time real queries. The
MCP tool is `sweep_index`.

With a GPU build of faiss (`pip install faiss-gpu-cu12` on Linux) the IVF index and the exact
baseline run on the GPU automatically; the output header names the device. HNSW has no GPU
implementation in faiss and stays on the CPU. Force a device with `--device cpu|gpu`.

## Use with an agent

`vecscope` ships an MCP server with three read-only tools: `audit_embeddings`, `sweep_index`
and `describe_findings`. Register it with any MCP client (Claude Code, Cursor, Codex, ...):

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

The agent skill in [`.agents/skills/vecscope/SKILL.md`](.agents/skills/vecscope/SKILL.md) tells
the agent when to run the audit and what to do about each verdict: center or swap the model,
deduplicate, re-rank, or rebuild the index. The server also carries those instructions, so an
agent that only has the MCP connection follows the same loop. `sweep_index` runs the
[index sweep](#index-sweep); `describe_findings` lists every finding with its thresholds,
causes and actions. Responses start with a one-line `summary`; pass `verbose=false` to drop
details for findings that are ok. Agents without MCP can run the CLI with `--json` and read
the same evidence.

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

## CLI options

```bash
vecscope audit vectors.npy -k 20                       # neighbors per item (default 10)
vecscope audit vectors.parquet --json                  # JSON evidence
vecscope audit vectors.npy --ann hnsw_neighbors.npy    # score ANN recall against exact k-NN
vecscope audit vectors.npy --queries queries.npy       # add query-side findings
vecscope audit vectors.npy --clusters 10               # append a per-cluster report
vecscope audit vectors.npy --thresholds th.json         # override (warn, fail) pairs
vecscope audit vectors.npy --fail-on warn              # exit 1 on WARN as well
vecscope audit vectors.npy --near-dup 0.98             # cosine threshold for near-duplicates
vecscope audit vectors.npy --sample 0                  # audit every row (default samples 100k)
vecscope sweep vectors.npy --target-recall 0.95        # HNSW/IVF parameter sweep (faiss extra)
vecscope sweep vectors.npy --device gpu                # IVF on the GPU (faiss GPU build)
```

## Input formats

The embedding file holds one vector per row. Every format is converted to a 2-D `float32` array
of shape `(n, dim)` before the audit. In the examples `embeddings` is that array, for instance
the output of a model's `encode`:

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
must come from an index built on exactly these rows, so `--ann` requires `--sample 0` (the audit
refuses to combine the two). Ids that point at the item itself are ignored, so an index that
returns self first should be queried with k + 1.

The optional `--queries` file (`queries_path` in MCP) holds query embeddings in any of the
same formats, with the same `dim` as the corpus.

## Performance

Exact k-NN is quadratic in the number of rows, so the full audit (`--sample 0`) costs this
on a 6-core laptop (Intel i7-8750H, NumPy only, single process). Memory is the peak tracked by
Python for the audit itself; the process needs the input matrix on top of that.

| rows | dim | time | peak memory |
|---|---|---|---|
| 10,000 | 384 | 16 s | 0.4 GiB |
| 10,000 | 1,536 | 53 s | 0.4 GiB |
| 50,000 | 384 | 87 s | 2.0 GiB |
| 50,000 | 1,536 | 109 s | 2.2 GiB |
| 100,000 | 768 | 191 s | 4.1 GiB |

The default 100k-row sample keeps any corpus inside the last row. Doubling the rows roughly
quadruples the time; `--queries` adds one more pass over the corpus per query batch, and
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

Thresholds are calibrated on public, pre-computed embeddings from eight models: OpenAI
ada-002 and text-embedding-3-large, Cohere embed-v3, all-MiniLM-L6-v2, bge-m3,
mxbai-embed-large-v1 and gte-base-en-v1.5. The table shows the measured value for the first 40k
rows of each dataset; cells without a label are `ok`. `python examples/real_models.py`
reproduces it, no model download needed.

| finding | Cohere v3 (ko wiki) | Cohere v3 (en wiki) | ada-002 (dbpedia) | 3-large (dbpedia) | MiniLM-L6 (wiki) | bge-m3 (wiki) | mxbai-large (wiki) | gte-base (arxiv) |
|---|---|---|---|---|---|---|---|---|
| hubness | 1.47 | 1.16 | 1.56 | 1.21 | 1.59 | 1.85 | 1.43 | 2.28<br>warn |
| antihubs | 0.009 | 0.005 | 0.013 | 0.005 | 0.022 | 0.026 | 0.019 | 0.013 |
| anisotropy | 0.333<br>warn | 0.228 | 0.692<br>**fail** | 0.086 | 0.026 | 0.307<br>warn | 0.330<br>warn | 0.210 |
| effective_rank | 0.158 | 0.178 | 0.090<br>warn | 0.148 | 0.285 | 0.143 | 0.087<br>warn | 0.136 |
| collapse | 0.002 | 0.002 | 0.000 | 0.000 | 0.001 | 0.000 | 0.017 | 0.000 |
| rogue_dimensions | 3.2 | 3.2 | 3.0 | 2.5 | 1.7 | 1.8 | 2.0 | 1.8 |
| norms | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 |
| metric_mismatch | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.952 |
| distance_concentration | 3.00 | 3.00 | 2.53 | 1.86 | 2.93 | 2.77 | 3.99 | 2.30 |
| outliers | 0.001 | 0.001 | 0.000 | 0.000 | 0.001 | 0.002 | 0.003 | 0.000 |
| connectivity | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

ada-002's well-known anisotropy shows up as the only `fail`: its mean unit vector has norm 0.83,
so unrelated documents start from a cosine floor near 0.7. Its successor is isotropic on the
same corpus (norm 0.29). Centering brings all three models to a random-pair cosine of 0.00, which
is why the finding reports that number. Hubness skewness of 1.2 to 1.9 is the normal range for
text embeddings, so the `warn` threshold sits at 2.0; gte-base on arXiv abstracts sits just
above it. Three good models land in the anisotropy `warn` band (0.30 to 0.33): mild, and
their `details` show that centering removes it.

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
islands and a handful of outliers can be missed. Use `--sample 0` to audit everything; see
[performance](#performance) for what that costs.

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

- [Agent skill](.agents/skills/vecscope/SKILL.md): when to run the audit and what to do with each verdict
- [Examples](examples/): synthetic matrices and the real-embedding calibration script
- [Changelog](CHANGELOG.md)

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to run
the tests and what a good change looks like. Security reports go through
[SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
