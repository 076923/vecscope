**English** · [한국어](ko/usage.md)

# Usage

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
- IVF's nlist follows the row count (4*sqrt(n)) unless `--nlist` fixes it. To tune nprobe for
  an existing index, pass that index's nlist.

The notes at the end are part of the JSON as `caveats`. The MCP tool is `sweep_index`, with
the same options as parameters (`n_queries`, `target_recall`, `device`, `margin`,
`timing_repeats`, `latency_queries`, `nlist`).

The sweep answers one question: which parameters reach the target recall on this corpus
and these queries, and what they cost. The numbers are relative to this machine and this
process; confirm the chosen setting in the serving environment with its own latency,
throughput and memory measurements before fixing it.

With a GPU build of faiss (`pip install "vecscope[index-gpu]"`, Meta's `faiss-gpu` for Linux x86_64
with CUDA 12.6 or later) the IVF index and the exact
baseline run on the GPU automatically; the output header names the device. HNSW has no GPU
implementation in faiss and stays on the CPU. Force a device with `--device cpu|gpu`.
`index` and `index-gpu` both install the `faiss` module, so install only one of them; `stores`
includes `index`, so on a GPU machine pick the store extras one by one
(`"vecscope[index-gpu,qdrant]"`). On an older driver, install the community builds
`faiss-gpu-cu12` (CUDA 12.1) or `faiss-gpu-cu11` yourself instead of `index-gpu`.

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

## Read from a vector store

The path can name a faiss index file or a vector database instead of an embedding file.

- Rows are read in ascending id order, so two reads of unchanged contents give the same
  matrix.
- The evidence records the source with a fingerprint of what was read.
- Row ids in `details.sample_ids` come with `sample_source_ids`, the store's own ids.

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
- `timeout=` seconds per request to a network store (default 60; a finite positive number,
  at most 3600)

Reading is read-only, except that a Milvus collection that is not loaded is loaded for the
read and released again.

Secrets stay out of the output:

- The evidence records the URI without user info, and with every query value other than the
  options above replaced by `REDACTED`.
- An error from a store client is re-raised with the API key and those values scrubbed.
- Chroma's HTTP client takes no timeout of its own, so `timeout=` is set on its session after
  it connects; the connection check the client makes when it is created has none.

The MCP server reads local stores inside its [allowed directories](mcp.md#what-the-server-may-read),
and network stores only when started with `--allow-network`.

## Large corpora: incremental audits

An audit of every row is quadratic in the rows. When a corpus grows by appending or has
rows re-embedded in place, `--knn-cache` keeps the exact neighbor lists between runs and
computes only what those rows change.

```bash
vecscope audit day1.npy --sample 0 --knn-cache knn.npz    # created: all rows
vecscope audit day2.npy --sample 0 --knn-cache knn.npz    # extended: only the appended rows
vecscope audit day3.npy --sample 0 --knn-cache knn.npz    # updated: changed rows and their neighbors
```

The cache holds a digest of every row it was computed on. It is:

- reused for the same rows
- extended when rows were appended and the cached rows are unchanged
- updated when some cached rows hold new vectors: those rows, and the rows whose neighbor
  list contained one of them, are recomputed against every row; every other row only looks
  at the changed and appended rows
- rebuilt otherwise: the rows to recompute would be more than half of all rows (an update
  would then cost about what a rebuild does), rows were removed, or `dim` changed.
  `knn_cache.reason` says which.

A change spreads: with k=10 every changed row sits in about ten other lists. Changing 1% of
the rows at random recomputes about 10% of them, and changing 5% about half, so an update
pays off for small, scattered re-embeddings and for appends.

The neighbor lists equal a full computation except among exactly tied distances, and the
findings agree to float32 rounding. In Python,
`vecscope.incremental.cached_knn_for_audit(x, k, path)` returns the lists for
`audit(x, knn=...)`.

On 100,000 rows x 768 dims (the machine of the [performance](performance.md#performance) table):

- appending 5,000 rows took 11 s to extend the cache and 4 s to audit from it, against 94 s
  for a full audit
- appending 20,000 rows took 43 s plus 4 s against 123 s

Every verdict matched the full audit and the largest relative value difference was 7e-5.
Creating the cache costs about what a full audit does (83 s here).

Re-embedding 1,000 of the 100,000 rows in place recomputed 10,625 rows: 15 s to update the
cache and 5 s to audit from it, against 95 s for a full audit, with the same verdicts. With
10,000 rows changed an update would recompute 68,933 rows in 83 s, about as long as the full
audit's 89 s, so the cache is rebuilt there instead. Re-run with that rule, 10,000 changed
rows rebuilt the cache in 87 s against a full audit's 87 s.

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
vecscope audit vectors.npy --queries q.npy --qrels qrels.tsv   # add nDCG/recall/MRR of exact search
vecscope audit vectors.npy --queries q.npy --qrels qrels.tsv --ranked topk.npy   # and of the production ranking
vecscope audit index.faiss --ann-from-source --sample 0     # read a faiss index or store; score its own neighbors
vecscope audit "qdrant://host:6333/docs" --save-ids ids.json   # vector store URI; write ids in row order
vecscope audit vectors.npy --sample 0 --knn-cache knn.npz   # reuse / extend / update exact neighbor lists between runs
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
vecscope sweep vectors.npy --nlist 128                 # sweep nprobe on the production index's list count
vecscope compare a.npy b.npy -k 10                     # agreement between two embeddings of the same rows
vecscope compare a.npy b.npy --neighbors-a a_knn.npy --neighbors-b b_knn.npy   # full-corpus neighbors on a sample of rows
vecscope calibrate fit_*.json --min-quality 0.45 --out th.json   # gate thresholds from your own pipeline states
vecscope calibrate fit_*.json --min-quality 0.45 --holdout hold_*.json --include-low-confidence   # check on held-out states; also write exploratory pairs
```

## Input formats

The embedding file holds one vector per row. Every format is converted to a 2-D `float32` array
of shape `(n, dim)` before the audit.

Values must be finite: NaN, Inf and magnitudes above 1e16 (whose squares overflow `float32`)
are rejected with a message.

In the examples `embeddings` is that array, for instance the output of a model's `encode`:

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

How much memory a format takes to load, measured on 50,000 x 384 float32 vectors (a 73 MiB
matrix):

| Format | How it is read | Peak memory |
| --- | --- | --- |
| `.npy` | as stored | 0.10 GiB |
| `.csv` | in chunks of lines | 0.18 GiB |
| `.jsonl` | line by line, converted in blocks of rows | 0.34 GiB |
| `.parquet` | only the vector column, in batches, into one array | 0.39 GiB |
| `.json` | the whole file parsed at once | 1.13 GiB |

For large corpora use `.npy`, `.csv` or `.parquet`. `.json` is parsed whole and needs several
times the matrix size.

The optional inputs:

- **`--ann`** (`ann_path` in MCP): an integer array of shape `(n, k)`. Row `i` holds the ids
  of the `k` neighbors an ANN index returned for item `i`, in the same row order as the
  embedding file. Any of the formats above works.
- **`--queries`** (`queries_path` in MCP): query embeddings in any of the same formats, with
  the same `dim` as the corpus.
- **The `query_shift` reference**: an earlier `--json` output of an audit run with
  `--queries` (`--baseline`, or in MCP `baseline` or `query_baseline_path`).

ANN ids refer to row positions, so the file must come from an index built on exactly these
rows, and every row has to be audited. Above the `--sample` size (100k rows by default) pass
`--sample 0`, or the audit refuses. Ids that point at the item itself are ignored, so an
index that returns self first should be queried with k + 1.

The `query_shift` reference is used only when it was audited on the same corpus or an
unchanged prefix of it (checked with `corpus_fingerprint`, a SHA-256 of the rows). After
normalizing or centering, the earlier query distances are in another space, so
`query_shift` is skipped and `query_reference_note` says why.
