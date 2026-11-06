# Agent cases over MCP

**English** · [한국어](ko/agent_case.md)

These are recorded runs of Claude Code in headless mode using the vecscope MCP server. In
each case it diagnosed a problem, changed something, and re-verified the change with a
re-audit. The numbers below come from the tool responses of the runs, not from the agent's
summary.

- [Case 1](#case-1-two-ingestion-faults): two ingestion faults in the corpus
- [Case 2](#case-2-an-ivf-index-searched-with-too-few-probes): an IVF index searched with
  too few probes

Both cases are built by [`examples/agent_case.py`](https://github.com/076923/vecscope/blob/main/examples/agent_case.py) from SciFact
(BEIR): 5,183 abstracts and 300 test queries with 339 relevance judgments, embedded with
bge-small-en-v1.5 (384 dims, unit length, the query instruction on queries). Each run used
Claude Code 2.1 with the model claude-opus-5-5, and could call the vecscope tools and run
Python, and nothing else.

## Case 1: two ingestion faults

### The data

Real embeddings with two injected faults (`--case ingestion`):

- Fault 1, a batch ingested twice: documents 1000 to 1599 appended again at the end, so the
  corpus has 5,783 rows.
- Fault 2, a worker that skipped normalization: documents 2600 onward stored with a per-row
  length between 0.6 and 1.6.
- `prod_topk.npy`: the top-10 per query of an inner-product index over the corpus file, as a
  faiss `IndexFlatIP` returns it.

The agent got the four files and this prompt, without being told what was injected:

```text
Our SciFact search keeps returning the same paper twice and some queries look
worse since the last ingestion. The corpus embeddings are corpus.npy, query embeddings
queries.npy, relevance labels qrels.tsv (query row, corpus row, grade; rows are positions in
those files) and the production index's top-10 per query prod_topk.npy (an inner-product
index over corpus.npy). Use the vecscope MCP tools to find what is wrong, fix the corpus
into corpus_fixed.npy (and recompute the production top-10 into prod_topk_fixed.npy the
same way), and show with a re-audit that the fix worked. Do not change the queries.
```

### How it was run

```bash
python examples/agent_case.py --case ingestion --out run
cd run
claude -p "$(cat PROMPT.txt)" \
  --mcp-config mcp.json --strict-mcp-config --setting-sources project \
  --output-format stream-json --verbose \
  --allowedTools "mcp__vecscope__audit_embeddings mcp__vecscope__describe_findings mcp__vecscope__compare_embeddings Bash(python:*) Bash(python3:*) Read Write"
```

`mcp.json` is written by hand and registers `vecscope-mcp --allow-dir <run directory>`.

### What the agent did (first run, 8 turns, 58 s)

1. **Looked at the files**: shapes and dtypes of the arrays, the head of `qrels.tsv`.
2. **Audited** with `audit_embeddings(corpus.npy, queries_path, qrels_path, ranked_path,
   sample=0, verbose=false)`. The summary line it read:

   ```text
   verdict=fail n=5783 dim=384 k=10; collapse=0.208 fail; norms=0.192 warn;
   metric_mismatch=0.0914 fail; retrieval nDCG@10=0.689 exact, 0.100 ranked;
   plan step 1 of 2: input_integrity (collapse): ... deduplicate before changing anything else
   ```

3. **Checked the findings in the data** with Python:
   - Rows 2600 to 5182 (2,581 rows) are not unit length; the queries are.
   - Rows 5183 to 5782 are byte-for-byte copies of rows 1000 to 1599.
   - `prod_topk.npy` is exactly a brute-force inner-product top-10, and every one of its
     result slots came from the long rows.
4. **Fixed** the corpus.
   - It dropped the 600 appended copies after checking they equal their originals. The
     remaining rows keep their positions, so `qrels.tsv` still lines up.
   - It scaled every row to unit length.
   - It recomputed the inner-product top-10 over the fixed corpus.
5. **Re-audited** with `baseline` set to the first `audit_id`:

   ```text
   verdict=ok n=5183 dim=384 k=10; all findings ok; retrieval nDCG@10=0.713 exact, 0.713 ranked;
   vs baseline: verdict fail -> ok, regressions: none, improved: metric_mismatch, collapse, norms,
   exact nDCG 0.689 -> 0.713, ranked nDCG 0.100 -> 0.713 (settings differ from the baseline: not comparable)
   ```

6. **Reported** the faults, the fix, and the before and after numbers. It said why the
   baseline comparison is marked not comparable: the row count changed. It also said that
   the retrieval numbers still compare, because both runs used the same labels.

The plan put deduplication first (`input_integrity`) and the metric second. The agent
applied both in one change, because its checks showed two independent faults, and the
re-audit judged both.

### Before and after

| | before | after |
|---|---|---|
| verdict | fail | ok, no regressions |
| collapse (`near_duplicate_ratio`) | 0.208 fail | 0.0 ok |
| norms (`norm_cv`) | 0.192 warn | 0.0 ok |
| metric_mismatch (`topk_overlap`) | 0.0914 fail | 1.0 ok |
| exact search nDCG@10 / recall@10 / MRR@10 | 0.689 / 0.826 / 0.655 | 0.713 / 0.836 / 0.682 |
| exact search `duplicate_result_share` | 0.096 | 0.0 |
| production nDCG@10 / recall@10 / MRR@10 | 0.100 / 0.127 / 0.101 | 0.713 / 0.836 / 0.682 |
| production `overlap_with_exact` | 0.093 | 1.0 |

### Second run (11 turns, 75 s)

The same prompt on a fresh copy of the data found the same two faults and made the same
change, with the same re-audit numbers. It took more turns because it wrote its checks to a
script first. It also confirmed with faiss `IndexFlatIP` that it could reproduce
`prod_topk.npy` before recomputing it.

## Case 2: an IVF index searched with too few probes

### The data

The embeddings are clean this time (`--case index`). `prod_topk.npy` is the top-10 of a faiss
`IndexIVFFlat` (inner product, nlist=128) searched with nprobe=1, so each query looks in one
of 128 lists. The agent got the same four files and this prompt:

```text
Since we moved our SciFact search to a faiss IVF index (IndexIVFFlat, inner
product, nlist=128, nprobe=1), answers got worse. The corpus embeddings are corpus.npy, query
embeddings queries.npy, relevance labels qrels.tsv (query row, corpus row, grade; rows are
positions in those files) and the production index's top-10 per query prod_topk.npy. Use the
vecscope MCP tools to find whether the embeddings or the index are at fault, choose index
settings (we would like to keep an IVF index, latency matters), recompute the production
top-10 with them into prod_topk_fixed.npy, and show with a re-audit that the fix worked. Do
not change corpus.npy or queries.npy.
```

It was run like case 1, with `sweep_index` added to `--allowedTools` and a faiss GPU build
on an NVIDIA L4 available to the server.

### What the agents did (two runs, 7 turns each, 37 s and 41 s)

1. **Audited** with the queries, qrels and production top-10:

   ```text
   verdict=ok n=5183 dim=384 k=10; all findings ok; retrieval nDCG@10=0.713 exact, 0.422 ranked
   ```

   Every embedding finding was `ok` while the production ranking lost 0.29 nDCG and shared
   41.6% of its top-10 with exact search. Both agents concluded the index was at fault.
2. **Swept** the index with `sweep_index` on the real queries. Run 1 forced the CPU and got
   HNSW as the fastest setting at the target. Run 2 left the device on `auto`, so IVF ran on
   the GPU, and got:

   ```text
   recommend ivf (nlist=132, nprobe=32): recall@10=0.969 (margin +0.019, tight), ... The
   fastest setting with margin >= 0.02 is ivf (nlist=132, nprobe=64) at recall 0.992
   ```

3. **Re-measured at nlist=128.** The sweep chose nlist from the row count (132), not the
   production 128. Both agents rebuilt the production index, confirmed it reproduced
   `prod_topk.npy` (overlap 0.416), and measured nprobe on it in Python.
4. **Chose nprobe and recomputed the top-10.** Run 1 took nprobe=32, the cheapest setting
   with room over 0.95 recall. Run 2 took nprobe=64, the sweep's `safer_alternative`, because
   the recommendation was marked tight.
5. **Re-audited** with `baseline` set to the first `audit_id`.

### Before and after

| production top-10 | before | run 1 (nprobe=32) | run 2 (nprobe=64) | exact search |
|---|---|---|---|---|
| nDCG@10 | 0.422 | 0.700 | 0.711 | 0.713 |
| recall@10 | 0.470 | 0.820 | 0.833 | 0.836 |
| MRR@10 | 0.417 | 0.671 | 0.681 | 0.682 |
| `overlap_with_exact` | 0.416 | 0.969 | 0.992 | |

Every embedding finding stayed `ok` and the baseline diff listed no regressions.

### What it changed in vecscope

Both runs spent a step re-measuring by hand because the sweep could not be told the
production nlist. `vecscope sweep --nlist` and `sweep_index(nlist=...)` now sweep nprobe on
the lists the production index has.

## What these show and what they do not

- The tools, the plan and the `baseline` re-audit were enough for an agent to go from a
  symptom ("same paper twice", "answers got worse") to a verified fix without being told the
  cause.
- In both cases the agent checked the findings in the data or against the index before
  changing anything. In case 1 it kept the qrels valid without being asked; in case 2 it
  confirmed its rebuilt index reproduced production before tuning it.
- In case 2 the clean audit and the retrieval block together placed the loss in the index:
  exact search on the same embeddings already scored 0.713.
- This is one model on two tasks with two runs each, on problems vecscope's findings are
  designed to catch. It does not show that every agent follows the plan, or how the
  findings do on faults they do not measure (a missing query prefix, for example; see
  [verification](verification.md)).
- `tests/test_agent_workflow.py` runs a similar diagnose, change and re-verify loop over the
  MCP stdio protocol on every test run, without a model.
