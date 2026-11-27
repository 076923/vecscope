# Agent cases over MCP

**English** · [한국어](ko/agent_case.md)

Recorded runs of Claude Code in headless mode with the vecscope MCP server: in each case it
diagnosed a problem, changed something and re-verified with a re-audit. The numbers come
from the tool responses, not from the agent's summary.

- [Case 1](#case-1-two-ingestion-faults): two ingestion faults in the corpus
- [Case 2](#case-2-an-ivf-index-searched-with-too-few-probes): an IVF index searched with
  too few probes
- [Case 3](#case-3-a-healthy-corpus-and-a-request-to-clean-it-up): a healthy corpus and a
  request to center or whiten it
- [Case 4](#case-4-the-wrong-cause-in-the-prompt): the prompt blames the corpus, the queries
  come from another model

Cases 3 and 4 test the opposite: whether the agent leaves the vectors alone when a change
would make search worse, even when the user asks for it.

All four cases are built by [`examples/agent_case.py`](https://github.com/076923/vecscope/blob/main/examples/agent_case.py) from SciFact
(BEIR): 5,183 abstracts and 300 test queries with 339 relevance judgments, embedded with
bge-small-en-v1.5 (384 dims, unit length, the query instruction on queries). Each run used
Claude Code 2.1 with the model claude-opus-5-5 and could only call the vecscope tools and run
Python.

## Case 1: two ingestion faults

### The data

Real embeddings with two injected faults (`--case ingestion`):

- Fault 1, a batch ingested twice: documents 1000 to 1599 appended again at the end, so the
  corpus has 5,783 rows.
- Fault 2, a worker that skipped normalization: documents 2600 onward stored with a per-row
  length between 0.6 and 1.6.
- `prod_topk.npy`: the top-10 per query of a faiss `IndexFlatIP` (inner product) over the
  corpus file.

The agent got the four files and this prompt, not what was injected:

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
   - `prod_topk.npy` is exactly a brute-force inner-product top-10, and every result slot
     came from the long rows.
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

6. **Reported** the faults, the fix and the before and after numbers, and why the baseline
   comparison is marked not comparable (the row count changed) while the retrieval numbers
   still compare (same labels).

The plan put deduplication first (`input_integrity`) and the metric second. The agent
applied both in one change because its checks showed two independent faults; the re-audit
judged both.

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

On a fresh copy of the data, the same prompt found the same two faults and made the same
change, with the same re-audit numbers. It took more turns because it wrote its checks to a
script first, and it confirmed with faiss `IndexFlatIP` that it could reproduce
`prod_topk.npy` before recomputing it.

## Case 2: an IVF index searched with too few probes

### The data

The embeddings are clean (`--case index`). `prod_topk.npy` is the top-10 of a faiss
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
   41.6% of its top-10 with exact search. Both agents blamed the index.
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

Both runs spent a step re-measuring by hand because the sweep picked its own nlist.
`vecscope sweep --nlist` and `sweep_index(nlist=...)` sweep nprobe on the lists the
production index has.

## Case 3: a healthy corpus and a request to clean it up

### The data

Clean SciFact embeddings, no production ranking (`--case healthy`). Every finding is `ok` and
exact search scores nDCG@10 0.713. The prompt asks for a fix that nothing calls for:

```text
Our SciFact search quality feels mediocre. I read that raw embeddings are often
anisotropic and that centering or whitening them improves retrieval. The corpus embeddings are
corpus.npy, query embeddings queries.npy, relevance labels qrels.tsv (query row, corpus row,
grade; rows are positions in those files). Use the vecscope MCP tools to clean up the
embeddings so search gets better: write the improved vectors to corpus_fixed.npy and
queries_fixed.npy, and show with a re-audit that it worked.
```

It was run like case 1, twice, on fresh copies of the data.

### What the agents did (two runs, 9 and 8 turns, 58 s and 42 s)

1. **Audited** the original: `verdict=ok ...; all findings ok; retrieval nDCG@10=0.713
   exact`.
2. **Wrote the two transforms to trial files**, not to `corpus_fixed.npy`: centering (the
   corpus mean subtracted from corpus and queries, re-normalized) and PCA whitening.
3. **Re-audited each trial** with the first audit as `baseline` and the same qrels. Both
   responses ended in a rejection, for example:

   ```text
   vs baseline: verdict ok -> ok, regressions: none, improved: none, exact nDCG 0.713 -> 0.690,
   RETRIEVAL WORSE beyond tolerance: exact.ndcg_at_k; change decision: REJECT (retrieval
   dropped: exact.ndcg_at_k 0.7127 -> 0.6902; keep the original)
   ```

4. **Wrote neither output file** and reported why, with the table below, in both runs.

| | nDCG@10 | recall@10 | MRR@10 | `change_decision` |
|---|---|---|---|---|
| original | 0.713 | 0.836 | 0.682 | |
| centered | 0.690 | 0.815 | 0.661 | reject |
| whitened | 0.637 | 0.745 to 0.750 | 0.608 to 0.610 | reject; `distance_concentration` turns `warn` |

Both transforms did what they are meant to do to the geometry (the random-pair cosine fell to
about 0, hubness from 1.60 to 1.27 and 0.58) and still cost retrieval, as the
[verification](verification.md#interventions) found for centering healthy models (up to
−0.034 nDCG@10). Here the tool decided, not the agent's reading of two numbers.

## Case 4: the wrong cause in the prompt

### The data

The corpus is the clean bge-small one; the queries come from another 384-dim model,
all-MiniLM-L6-v2, as from a query service on the wrong model (`--case misdirected`). The
prompt names the wrong cause:

```text
Users say our SciFact search got much worse this week. We think the corpus
embeddings are anisotropic and need to be centered. The corpus embeddings are corpus.npy, query
embeddings queries.npy (from the query service), relevance labels qrels.tsv (query row, corpus
row, grade; rows are positions in those files). Use the vecscope MCP tools to fix it: center
the corpus into corpus_fixed.npy (and the queries into queries_fixed.npy if they need the same
transform), and show with a re-audit that search is better.
```

### What the agents did (two runs, 6 turns each, 32 s and 31 s)

1. **Audited**:

   ```text
   verdict=fail n=5183 dim=384 k=10 scope=full; ood_queries=1 fail; query_gap=5.51 fail;
   retrieval nDCG@10=0.221 exact; plan step 1 of 1: query_pipeline (query_gap, ood_queries):
   embed a few queries with the corpus model and compare query_gap; ...
   ```

   Both agents reported that `anisotropy` was `ok` (0.628) and that every failing finding was
   on the query side.
2. **Centered corpus and queries**, as the prompt asked, into the `_fixed` files, and
   re-audited them against the first audit with `retrieval_tolerances={"ndcg_at_k": 0.01}`,
   which neither prompt nor skill had asked for.
3. **Got a rejection** (nDCG@10 0.221 to 0.133) and told the user not to deploy the files;
   both named the query service's model, version or prefix as the next thing to check.

| | before | centered |
|---|---|---|
| nDCG@10 / recall@10 / MRR@10 | 0.221 / 0.327 / 0.190 | 0.133 / 0.220 / 0.107 |
| `query_gap` | 5.51 fail | 1.94 ok |
| `query_hubness` | 3.80 ok | 8.49 warn |
| `change_decision` | | reject |

`query_gap` reads `ok` after centering, and the diff lists it under `improvements`, while
retrieval fell. Both agents saw why: the median query distance barely moved (0.818 to 0.797)
while centering spread the corpus apart (median neighbor distance 0.148 to 0.410), and
`query_gap` divides one by the other. A finding computed against the corpus geometry does not
compare across a transform of that corpus, so the decision rests on the labeled retrieval
numbers.

Computed outside the runs: centering the corpus alone raises nDCG@10 to 0.232 and is
adopted, with `query_gap` still in `still_not_ok`. That is why an `adopt` names what is still
not ok: a change that misses the fault can still nudge retrieval up.

## What these show and what they do not

- The tools, the plan and the `baseline` re-audit were enough for an agent to go from a
  symptom ("same paper twice", "answers got worse") to a verified fix without being told the
  cause.
- In cases 1 and 2 the agent checked the findings in the data or against the index before
  changing anything. In case 1 it kept the qrels valid without being asked; in case 2 it
  confirmed its rebuilt index reproduced production before tuning it.
- In case 2 the clean audit and the retrieval block together placed the loss in the index:
  exact search on the same embeddings already scored 0.713.
- In cases 3 and 4 the requested change would have made search worse. The agents tried it on
  copies, the re-audit rejected it on the labels, and the originals stayed; in case 4 they
  also put the fault on the queries, as the findings did, against the prompt.
- This is one model on four tasks with two runs each, on problems vecscope's findings are
  designed to catch. It does not show that every agent follows the plan, or how the
  findings do on faults they do not measure (a missing query prefix, for example; see
  [verification](verification.md)).
- `tests/test_agent_workflow.py` runs a similar diagnose, change and re-verify loop over the
  MCP stdio protocol on every test run, without a model.
