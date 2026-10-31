# Agent case: two ingestion faults found and fixed over MCP

**English** · [한국어](ko/agent_case.md)

This is a recorded run of Claude Code in headless mode using the vecscope MCP server: it
diagnosed a corpus, changed it, and re-verified the change with a re-audit. The numbers
below come from the tool responses of the run, not from the agent's summary.

## The data

Real embeddings with two injected faults, built by
[`examples/agent_case.py`](../examples/agent_case.py):

- SciFact (BEIR): 5,183 abstracts and 300 test queries with 339 relevance judgments,
  embedded with bge-small-en-v1.5 (384 dims, unit length, the query instruction on queries).
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

## How it was run

```bash
python examples/agent_case.py --out run
cd run
claude -p "$(cat PROMPT.txt)" \
  --mcp-config mcp.json --strict-mcp-config --setting-sources project \
  --output-format stream-json --verbose \
  --allowedTools "mcp__vecscope__audit_embeddings mcp__vecscope__describe_findings mcp__vecscope__compare_embeddings Bash(python:*) Bash(python3:*) Read Write"
```

- `mcp.json` is written by hand and registers `vecscope-mcp --allow-dir <run directory>`.
- The agent could call the vecscope tools and run Python, and nothing else.
- Claude Code 2.1, model claude-opus-5-5.

## What the agent did (first run, 8 turns, 58 s)

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

## Before and after

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

## Second run (11 turns, 75 s)

The same prompt on a fresh copy of the data found the same two faults and made the same
change, with the same re-audit numbers. It took more turns because it wrote its checks to a
script first. It also confirmed with faiss `IndexFlatIP` that it could reproduce
`prod_topk.npy` before recomputing it.

## What this shows and what it does not

- The tools, the plan and the `baseline` re-audit were enough for an agent to go from a
  symptom ("same paper twice, worse queries") to a verified fix without being told the
  cause.
- The agent verified each finding in the data before changing anything, and kept the qrels
  valid without being asked.
- This is one model on one task with two runs, on faults that vecscope's findings are
  designed to catch. It does not show that every agent follows the plan, or how the
  findings do on faults they do not measure (a missing query prefix, for example; see
  [verification](verification.md)).
- `tests/test_agent_workflow.py` runs a similar diagnose, change and re-verify loop over the
  MCP stdio protocol on every test run, without a model.
