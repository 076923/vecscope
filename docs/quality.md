**English** · [한국어](ko/quality.md)

# Retrieval quality and thresholds

## Retrieval quality and the fix order

The findings describe the vector space; whether the right documents come back needs labels.

Pass relevance judgments for the queries and the audit adds a `retrieval` block:

- nDCG@k, recall@k, MRR@k and hit rate of exact cosine search over every row (also when the
  findings are sampled)
- `duplicate_result_share`, the share of result slots taken by a near-duplicate of a
  higher-ranked result

`--ranked` scores the top-k your production index returned next to exact search, with
`overlap_with_exact`.

`--ranked` takes the final top-k of whatever serves the queries: an ANN index, a store query
with filters, hybrid search or a reranker. The exact baseline is plain cosine over every row,
with no filter, keyword score or reranking.

- With those stages in the pipeline, `overlap_with_exact` measures how far the pipeline
  departs from vector search, not ANN recall alone.
- vecscope does not score the stages separately. To find the stage that loses quality, pass
  each stage's output as `--ranked` in turn.

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

The scores average over the queries with at least one relevant (grade > 0) judgment, since
nDCG and recall are undefined without one. Queries with none, or with no judgment at all, are
counted in `unscored_queries` and named in the explanation.

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

### The fix order

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

A few findings move to the step whose cause can produce them:

- A near-duplicate share caused by a shared direction (every cosine above the cutoff, no
  exact copies) goes with the direction, not with deduplication.
- A `query_gap` next to duplicates goes with them. Near-copies shrink the corpus distance the
  gap divides by, so it is read again after deduplicating before the queries are blamed.
- Islands next to a near-duplicate group smaller than the island size go with the
  duplicates, since the copies may be the islands.

The rules are fixed and listed in `src/vecscope/plan.py`. A step is a hypothesis to test with
its re-measure line, not a diagnosis.

Each finding in a step carries `placed_because`, one of:

- "this group's cause covers it"
- the other finding whose verdict put it there, for example "rogue_dimensions is not ok; one
  coordinate moves this finding too". This is a grouping by co-occurrence, so check it with
  the step's re-measure numbers.

What the fixes did to retrieval on real embeddings is in the
[verification](https://github.com/076923/vecscope/blob/main/docs/verification.md#interventions).

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
  default), its 95% bootstrap interval, and whether the sign is the one the finding implies
- a one-sided permutation p-value in that direction, Holm-adjusted over every finding tested
- the worst value among acceptable states and the best among the others
- a (warn, fail) pair only when the finding tracks quality and separates the two groups,
  placed a third and two thirds of the way across the gap

A finding that does not move with quality on your data, or moves the other way, gets no
pair: keep its default as a reason to look.

With about fifteen findings tested at once, one of them tracks quality by chance on a few
states. So a pair goes into `--out` only when its adjusted p-value is below 0.05:

- The other pairs are reported as `exploratory` and left out of the gate file.
  `--include-low-confidence` writes them too.
- On five states nothing can reach that bar (the smallest one-sided p-value is 1/120 before
  the adjustment). Ten or more states are recommended, and the report says when there are
  fewer.

Two checks judge the proposal on states it was not fitted on:

- `leave_one_out` fits on all states but one and judges the one left out, for every state.
- The holdout line counts the unacceptable `--holdout` states the gate catches and the false
  alarms on acceptable ones. Holdout states from another dataset check whether the
  thresholds carry over.

Calibration needs at least three states, with some on each side of `--min-quality`. States
scored on different qrels or k are refused, because their quality values do not compare
(`--allow-mixed-qrels` overrides).

The same procedure applies to a `suggested_action`: it is a hypothesis about the cause, and
centering, standardizing or truncating vectors changes rankings.

- Keep the original vectors and retrieval results as the baseline.
- Apply the transform to a copy.
- Measure quality, latency and memory on both.
- Adopt the change only when the labeled queries improve, not when the finding turns `ok`.
