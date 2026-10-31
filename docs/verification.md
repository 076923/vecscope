**English** · [한국어](ko/verification.md)

# Verification

This page records how vecscope's claims were checked and what the measurements show:

- exactness against float64
- the ANN recall scorer
- sampled audits
- the thresholds against retrieval quality on labeled data
- the query findings
- what happens to retrieval when a fix from the plan is applied
- an agent following the plan over MCP

The thresholds and the plan rules are set from these results.

- Environment: AMD EPYC 7R13, 8 vCPU, 26 GB RAM, NVIDIA L4; Python 3.12.3, numpy 2.2.6,
  faiss 1.15.1, sentence-transformers 3.4.1, torch 2.6.0
- Corpora:
  - six synthetic shapes (`examples/bench_common.py`: clustered, isotropic, anisotropic,
    duplicates, lowrank, zipf)
  - the eight public pre-computed embedding sets of the README table
  - nine labeled datasets encoded with 19 models: BEIR SciFact, NFCorpus, ArguAna, SCIDOCS
    and FiQA-2018, and four Korean sets (ko-strategyqa, and from MTEB XPQA kor-kor, AutoRAG
    and MultiLongDoc ko)
- Vector stores: faiss 1.15.1, Qdrant client 1.20 (local mode), Chroma 1.5, Milvus Lite 3.2,
  Weaviate client 4.24

## Results

| area | result |
|---|---|
| tests | `pytest -q` and `pytest -m slow` pass; the vector store tests run against real local Qdrant, Chroma and Milvus Lite instances and a faiss index file |
| exact k-NN (`exact_knn`, `query_knn`, faiss flat CPU and GPU) | 0 rank errors against float64 on six shapes at 100k x 768 and on 76 (model, dataset) pairs of real embeddings; float32 distance error at most 5.9e-7 |
| ANN recall scoring | tie slack is `1e-6 + 2e-6 x distance`, about three times the float32 error. On 24 index settings at 100k x 768 the score is within 1e-4 of a strict float64 scorer (at most one query in 1,000) and exact copies still count as interchangeable; on real queries within 3.1e-4 (one query in 323) |
| sampled audit, values | distribution findings within 2.3% of the full value; hubness and antihubs a lower bound everywhere; the collapse `estimated_full_share` is 1.015 to 1.026 x the full share |
| sampled audit, verdicts | every sampled verdict below the full-corpus one is listed in `inconclusive_findings` (clustered and duplicates hubness and antihubs, zipf antihubs and connectivity) |
| `sample_repeats` | `value_range` is the spread between samples; it often does not contain the full-corpus value, so it is not an interval estimate |
| thresholds | on the 18 pipelines that reproduce published nDCG@10 within 0.007, no `anisotropy`, `query_gap` or `query_hubness` verdict is worse than `ok`; all 11 English-only models on Korean data raise `hubness` or `query_hubness` |
| README "Real embeddings" table | all 88 values and verdicts reproduced on the first 40k rows of each set |
| incremental audit (`--knn-cache`) | on 100k x 768 with 5k or 20k appended rows, verdicts identical to a full audit and values within 7e-5 relative; 15 s and 47 s against 94 s and 123 s for the full audit |
| sweep | recommendations identical across three seeds; IVF recall at a fixed nprobe can rise or fall with corpus size depending on how the data's clusters line up with nlist |

## Finding values against nDCG@10

Spearman correlation between each finding's value and nDCG@10 across the 19 models, per
dataset. A negative value means the finding rises as retrieval gets worse.

ArguAna nDCG@10 is computed as BEIR does, without the query's own document, which is in the
corpus.

| finding | SciFact | NFCorpus | ko-strategyqa | ArguAna | SCIDOCS | FiQA |
|---|---|---|---|---|---|---|
| query_hubness | −0.57 | −0.46 | −0.90 | −0.85 | −0.71 | −0.46 |
| hubness | −0.52 | −0.48 | −0.76 | −0.03 | −0.68 | −0.48 |
| antihubs | −0.27 | −0.12 | −0.72 | −0.72 | −0.60 | −0.40 |
| query_gap | −0.23 | −0.02 | −0.81 | −0.40 | −0.27 | −0.25 |
| effective_rank | +0.06 | −0.05 | −0.30 | −0.04 | −0.20 | −0.38 |
| rogue_dimensions | −0.38 | −0.38 | −0.05 | −0.12 | +0.18 | +0.05 |
| anisotropy | +0.29 | +0.21 | −0.11 | −0.24 | −0.20 | −0.24 |
| distance_concentration | +0.19 | +0.34 | −0.82 | −0.14 | +0.41 | +0.16 |
| outliers | 0.00 | +0.27 | −0.88 | +0.14 | +0.19 | +0.09 |
| collapse | n/a | +0.31 | −0.50 | +0.09 | +0.31 | −0.35 |

What this supports:

- `hubness` and `query_hubness` are negative on all six datasets (−0.03 to −0.90). They carry
  a signal about retrieval quality across models, strongest where some models do not handle
  the language or domain at all (ko-strategyqa, ArguAna, SCIDOCS).
- `query_gap` is negative on all six (−0.02 to −0.81) but weak on English data, where every
  reasonable model keeps it between 0.9 and 2.6 (ArguAna 0.05 to 0.93, because its queries
  are arguments taken from the corpus). Its job is to catch queries from another space, not
  to rank models.
- `anisotropy` and `rogue_dimensions` change sign between datasets; they are not quality
  scores. `anisotropy` is `ok` on all 114 pairs.
- ArguAna, SCIDOCS and FiQA were not used to set any threshold. On them:
  - `query_gap` flags only GloVe on SCIDOCS (2.90, `warn`); good models stay at or below 1.59.
  - `query_hubness` warns on 17 of the 19 models on FiQA, the good ones at 4.0 to 4.2, just
    above the `warn` threshold of 4 because the skewness grows with corpus size (English
    models: 1.5 to 3.4 up to 26k documents, 4.0 to 4.8 on FiQA's 58k).
    - The thresholds were kept, since refitting them on a held-out set would make it a
      training set; the explanation states the range instead.
    - It also fails the multilingual models on SCIDOCS (14 to 25, nDCG@10 0.15 to 0.17
      against 0.23 for the best English models).
  - `hubness` warns on most models on SCIDOCS (25k documents) and FiQA (57k): skewness grows
    with corpus size, so compare it against a baseline of the same corpus rather than read the
    absolute `warn`.
- A pipeline that is wrong in an obvious way (GloVe on BEIR, a Korean-only model on English,
  an English-only model on Korean) raises several findings at once. The count of warnings
  separates such pipelines; a single `fail` does not.

## Korean data and document length

Three more Korean sets from MTEB, none used to set a threshold, differ mainly in document
length. Documents were truncated to 512 tokens for every model.

| dataset | documents | median length | good multilingual models (7): nDCG@10 | English-only models (9): nDCG@10 |
|---|---|---|---|---|
| XPQA kor-kor | 889 | 32 characters | 0.307 to 0.382 | 0.102 to 0.174 |
| AutoRAG | 720 | 870 characters | 0.767 to 0.854 | 0.062 to 0.248 |
| MultiLongDoc ko | 6,176 | 12,878 characters | 0.225 to 0.286 | 0.004 to 0.030 |

The good models are KURE-v1, BGE-m3-ko, bge-m3, multilingual-e5-base and -large,
gte-multilingual-base and Qwen3-Embedding-0.6B.

- **Long documents (MultiLongDoc).** The findings behave as on BEIR.
  - `hubness` or `query_hubness` flags all 9 English-only models.
  - The good models are clean except one `hubness` and two `query_hubness` warnings (and the
    deprecated `ood_queries`, which still flags the e5 models).
  - Spearman with nDCG@10: `query_hubness` −0.86, `hubness` −0.76, `query_gap` −0.80.
- **A narrow-domain corpus (AutoRAG, finance and commerce documents).** The ranking signal
  holds, but one default does not fit.
  - `query_hubness` correlates at −0.94 and, with `hubness`, flags 6 of the 9 English-only
    models; no good model is flagged by either.
  - `effective_rank` warns on all 7 good models (0.064 to 0.094 against a `warn` of 0.10): a
    corpus about a few topics uses few directions. This is the domain effect `vecscope
    calibrate` exists for, not a model fault.
- **Very short documents (XPQA product answers).** The corpus findings describe the data,
  not the pipeline.
  - Every model, good or not, fails `collapse` (0.15 to 0.47) and `effective_rank`, and 15 of
    16 flag `query_gap`. Only 4% of the answers have identical text, but answers such as
    "네, 가능합니다" and "네, 가능해요" embed almost identically in every model.
  - `hubness` warns on 6 of the 7 good models and on none of the English-only ones, so its
    correlation with nDCG@10 turns positive (+0.43).
  - Read this as "many documents are interchangeable for retrieval", which the low nDCG@10 of
    every model confirms. The plan puts `query_gap` with the duplicates, so the queries are
    not blamed for it.

## Query findings

`query_gap` and `query_shift` were chosen from three candidates measured on the cached
embeddings of the 19 models on SciFact, NFCorpus and ko-strategyqa. Every (model, dataset)
pair was scored on its own queries and on three injected faults:

- the query prefix dropped (bge, e5, mxbai; 21 sets)
- queries embedded by every other model of the same dimension (312 sets)
- English-only models on Korean data (11 sets)

q_nn is the nearest query-to-corpus distance. The thresholds sit just above the worst
reproduced pipeline, so they are fitted to this data without a held-out split.

| criterion | query_gap: median q_nn / median corpus NN | query_shift: share beyond reference median + 3 MAD | mutual: top-k documents that neighbor each other | ood_queries |
|---|---|---|---|---|
| warn or worse on the 18 reproduced pipelines | **0** | **0** | **0** | 4 |
| flagged among 268 swaps that lose over 25% of nDCG@10 | 250 | 266 | 262 | 267 |
| flagged among 44 swaps that keep 75% or more | 16 | 19 | 7 | 24 |
| 768-dim swaps on SciFact and NFCorpus that fail | 108 of 112 | 110 of 112 | 92 of 112 | 110 of 112 |
| English-only model on Korean that fails | 4 of 11 | 0 of 11 | 4 of 11 | 2 of 11 |
| prefix dropped scores worse than with it | 10 of 21 | 10 of 21 | 12 of 21 | 11 of 21 |
| Spearman with nDCG@10 (SciFact, NFCorpus, ko) | −0.23, −0.02, −0.81 | n/a (one reference per set) | +0.64, +0.74, +0.09 | −0.33, −0.33, +0.23 |
| verdict changes on 5 samples of 100 queries | 0.1% | 0.2% | 1.9% | 1.0% |

`query_gap` needs no reference; `query_shift` compares against the query distances stored in
a baseline. `ood_queries` stays in the output for comparison and is deprecated.

Three of the fault criteria turned out to describe something other than a query-side fault:

- **Missing prefix.** Dropping the bge, e5 or mxbai query prefix changed nDCG@10 by −0.014
  to +0.007 on SciFact and NFCorpus, and moved the query distances closer (bge-base
  `query_gap` 1.55 to 1.21 on SciFact) or farther (e5-base 1.64 to 1.75 on NFCorpus).

  There is no geometric signal to detect, and on this data little harm.
- **Another model.** Models fine-tuned from the same base share a space: the mean cosine of
  the same document under mxbai-embed-large and bge-large is 0.985, under gte-base and
  bge-base 0.885 (0.004 for mpnet and bge-base).

  Their swapped queries keep most of the nDCG@10 (gte-base queries on the bge-base corpus:
  0.697 against 0.740), and they are the two 768-dim swaps that `query_gap` and `query_shift`
  did not fail.
- **Another language.** An English-only model on ko-strategyqa encodes the Korean queries
  and documents equally badly, so nothing is asymmetric between them; `query_hubness`
  fails 10 of the 11 (11 to 30) and warns on the last (8.3).

## Interventions

Does retrieval quality move when a plan step is applied?

The verification harness (kept outside this repository, like the other measurements on this
page):

- takes 21 real (model, dataset) pairs (8 models on SciFact, 8 on NFCorpus, 5 on
  ko-strategyqa)
- applies each change to the corpus and the same transform to the queries
- records the findings, the plan and `retrieval` (nDCG@10, recall@10, duplicate share) before
  and after

| intervention | finding | retrieval | plan |
|---|---|---|---|
| center: subtract the corpus mean, re-normalize | `anisotropy` 0.13 to 0.82 (all `ok`) falls to 0.0003 or less | nDCG@10 median −0.008, worst −0.034, better on 3 of 21; recall@10 median −0.005 (−0.023 to +0.023) | no step asked for it: `anisotropy` was `ok` everywhere |
| inject 20% copies (half exact, half 1e-4 jitter), then deduplicate | `collapse` 0.33 to 0.35, `fail` on all 21 | duplicate share of the top 10 0.15 (0.13 to 0.18) falls to 0.016; recall@10 +0.011, nDCG@10 +0.032 (median) | `input_integrity` first on 21 of 21 |
| scale one coordinate by 30, then z-score dimensions | `rogue_dimensions` 263 to 523, `fail` on all 21; 1.0 after z-scoring | nDCG@10 falls to 57% of the original (median; 7% to 68%); z-scoring restores 98% (88% to 109%) | `dominant_dimension` on 21 of 21, after `input_integrity` on the 8 NFCorpus pairs, whose corpus has its own duplicates |
| re-rank the exact top 10 with CSLS against the queries: 2 cos(q, d) minus d's mean cosine to its 10 nearest queries | k-occurrence skewness of the query results lower on 19 of 21 (median −0.73); on the 4 pairs with `query_hubness` not ok 5.4, 9.1, 11.2 and 28.5 fall to 4.7, 2.3, 2.5 and 17.2 | on those 4, nDCG@10 +0.002, +0.004, +0.016, +0.022; where `query_hubness` was `ok`, −0.015 to +0.018 (median −0.003) | `corpus_structure` (or `query_pipeline`) step; the action names CSLS against the queries. The same scaling by corpus neighbors raised the skewness (median +1.8) and lowered nDCG@10 on 16 of 21 |
| HNSW M=16 at ef_search 4, 8, 16, 64 | `ann_recall` median 0.883, 0.958, 0.984, 0.998 | ranked nDCG@10 as a share of exact 0.83, 0.92, 0.96, 0.995; Spearman 0.94 between the two over all 84 runs | `index` step on 21, 7, 0, 0 of 21 |

What this supports:

- A finding the plan acts on and a retrieval change move together when the defect is real:
  deduplicating, z-scoring a dominant coordinate and raising ef_search each recover retrieval
  quality, and `ann_recall` ranks index settings the way ranked nDCG@10 does.
- Hub reduction helps where the finding says there are hubs: CSLS against the queries raised
  nDCG@10 on every pair with `query_hubness` not `ok` and was a coin flip elsewhere, so the
  plan proposes it only for non-ok hub findings and the re-measure line asks for labeled
  queries.
- Search latency: on these corpora (5,000 documents or fewer) one batch of queries takes
  0.002 to 0.04 ms per query at every ef_search, below the measurement noise, so the latency
  side of the ef_search trade-off is measured by `vecscope sweep` on 100k rows (README
  "Index sweep"), not here.
- Centering a healthy model does not help: it removed `anisotropy` and cost up to 0.034
  nDCG@10. The plan only proposes it when `anisotropy` is not `ok`, and its re-measure line
  says to check retrieval on labeled queries.
- A dominant coordinate also spreads the norms and splits cosine from inner product.
  Normalizing does not remove the coordinate, so the plan puts those findings in the
  `dominant_dimension` step with the z-score change.

## Agent integration

- `tests/test_agent_workflow.py` drives the MCP server over stdio: a corpus with
  unnormalized vectors and a shared offset, audited with labeled queries.
  - The loop applies the first plan step and re-audits until no injected defect is left.
  - The plan asks for `metric` (normalize) and then `shared_direction` (center), after which
    `anisotropy`, `metric_mismatch`, `collapse` and `norms` are `ok` and nDCG@10 is not lower.
- The test also re-audits with `baseline` set to the previous `audit_id` and checks that
  `baseline_diff` lists the step's findings as improved, none of them as regressions, and no
  loss of nDCG@10.
  - It found that a baseline made before a transform must not be the reference for
    `query_shift` (the centered queries all looked shifted).
  - The reference is now used only for the same corpus, checked by `corpus_fingerprint`.
- The same task given to Claude Code 2.1 in headless mode (`claude -p` with the vecscope MCP
  server and `--allow-dir`), twice: 10 to 11 turns, 51 to 54 seconds.
  - It followed the plan in order (metric, then shared_direction).
  - It re-audited with `baseline` set to the previous `audit_id` without being told to.
  - It took `metric_mismatch`, `anisotropy`, `hubness`, `antihubs` and `collapse` to `ok`
    with nDCG@10 unchanged at 1.0.
  - It left the remaining `outliers` warn (0.011, threshold 0.01) alone: the rows were not
    broken vectors and one was a labeled answer.

## Limitations

- The thresholds and the query findings were fitted on the data above; there is no held-out
  split. Production query logs and private corpora were not measured.
- The real-model measurements cover nine datasets in two languages. On very short templated
  documents and on narrow-domain corpora some defaults (`collapse`, `effective_rank`,
  `query_gap`) flag good models; calibrate on such data before gating. Exactness on corpora over
  20,000 rows was checked on synthetic data; the real-data exactness check covers the 76
  smaller pairs.
- Sampled audits use uniform random rows. Stratified (per-cluster) sampling was considered and
  not adopted: an island must be smaller than max(k + 1, 0.5% of the rows) and isolated, so a
  sample misses small islands whatever the strata, while stratifying would bias the
  distribution findings. Exact copies are counted over every row instead.
- The Weaviate adapter was tested only against an embedded server in opt-in tests
  (`VECSCOPE_TEST_WEAVIATE=1`); remote Qdrant, Milvus and Weaviate servers were not tested.
- The agent run is one model on one task; it shows the tools and the plan are usable, not
  that every agent follows them.

## Reproducing

- `pytest -q` runs the verification tests (`tests/test_new_features.py`,
  `tests/test_scale_exactness.py`, `tests/test_agent_workflow.py`); `pytest -m slow` adds the
  100k x 384 exactness checks.
- `python examples/benchmark_scale.py --rows 10000 50000 100000 --dim 768 --json scale.json`
  and `python examples/benchmark_sweep.py --rows 100000 --dim 768 --corpus clustered zipf
  --json sweep.json` re-measure the tables and record the machine in the JSON.
- `python examples/retrieval_quality.py` runs the labeled-data study for SciFact and
  NFCorpus.
- The 19-model, nine-dataset, query-finding and intervention runs use a harness that is not
  shipped with the package. The behavior they set is pinned by tests instead:
  - `tests/test_query_findings.py` (query_gap and query_shift)
  - `tests/test_plan_incremental.py` (the plan rules, including the dominant-dimension and
    duplicate cases)
  - `tests/test_regression_injection.py` (injected faults)
