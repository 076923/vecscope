**English** · [한국어](ko/verification.md)

# Verification

This page records how vecscope's claims were checked and what the measurements show, in five
areas:

1. [Algorithm exactness](#1-algorithm-exactness): the k-NN, the incremental audit and the
   published tables against float64 and against a full recomputation
2. [Thresholds and retrieval quality](#2-thresholds-and-retrieval-quality): every finding
   against nDCG@10 on labeled data, the query findings, what retrieval does when a plan
   step is applied, and the defaults on 19 held-out public datasets
3. [Sampled audits](#3-sampled-audits): what a sample can and cannot stand for
4. [ANN recall and index sweeps](#4-ann-recall-and-index-sweeps): the recall scorer and the
   sweep
5. [MCP security and operations](#5-mcp-security-and-operations): the server's guards and
   agents following the plan over MCP

The thresholds and the plan rules are set from these results. Each area ends with what it
does not cover; the [limitations](#limitations) collect them.

- Environment: AMD EPYC 7R13, 8 vCPU, 26 GB RAM, NVIDIA L4; Python 3.12.3, numpy 2.2.6,
  faiss 1.15.1, sentence-transformers 3.4.1, torch 2.6.0
- Corpora:
  - six synthetic shapes (`examples/bench_common.py`: clustered, isotropic, anisotropic,
    duplicates, lowrank, zipf)
  - the eight public pre-computed embedding sets of the
    [Real embeddings](quality.md#real-embeddings) table
  - nine labeled datasets encoded with 19 models: BEIR SciFact, NFCorpus, ArguAna, SCIDOCS
    and FiQA-2018, and four Korean sets (ko-strategyqa, and from MTEB XPQA kor-kor, AutoRAG
    and MultiLongDoc ko)
- Vector stores: faiss 1.15.1, Qdrant client 1.20 (local mode), Chroma 1.5, Milvus Lite 3.2,
  Weaviate client 4.24
- Tests: `pytest -q` and `pytest -m slow` pass; the vector store tests run against real
  local Qdrant, Chroma and Milvus Lite instances and a faiss index file

## 1. Algorithm exactness

| claim | result |
|---|---|
| exact k-NN (`exact_knn`, `query_knn`, faiss flat CPU and GPU) | 0 rank errors against float64 on six shapes at 100k x 768 and on 76 (model, dataset) pairs of real embeddings; float32 distance error at most 5.9e-7 |
| [Real embeddings](quality.md#real-embeddings) table | all 88 values and verdicts reproduced on the first 40k rows of each set |
| incremental audit (`--knn-cache`) | on 100k x 768 with 5k or 20k appended rows, verdicts identical to a full audit and values within 7e-5 relative; 15 s and 47 s against 94 s and 123 s for the full audit. With 1,000 rows re-embedded in place, 10,625 rows recomputed, same verdicts, values within 4e-5, 19 s against 95 s; with 10,000 the update took 83 s against 89 s, which is why the cache rebuilds once more than half the rows would be recomputed; re-run with that rule, 10,000 changed rows rebuilt it in 87 s against 87 s |
| determinism | two runs on the same input produce identical JSON (fixed seeds, exact neighbors); the golden files under `tests/golden/` pin the example matrices |

Not covered: exactness on real corpora over 20,000 rows was checked on synthetic data; the
real-data check covers the 76 smaller pairs.

## 2. Thresholds and retrieval quality

| claim | result |
|---|---|
| thresholds | on the 18 pipelines that reproduce published nDCG@10 within 0.007, no `anisotropy`, `query_gap` or `query_hubness` verdict is worse than `ok`; all 11 English-only models on Korean data raise `hubness` or `query_hubness` |
| a finding the plan acts on tracks retrieval | deduplicating, z-scoring a dominant coordinate and raising ef_search each recover nDCG@10 ([interventions](#interventions)) |
| a change the plan does not ask for is not free | centering a healthy model removed `anisotropy` and cost up to 0.034 nDCG@10 |

### Finding values against nDCG@10

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

### Korean data and document length

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

### Query findings

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

### Interventions

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
  side of the ef_search trade-off is measured by `vecscope sweep` on 100k rows
  ([Index sweep](usage.md#index-sweep)), not here.
- Centering a healthy model does not help: it removed `anisotropy` and cost up to 0.034
  nDCG@10. The plan only proposes it when `anisotropy` is not `ok`, and its re-measure line
  says to check retrieval on labeled queries. A plan step is a hypothesis to test on the
  corpus at hand, not a diagnosis; an agent that applies one must re-audit and compare
  retrieval before adopting it.
- A dominant coordinate also spreads the norms and splits cosine from inner product.
  Normalizing does not remove the coordinate, so the plan puts those findings in the
  `dominant_dimension` step with the z-score change.

### Held-out public data

The defaults were then run on 19 public datasets none of them was fitted on, each encoded
with 14 public models: 266 pipelines. The datasets cover XPQA product answers in seven
languages, Amazon ESCI product titles (us, es, jp), MIRACL Wikipedia (de, fr, hi, 60k to 80k
passages), CodeSearchNet (python, go, java) and CoSQA, LegalBench terms-of-service summaries
and TREC-COVID (171k abstracts). A pipeline is good at 90% or more of the best nDCG@10 on its
dataset and bad at 50% or less. The tables per language, document length, corpus size and
domain are in [held-out validation](validation.md); the harness is `examples/validation/`.

The first run found a defect in `query_hubness`, not a threshold to move. With few queries
over a large corpus most items appear in no top-k list, so the k-occurrence skewness is high
even when no item is a hub: if every query drew its k results at random, an item's count would
be Binomial(queries, k/n) with skewness (1 - 2p) / sqrt(queries p (1 - p)), p = k/n. Random
data with no structure gives exactly that (4.86 at 300 queries over 70k items, 18.36 at 50
over 170k), so 15 of 15 good pipelines over 50k documents read `warn` or `fail` from chance
alone. The finding now reports that value (`details.null_skewness`) and gives no verdict
level that chance reaches (`details.not_judged`, with the queries that would judge it). The
rule follows from the null model, not from these datasets; the thresholds are unchanged, and
on the data they were fitted on chance stays near 1.1 to 1.3, below warn_at 4. The numbers
below are from the re-run with that change.

| claim | result |
|---|---|
| bad pipelines | 54 of 67 end at `fail` (the default `--fail-on fail` blocks them) and 66 have a finding that is not `ok`. The one without: an English-only e5 on French MIRACL (nDCG@10 0.26 against 0.59), which the first run flagged only through `query_hubness` 4.57, below the 4.69 chance gives at its 343 queries |
| good pipelines | 26 of 114 end at `fail` and 26 have every finding `ok`; on documents of 200 characters or more 4 of 54 end at `fail`, on shorter ones 22 of 60 |
| `query_hubness` | not ok on 9 good pipelines (24 in the first run) and on none of the 15 over 50k documents; not judged at warn on 49 pipelines; it still correlates negatively with nDCG@10 on all 19 datasets (Spearman -0.42 to -0.90, and -0.09 on legal summarization) |
| findings that misfire on short text | `effective_rank` (47% of good pipelines) and `collapse` (42%); `effective_rank` is not a small-sample bias (1,000 rows of a MIRACL corpus give 0.152, all 71k rows 0.175), short texts occupy fewer directions. `query_gap`, `anisotropy` and `effective_rank` change sign against nDCG@10 from one dataset to the next |
| `collapse` on duplicated text | CoSQA's documents are 70% duplicate texts, and `collapse` says so on every model; the report counts that as correct |
| calibrating a domain the defaults misfire on | on XPQA ja, `vecscope calibrate` over the 14 models (fitted on half the queries, checked on the other half) proposes no threshold: `query_hubness` and `distance_concentration` move with quality (Spearman -0.70 and 0.70, Holm p 0.045) but the good and bad models overlap, and `collapse`, `effective_rank` and `hubness`, which flag every good model there, do not move with quality |

54 of the 67 bad pipelines are English-only models on other languages, an easy fault. To
measure harder ones, `examples/validation/faults.py` injected six faults one at a time into
the 99 good pipelines on corpora of at most 25k documents and judged each run as CI would
([table](validation.md#injected-faults)). Harmful means exact nDCG@10 fell by more than 0.01.

| fault | harmful | caught by the findings (absolute `fail` or a verdict worse than the clean baseline) | of those, by the absolute verdict alone | findings alarms on harmless runs |
|---|---|---|---|---|
| queries from another model of the same dimension | 87 of 99 | 86 | 61 | 1 of 12 |
| 20% of documents stored twice | 99 of 99 | 93 | 72 | none harmless |
| 5% of documents are zero vectors | 84 of 99 | 84 | 42 | 15 of 15 |
| first half of the dimensions only | 63 of 99 | 62 | 7 | 28 of 36 |
| 4-bit rounding without per-vector scale | 32 of 99 | 27 | 0 | 14 of 67 |
| 10% of rows hold another document's vector | 99 of 99 | 0 | 0 | none harmless |

Read with the labeled retrieval tolerance (`--retrieval-tolerance ndcg_at_k=0.01`), every
harmful run fails, by construction of the harmful cut. Without labels the baseline diff
catches far more than the absolute verdict (truncation 62 against 7), and a lost id mapping
moves no finding at all: the rows are the same vectors in a different order, so only labeled
retrieval sees it. The findings alarms on harmless runs are real faults that cost 0.01
nDCG@10 or less (zero vectors cost ESCI 0.005 to 0.010), so the baseline diff reports them as
changes to look at. 11 of the 12 harmless query-model swaps paired KURE-v1 with bge-m3, which
it was fine-tuned from, and the twelfth gte-base with bge-base-en-v1.5, both from the same
BERT (the same document's two embeddings have cosine 0.865): their queries share the space,
and the findings did not flag 11 of them.

What this changes in practice:

- Gate on the baseline diff (`--fail-on none --baseline`), and on `--retrieval-tolerance`
  wherever there are labeled queries ([CI](ci.md)). The absolute verdict misses most
  truncation and quantization faults and blocks good short-text pipelines.
- On short text and non-English text, treat the absolute verdict as a list of things to look
  at.
- Pass enough queries for `query_hubness`, or read `details.not_judged`.
- The XPQA ja case is what `calibrate` should say when no finding separates good from bad
  states on a domain: nothing to propose, keep the defaults as reasons to look.

Not covered: the thresholds and the query findings were fitted on SciFact, NFCorpus and
ko-strategyqa; the held-out run above measured them and did not refit. Production query logs
and private corpora were not measured.

### A second held-out set, preregistered

Four changes were designed after reading the first held-out set: the chance rule for
`query_hubness`, the reference model (`--reference`), the label-free result change
(`--result-sample`) and the labeled-query budget. That set cannot confirm them. So the code,
the thresholds and four pass criteria were committed
([preregistration](https://github.com/076923/vecscope/blob/main/examples/validation/preregistration.md))
before seven more public datasets were downloaded. The datasets are:

- WebFAQ Korean, Thai and Vietnamese (102k to 124k FAQ answers in two of them)
- JaGovFaqs
- GerDaLIR (German legal)
- CQADupStack webmasters
- TurHistQuad (Turkish)

They were run once with the same 14 models: 98 pipelines, 36 good and 41 bad. Nothing was
tuned on them, and `examples/validation/confirm.py` scores the criteria
(`confirm_summary.json`). One criterion failed.

| prediction | criterion | result |
|---|---|---|
| P1 `query_hubness` against chance | good pipelines flagged with the rule no more than with the thresholds alone, and at least 90% of the bad ones at `fail` kept | **passed, with nothing to test**: 9 and 9 good flagged, 20 of 20 bad kept. Every dataset here has 1,000 or more queries, so chance stayed at 0.3 to 3.2, below warn 4, and the rule changed no verdict |
| P2a reference model, good | good pipelines at `fail` with the reference at most 75% of those without | **failed**: 6 and 6 |
| P2b reference model, bad | bad pipelines at `fail` with the reference at least 95% of those without | passed: 34 and 34 of 41; all 41 still flagged |
| P3 labeled-query budget | with 100 labeled queries, at least 85% of harmful faults caught and 90% of those dropping more than 0.02 | passed: 87% and 97% |
| P4 result change | every injected fault changes the sampled top-k, an identical re-run changes none | passed: 120 of 120 faults, 4 of 4 re-runs at 0 |

Why P2a failed. The reference did take most flags off good pipelines: flagged went from 34 to
14 of 36, and all-ok from 2 to 22. It explained hubness on 23 good pipelines and collapse on
10. It did not take any good pipeline off `fail`. The six that stay are each well past their
reference on the same rows, so by the rule the rows do not explain them:

| dataset | model | finding | value | reference value (bge-m3, or multilingual-e5-base for bge-m3) |
|---|---|---|---|---|
| WebFAQ ko | multilingual-e5-large | query_hubness / hubness | 70.0 / 12.2 | 3.7 / 2.8 |
| WebFAQ ko | multilingual-e5-base | query_hubness / hubness | 26.4 / 4.7 | 3.7 / 2.8 |
| WebFAQ th | multilingual-e5-large | query_hubness | 48.5 | 3.8 |
| WebFAQ th | multilingual-e5-base | query_hubness | 11.1 | 3.8 |
| WebFAQ th | bge-m3 | hubness | 4.1 | 2.8 |
| WebFAQ vi | Qwen3-Embedding-0.6B | hubness | 6.2 | 2.9 |

These are real hubs that a good nDCG@10 tolerates: an e5 model on Korean FAQs puts a few
answers in a large share of the top-10 lists and still ranks the right answer first often
enough. The reference rule is meant to clear what the rows cause, not what one model does
on them. So it fixes most of the short-text noise (`warn` flags), and it does not fix the
`fail` level on the multilingual-e5 family. On the first set too, the good pipelines left at
`fail` were mostly that family.

P3, read with its limits. The table, on 120 faults of 20 good pipelines over the four corpora
with at most 25k documents:

| labeled queries | harmful caught (95) | drop 0.02 to 0.05 (43) | drop above 0.05 (24) | harmless failed (25) |
|---|---|---|---|---|
| 25 | 75% | 77% | 98% | 34% |
| 50 | 82% | 87% | 99% | 30% |
| 100 | 87% | 95% | 100% | 26% |
| 200 | 91% | 99% | 100% | 18% |
| 400 | 95% | 100% | 100% | 11% |

It is a few points below the first set at 25 to 100 queries: 28 of the 95 harmful faults here
cost between 0.01 and 0.02, and 100 queries catch 65% of those. Only 25 harmless runs were
there, so the false-alarm share is rough.

P4 confirms that every one of the six faults moves the sampled results, the lost id mapping
included, and that the comparison is exact. It does not say whether a change is harmful: on
the first set's 594 faults the changed share separated harmful from harmless at AUC 0.58.

What this changes:

- The reference model clears short-text `warn` flags. It is not a reason to trust `fail` on a
  good-scoring pipeline: a finding that stays at `fail` with the reference is the model's
  own, and only labeled retrieval says whether it costs anything.
- 100 labeled queries are enough to gate drops above two nDCG points on corpora like these.
  Smaller drops need several hundred.
- The chance rule for `query_hubness` was not tested here, because every dataset had enough
  queries. The first set and the random-data checks are still its only evidence.

### A third held-out set, preregistered

The second set left the chance rule for `query_hubness` untested. Two changes were designed
after it, on the first set's faults: the overlap with a reference model's results as a
label-free sign of harm (`--reference-tolerance`), and the labeled gate judged against the
noise of its queries (`--retrieval-confidence`). The code, the thresholds and the criteria
were committed
([preregistration](https://github.com/076923/vecscope/blob/main/examples/validation/preregistration_third.md))
before nine more public datasets were downloaded:

- four with few queries over 25k to 117k documents, so chance alone puts `query_hubness`
  above warn: MIRACL Korean and Thai, SCIDOCS, WebFAQ Indonesian
- five with at most 25k documents for the injected faults: ArguAna, StackOverflowQA,
  Wikipedia Bengali and Czech, AutoRAG (Korean)

They were run once with the same 14 models: 126 pipelines, 47 good and 40 bad.
`examples/validation/third.py` scores the criteria (`third_summary.json`). No bug was found
in the script after the freeze. One criterion failed.

| prediction | criterion | result |
|---|---|---|
| P1a `query_hubness` against chance, good | good pipelines flagged with the rule at most half of those flagged by the thresholds alone | passed: 21 of 21 flagged by the thresholds alone, 0 with the rule (chance skewness 4.5 to 8.2) |
| P1b same, bad | at least 90% of the bad pipelines at `fail` by the thresholds alone kept at `fail` | passed: 17 of 17 |
| P5a reference overlap, separation | z of the paired drop separates harmful from harmless faults at AUC 0.80 or more, and 0.10 above the changed share | passed, at the edge: 0.802; the changed share gave 0.468 |
| P5b reference overlap at 0.02 | at least 70% of the faults costing more than 0.02 flagged, at most 35% of the harmless ones | **failed**: 64 of 81 (79%) flagged, but also 18 of 41 harmless (44%) |
| P6a labeled gate at 0.95, harmful | harmful faults pass at most 3% of the time and at most half as often as with the mean alone | passed: 0.4%, against 3.9% with the mean |
| P6b same, harmless | harmless faults fail at most 0.75 times as often as with the mean alone | passed: 10.1%, against 23.7% |
| P7 determinism | the audit with reference, result sample and qrels gives identical JSON twice | passed: 5 of 5 |

The faults were the six of `faults.py` on 26 good pipelines (156 runs; 130 without the
duplicates; 89 of those harmful, 41 harmless). P6 drew 200 subsets of 100 labeled queries
per fault.

Why P5b failed. The overlap tracks distance from the reference, and some changes move away
from it without costing labeled retrieval:

| fault | runs | nDCG@10 drop above 0.01 | flagged at 0.02 | harmless flagged |
|---|---|---|---|---|
| query model swapped | 26 | 23 | 23 | 0 of 3 |
| id mapping lost (10% of rows) | 26 | 26 | 26 | none harmless |
| truncated to half the dimensions | 26 | 11 | 22 | 11 of 15 |
| quantized to 4 bits | 26 | 5 | 10 | 6 of 21 |
| 5% zero rows | 26 | 24 | 8 | 1 of 2 |
| 20% duplicated rows | 26 | 26 | 0 | none harmless |

- Truncation and quantization: most of the false alarms. On the Wikipedia and AutoRAG
  corpora they pull the results 0.02 to 0.11 away from the reference while nDCG@10 falls by
  less than 0.01. A reference shares only part of the top-k with any model, and moving
  within the rest does not always cost relevant results.
- Zero rows: missed 17 of 24 times. Losing 5% of the rows lowered the overlap by a median of
  0.018, under the tolerance, while it cost a median 0.036 nDCG@10 because the lost rows
  include relevant documents. The findings catch zero rows: on the first set the baseline
  diff caught all 84 harmful ones.
- Duplicates: unseen, as predicted. The reference embeds the duplicated rows too.

P6 confirms the labeled gate judged against noise. With 100 labeled queries, harmful
faults failed 87.5% of the time, were inconclusive 12.0% and passed 0.4%. Harmless faults
failed 10.1%, were inconclusive 68.1% and passed 21.8%. Most harmless faults cost just
under 0.01, and 100 labels cannot settle them; the gate says so instead of guessing.

P1 confirms the chance rule where it applies: on four corpora with 60 to 213 queries, every
good pipeline that the thresholds put on `query_hubness` was cleared, and no bad pipeline at
`fail` was.

The reference rule for findings, reported without a prediction, behaved as on the second
set. Good pipelines flagged went from 32 to 15 of 47. The same 6 stayed at `fail`: e5 models
on StackOverflowQA and WebFAQ Indonesian, and KURE-v1 and Qwen3 on WebFAQ Indonesian, all
on `hubness`. Of the bad pipelines, 39 of 40 stayed at `fail` and all 40 were still flagged.

What this changes:

- `--retrieval-confidence 0.95` is the recommended labeled gate: on unseen data it let
  harmful changes through at a tenth of the mean rule's rate and failed harmless ones less
  than half as often. Read inconclusive as "label more queries".
- The reference overlap orders changes by likely harm (AUC 0.80), but at 0.02 it also flags
  close to half of the harmless changes on these corpora, and it misses lost rows. Treat
  `reference_regressions` as "check with labels before merging", not as a failure on its
  own.
- The chance rule for `query_hubness` held on its first test: on small query sets, trust it
  over the thresholds alone.

## 3. Sampled audits

| claim | result |
|---|---|
| values | distribution findings within 2.3% of the full value; hubness and antihubs a lower bound everywhere; the collapse `estimated_full_share` is 1.015 to 1.026 x the full share |
| verdicts | every sampled verdict below the full-corpus one is listed in `inconclusive_findings` (clustered and duplicates hubness and antihubs, zipf antihubs and connectivity) |
| `sample_repeats` | `value_range` is the spread between samples; it often does not contain the full-corpus value, so it is not an interval estimate |
| scope in the output | a sampled audit sets `evidence_scope: "sampled"` and lists in `needs_full_audit` the inconclusive and unstable findings; the MCP summary line starts with the scope, so a sampled estimate is never read as a full audit (`tests/test_mcp_security.py`) |

Not covered: sampled audits use uniform random rows. Stratified (per-cluster) sampling was
considered and not adopted: an island must be smaller than max(k + 1, 0.5% of the rows) and
isolated, so a sample misses small islands whatever the strata, while stratifying would bias
the distribution findings. Exact copies are counted over every row instead.

## 4. ANN recall and index sweeps

| claim | result |
|---|---|
| ANN recall scoring | tie slack is `1e-6 + 2e-6 x distance`, about three times the float32 error. On 24 index settings at 100k x 768 the score is within 1e-4 of a strict float64 scorer (at most one query in 1,000) and exact copies still count as interchangeable; on real queries within 3.1e-4 (one query in 323) |
| `ann_recall` tracks ranked retrieval | HNSW M=16 at ef_search 4, 8, 16, 64: `ann_recall` median 0.883, 0.958, 0.984, 0.998 against ranked nDCG@10 at 0.83, 0.92, 0.96, 0.995 of exact; Spearman 0.94 over 84 runs ([interventions](#interventions)) |
| sweep | recommendations identical across three seeds; IVF recall at a fixed nprobe can rise or fall with corpus size depending on how the data's clusters line up with nlist |

Not covered: search latency on corpora under 5,000 rows is below the measurement noise; the
latency side of the trade-off is measured by `vecscope sweep` on 100k rows
([Index sweep](usage.md#index-sweep)). Index types other than HNSW and IVF were not swept.

## 5. MCP security and operations

The server's guards are pinned by `tests/test_mcp_security.py` on small matrices:

- a corpus, queries or comparison file outside the allowed directories is refused, and a
  directory or a missing file is reported as a user error
- `--max-read-mb` refuses a file before it is parsed, `0` lifts the limit, and a value that
  is not a number is reported; `--max-rows` and `--max-dim` refuse a matrix (or a queries
  file) over the limit whatever its byte size, and the command-line options set them
- a network store is refused without `--allow-network`, and the error never carries the
  password, the API key or the other URI secrets; the `timeout=` option must be a finite
  positive number of seconds and at most 3,600
- a `MemoryError` while reading or auditing reaches the agent as a tool error that names the
  ways out (a sample, a smaller input, the limits) instead of ending the call silently
- a sampled response states `evidence_scope`, `needs_full_audit` and the scope in its
  summary line

Vector store reads were tested against real local Qdrant, Chroma and Milvus Lite instances
and a faiss index file (`tests/test_adapters.py`), including the byte limit stopping a read.

### Agent integration

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
- A case on real data ([agent_case.md](agent_case.md)): SciFact with bge-small-en-v1.5, a
  batch ingested twice and a worker that skipped normalization, labeled queries and an
  inner-product production top-10. Two runs of Claude Code with the model claude-opus-5-5,
  8 and 11 turns, 58 and 75 seconds.
  - Told only the symptom, both runs found the 600 copies and the 2,581 unnormalized rows,
    checked them in the data, removed the copies while keeping the qrels positions, and
    normalized.
  - Both re-audited with `baseline` and got `ok` on every finding. Production nDCG@10 went
    from 0.100 to 0.713, and exact search from 0.689 to 0.713.
- A second case on the same data with clean embeddings and an IVF index (nlist=128) searched
  with nprobe=1. Two runs, 7 turns each, 37 and 41 seconds.
  - Both audited, found every finding `ok` and production nDCG@10 at 0.422 against 0.713 for
    exact search, and put the fault on the index.
  - Both swept the index, re-measured nprobe at the production nlist, and chose nprobe=32
    and 64. Production nDCG@10 went to 0.700 and 0.711, with no finding regressed.
  - Both had to re-measure by hand because the sweep chose its own nlist (132), which led to
    `sweep --nlist`.

Not covered: the Weaviate adapter was tested only against an embedded server in opt-in tests
(`VECSCOPE_TEST_WEAVIATE=1`); remote Qdrant, Milvus and Weaviate servers were not tested.
The `timeout=` option and the scrubbing of client errors were tested against an unreachable
Qdrant port and a stubbed client, not against a slow server. The row and column limits and
the `MemoryError` path were tested on small matrices with the limits set low, not by
exhausting a machine. The memory of a long-running server under many calls was not measured.
The agent runs are one model on three tasks; they show the tools and the plan are usable,
not that every agent follows them.

## Limitations

- The thresholds and the query findings were fitted on SciFact, NFCorpus and ko-strategyqa.
  On 19 held-out public datasets ([held-out public data](#held-out-public-data)) 26 of 114
  good pipelines end at `fail`, mostly on text under 200 characters, and one of 67 bad
  pipelines has no flag: gate on the baseline diff and retrieval tolerances, not on the
  absolute verdict. Production query logs and private corpora were not measured.
- The `query_hubness` change was prompted by the held-out results; it follows from a null
  model and was checked on random data, but those datasets are no longer blind for that
  finding. The preregistered second set had enough queries everywhere to leave it untested;
  the third set tested it on four corpora with 60 to 213 queries and it passed.
- On the second set ([preregistered](#a-second-held-out-set-preregistered)) the reference model
  cut flagged good pipelines from 34 to 14 of 36 but left all 6 good pipelines at `fail`
  (prediction P2a failed): hubs one model makes on rows a reference handles are not
  explained away. That set is no longer blind for the reference rule either.
- The label-free result change detects change, not harm (AUC 0.58 between harmful and
  harmless faults); gate on it only where results must not change.
- The overlap with a reference model is a sign of harm, not a measure of it. On the third
  set ([preregistered](#a-third-held-out-set-preregistered)) it separated harmful from
  harmless faults at AUC 0.80, but at 0.02 it flagged 44% of the harmless ones (prediction
  P5b failed), mostly truncation and quantization, and it missed 17 of 24 harmful zero-row
  faults and every duplicate fault.
- The labeled gate judged against noise passed its test with 100 labeled queries; with
  fewer, most changes are inconclusive, and it assumes the labeled queries are a random
  sample of the traffic.
- The injected faults are six kinds applied one at a time to corpora of at most 25k
  documents; mixed faults and larger corpora were not injected. A lost id mapping moves no
  finding and is caught only with labeled queries.
- `query_gap` compares queries with the corpus they search, so it cannot be compared before
  and after a transform applied to the corpus alone: centering the corpus took it from 5.51 to
  `ok` while nDCG@10 fell ([agent case](agent_case.md)). Compare retrieval on labeled queries
  instead.
- Exactness on corpora over 20,000 rows was checked on synthetic data; the real-data exactness
  check covers the 76 smaller pairs.
- Sampled audits use uniform random rows (see [sampled audits](#3-sampled-audits) for why
  stratified sampling was not adopted).
- The Weaviate adapter was tested only against an embedded server in opt-in tests
  (`VECSCOPE_TEST_WEAVIATE=1`); remote Qdrant, Milvus and Weaviate servers were not tested.
  The `timeout=` option and the scrubbing of client errors were tested against an unreachable
  Qdrant port and a stubbed client, not against a slow server.
- The MCP guards were tested with low limits on small inputs, not by exhausting a machine, and
  a long-running server was not measured.
- The agent runs are one model on three tasks; they show the tools and the plan are usable,
  not that every agent follows them.

## Reproducing

- `pytest -q` runs the verification tests (`tests/test_new_features.py`,
  `tests/test_scale_exactness.py`, `tests/test_agent_workflow.py`,
  `tests/test_mcp_security.py`); `pytest -m slow` adds the 100k x 384 exactness checks.
- `python examples/benchmark_scale.py --rows 10000 50000 100000 --dim 768 --json scale.json`
  and `python examples/benchmark_sweep.py --rows 100000 --dim 768 --corpus clustered zipf
  --json sweep.json` re-measure the tables and record the machine in the JSON.
- `python examples/retrieval_quality.py` runs the labeled-data study for SciFact and
  NFCorpus.
- `python examples/validation/run.py`, `python examples/validation/faults.py` and
  `python examples/validation/report.py` re-run the held-out validation and the injected
  faults and rewrite [held-out validation](validation.md) (sentence-transformers and a GPU;
  several hours on one L4, resumable; `run.py --reaudit` audits the cached embeddings again
  after a change to vecscope), and
  `python examples/validation/calibrate_case.py --dataset xpqa-ja` the calibration case.
- The second held-out set: `run.py` on the seven datasets of
  `examples/validation/preregistration.md`, then `faults.py --role confirm`,
  `label_budget.py --role confirm` and `confirm.py`, which writes
  `examples/validation/confirm_summary.json`. `label_budget.py` without `--role` gives the
  budget on the first set.
- The third held-out set: `run.py` on the nine datasets of
  `examples/validation/preregistration_third.md`, then `third.py`, which injects the faults,
  scores P1 and P5 to P7 and writes `examples/validation/third_summary.json` (about two hours
  on one L4 after the embeddings).
- The 19-model, nine-dataset, query-finding and intervention runs use a harness that is not
  shipped with the package. The behavior they set is pinned by tests instead:
  - `tests/test_query_findings.py` (query_gap and query_shift)
  - `tests/test_plan_incremental.py` (the plan rules, including the dominant-dimension and
    duplicate cases)
  - `tests/test_regression_injection.py` (injected faults)
