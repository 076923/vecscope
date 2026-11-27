# Changing the embedding model

**English** · [한국어](ko/case_model_change.md)

A team runs product search on multilingual-e5-base and is asked whether another model would
be better. This page replays that decision on public data, where full labels show the
answer, and records what vecscope said about each candidate with and without labeled
queries. The rows come from the preregistered sixth held-out set
([Verification](verification.md#a-sixth-held-out-set-preregistered), prediction P18), and
`examples/validation/case_model_change.py` writes them (`case_model_change.json`). Nothing
here was tuned on these rows.

## The data

[XPQA](https://huggingface.co/datasets/mteb/XPQARetrieval) holds real questions about Amazon
products and the answers that address them, short texts (median 48 to 67 characters) like
those of a product catalog. Three
languages, none used before by vecscope's validation:

| set | answers (corpus) | judged queries | baseline nDCG@10 |
|---|---|---|---|
| Italian (ita-ita) | 1,272 | 663 | 0.703 |
| Polish (pol-pol) | 1,579 | 785 | 0.480 |
| Portuguese (por-por) | 1,622 | 800 | 0.448 |

The baseline is multilingual-e5-base. The candidates are the 12 other public models of the
validation, English-only models included, because they are a frequent mistake. The
reference model is bge-m3, so it is not a candidate. The truth is the change of nDCG@10 on
every judged query.

## How it was run

Once, on the current model, with the reference and a sample of results:

```bash
vecscope audit e5_docs.npy --queries e5_queries.npy --qrels qrels.json \
  --reference bgem3_docs.npy --reference-queries bgem3_queries.npy \
  --result-sample 200 --json > baseline.json
```

Then for each candidate, on the same rows and queries:

```bash
vecscope audit cand_docs.npy --queries cand_queries.npy --qrels qrels.json \
  --result-sample 200 --baseline baseline.json --fail-on none \
  --reference-tolerance 0.05 \
  --retrieval-tolerance ndcg_at_k=0.01 --retrieval-confidence 0.95
```

- `--reference-tolerance 0.05` gives the label-free verdict: did the candidate lose the top
  hits the old model and the reference agreed on? (`reference_regressions` beyond 0.05,
  `reference_review` above 0.03.)
- `--retrieval-confidence 0.95` gives the labeled verdict. To show what a team with a small
  labeled set would see, it was run on 200 seeded subsets of 100 labeled queries, and the
  table gives how often it passed, was inconclusive or failed.

## Results

Italian:

| candidate | languages | nDCG@10 change | agreed top hits lost | label-free | labeled gate pass / inconclusive / fail |
|---|---|---|---|---|---|
| KURE-v1 | multilingual | +0.063 | 0.0% | — | 96% / 4% / 0% |
| multilingual-e5-large | multilingual | +0.040 | 2.2% | — | 78% / 22% / 0% |
| Qwen3-Embedding-0.6B | multilingual | +0.012 | 5.0% | `reference_review` | 28% / 69% / 3% |
| gte-multilingual-base | multilingual | -0.011 | 6.6% | `reference_regressions` | 4% / 88% / 8% |
| bge-base-en-v1.5 | English | -0.067 | 15.5% | `reference_regressions` | 0% / 33% / 67% |
| gte-base | English | -0.073 | 15.5% | `reference_regressions` | 0% / 26% / 74% |
| bge-small-en-v1.5 | English | -0.101 | 20.4% | `reference_regressions` | 0% / 5% / 95% |
| e5-base-v2 | English | -0.125 | 18.8% | `reference_regressions` | 0% / 1% / 99% |
| all-MiniLM-L6-v2 | English | -0.157 | 30.4% | `reference_regressions` | 0% / 0% / 100% |
| all-mpnet-base-v2 | English | -0.162 | 26.5% | `reference_regressions` | 0% / 0% / 100% |
| paraphrase-multilingual-MiniLM-L12-v2 | multilingual | -0.187 | 22.7% | `reference_regressions` | 0% / 0% / 100% |
| glove.6B.300d | English | -0.470 | 70.7% | `reference_regressions` | 0% / 0% / 100% |

Polish:

| candidate | languages | nDCG@10 change | agreed top hits lost | label-free | labeled gate pass / inconclusive / fail |
|---|---|---|---|---|---|
| KURE-v1 | multilingual | +0.038 | 0.0% | — | 58% / 42% / 0% |
| multilingual-e5-large | multilingual | +0.030 | 6.4% | `reference_regressions` | 56% / 44% / 0% |
| Qwen3-Embedding-0.6B | multilingual | -0.043 | 17.8% | `reference_regressions` | 0% / 48% / 52% |
| gte-multilingual-base | multilingual | -0.047 | 15.3% | `reference_regressions` | 0% / 50% / 50% |
| e5-base-v2 | English | -0.132 | 31.8% | `reference_regressions` | 0% / 0% / 100% |
| gte-base | English | -0.141 | 34.4% | `reference_regressions` | 0% / 0% / 100% |
| bge-base-en-v1.5 | English | -0.170 | 38.2% | `reference_regressions` | 0% / 0% / 100% |
| bge-small-en-v1.5 | English | -0.177 | 38.2% | `reference_regressions` | 0% / 0% / 100% |
| paraphrase-multilingual-MiniLM-L12-v2 | multilingual | -0.196 | 34.4% | `reference_regressions` | 0% / 0% / 100% |
| all-mpnet-base-v2 | English | -0.244 | 46.5% | `reference_regressions` | 0% / 0% / 100% |
| all-MiniLM-L6-v2 | English | -0.257 | 46.5% | `reference_regressions` | 0% / 0% / 100% |
| glove.6B.300d | English | -0.387 | 79.0% | `reference_regressions` | 0% / 0% / 100% |

Portuguese:

| candidate | languages | nDCG@10 change | agreed top hits lost | label-free | labeled gate pass / inconclusive / fail |
|---|---|---|---|---|---|
| KURE-v1 | multilingual | +0.048 | 0.6% | — | 78% / 22% / 0% |
| Qwen3-Embedding-0.6B | multilingual | +0.034 | 6.4% | `reference_regressions` | 45% / 55% / 0% |
| gte-multilingual-base | multilingual | +0.027 | 5.7% | `reference_regressions` | 40% / 60% / 0% |
| multilingual-e5-large | multilingual | -0.037 | 12.7% | `reference_regressions` | 1% / 50% / 49% |
| gte-base | English | -0.062 | 22.9% | `reference_regressions` | 0% / 42% / 58% |
| e5-base-v2 | English | -0.070 | 17.8% | `reference_regressions` | 0% / 18% / 82% |
| bge-small-en-v1.5 | English | -0.074 | 26.1% | `reference_regressions` | 0% / 28% / 72% |
| bge-base-en-v1.5 | English | -0.076 | 26.1% | `reference_regressions` | 0% / 23% / 77% |
| all-mpnet-base-v2 | English | -0.113 | 30.6% | `reference_regressions` | 0% / 2% / 98% |
| paraphrase-multilingual-MiniLM-L12-v2 | multilingual | -0.125 | 24.2% | `reference_regressions` | 0% / 2% / 98% |
| all-MiniLM-L6-v2 | English | -0.128 | 29.9% | `reference_regressions` | 0% / 2% / 98% |
| glove.6B.300d | English | -0.322 | 74.5% | `reference_regressions` | 0% / 0% / 100% |

## What it shows

- **The label-free signal ranks the candidates.** Every candidate that lost 0.03 nDCG@10 or
  more was flagged, and the share of agreed top hits lost grew with the loss: 13% to 18% for
  the multilingual models that lost a little, 15% to 79% for the English-only models.
- **It also flagged three of the seven improvements.** multilingual-e5-large on Polish
  (+0.030), Qwen3-Embedding-0.6B and gte-multilingual-base on Portuguese (+0.034, +0.027)
  lost 5.7% to 6.4% of the agreed top hits, just over 0.05. A new model swaps many results;
  it can lose some of the old model's agreed hits and still find more elsewhere, and the diff
  counts only what was lost. Across the nine datasets of the set this happened on 5 of 27
  improvements, and the preregistered criterion failed. Read the flag on a model change as
  "this candidate disagrees where the old model and the reference agreed, check with labels".
- **The labeled gate almost never got the direction wrong.** With 100 labeled queries, a
  candidate that lost 0.03 or more passed in at most 1% of the subsets, and an improvement
  failed in at most 3% (Qwen3-Embedding-0.6B on Italian, +0.012). Otherwise it failed the
  losses, passed the improvements or said inconclusive. Small differences were often
  inconclusive:
  Polish e5-large (+0.030) passed in 56% of the subsets. The run then reports how many labeled
  queries would settle it (`retrieval_confidence_checks`).
- **KURE-v1 looks perfect to the reference because it is the reference's fine-tune.** It lost
  0.0% to 0.6% of the agreed top hits on every dataset of the set. Here it was also the best
  candidate, but the agreed top hits did not show it; they cannot judge a model from the
  reference's own family.
- **The absolute verdict is not a reason to switch.** multilingual-e5-base was at `fail` on
  all three sets (`ood_queries`, `collapse` or `query_gap`) and is still among the better
  models here; the reference model and the lowering rules did not clear those findings. The
  diff against the baseline decides a change, not the verdict of one run.

## What to do

1. Embed the same rows and queries with each candidate and with a reference from another
   model family.
2. Audit the current model once with the reference and `--result-sample`, and keep the JSON.
3. Drop candidates with a large lost share of agreed top hits: on this data every one over
   0.10 lost nDCG@10.
4. Decide among the rest with labeled queries and `--retrieval-confidence 0.95`. If the
   gate is inconclusive, label the number of queries it asks for before switching.

## What this does not show

- XPQA stands in for product search: one public source, three languages, short answers.
  Production query logs and catalogs were not measured.
- The baseline is one model, multilingual-e5-base.
- The labeled gate assumes the labeled queries are a random sample of the traffic.
