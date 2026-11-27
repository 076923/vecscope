# Changing the embedding model

**English** · [한국어](ko/case_model_change.md)

A team runs product search on multilingual-e5-base and is asked whether another model would
be better. This page replays that decision on public data with full labels and records what
vecscope said about each candidate with and without labeled queries. The rows come from a
preregistered held-out set ([Verification](verification.md#preregistered-held-out-sets));
`examples/validation/case_model_change.py` writes them (`case_model_change.json`). The net
drop threshold was chosen on data that includes these rows, so the page shows the rules at
work, not a held-out test of them; that test is in
[Verification](verification.md#agreed-top-hits).

## The data

[XPQA](https://huggingface.co/datasets/mteb/XPQARetrieval) holds real questions about Amazon
products and their answers: short texts (median 48 to 67 characters) like a product
catalog's, in three languages:

| set | answers (corpus) | judged queries | baseline nDCG@10 |
|---|---|---|---|
| Italian (ita-ita) | 1,272 | 663 | 0.703 |
| Polish (pol-pol) | 1,579 | 785 | 0.480 |
| Portuguese (por-por) | 1,622 | 800 | 0.448 |

The baseline is multilingual-e5-base. The candidates are the validation's 12 other public
models, English-only ones included because they are a frequent mistake. bge-m3 is the
reference, not a candidate. The truth is the change of nDCG@10 on every judged query.

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
  --reference-tolerance 0.03 \
  --retrieval-tolerance ndcg_at_k=0.01 --retrieval-confidence 0.95
```

- `--reference-tolerance 0.03` gives the label-free verdict: did the candidate lose, on
  balance, the top hits the old model and the reference agreed on? (`reference_regressions`
  for a net drop beyond 0.03, `reference_review` above 0.015, both beyond 2 standard
  errors.)
- `--retrieval-confidence 0.95` gives the labeled verdict. To show what a team with a small
  labeled set would see, it ran on 200 seeded subsets of 100 labeled queries; the table gives
  how often it passed, was inconclusive or failed.

## Results

Italian:

| candidate | languages | nDCG@10 change | agreed top hits lost | net drop | label-free | labeled gate pass / inconclusive / fail |
|---|---|---|---|---|---|---|
| KURE-v1 | multilingual | +0.063 | 0.0% | -0.095 | — | 96% / 4% / 0% |
| multilingual-e5-large | multilingual | +0.040 | 2.2% | -0.020 | — | 78% / 22% / 0% |
| Qwen3-Embedding-0.6B | multilingual | +0.012 | 5.0% | -0.010 | — | 28% / 69% / 3% |
| gte-multilingual-base | multilingual | -0.011 | 6.6% | +0.025 | — | 4% / 88% / 8% |
| bge-base-en-v1.5 | English | -0.067 | 15.5% | +0.105 | `reference_regressions` | 0% / 33% / 67% |
| gte-base | English | -0.073 | 15.5% | +0.120 | `reference_regressions` | 0% / 26% / 74% |
| bge-small-en-v1.5 | English | -0.101 | 20.4% | +0.145 | `reference_regressions` | 0% / 5% / 95% |
| e5-base-v2 | English | -0.125 | 18.8% | +0.150 | `reference_regressions` | 0% / 1% / 99% |
| all-MiniLM-L6-v2 | English | -0.157 | 30.4% | +0.250 | `reference_regressions` | 0% / 0% / 100% |
| all-mpnet-base-v2 | English | -0.162 | 26.5% | +0.220 | `reference_regressions` | 0% / 0% / 100% |
| paraphrase-multilingual-MiniLM-L12-v2 | multilingual | -0.187 | 22.7% | +0.180 | `reference_regressions` | 0% / 0% / 100% |
| glove.6B.300d | English | -0.470 | 70.7% | +0.630 | `reference_regressions` | 0% / 0% / 100% |

Polish:

| candidate | languages | nDCG@10 change | agreed top hits lost | net drop | label-free | labeled gate pass / inconclusive / fail |
|---|---|---|---|---|---|---|
| KURE-v1 | multilingual | +0.038 | 0.0% | -0.195 | — | 58% / 42% / 0% |
| multilingual-e5-large | multilingual | +0.030 | 6.4% | -0.010 | — | 56% / 44% / 0% |
| Qwen3-Embedding-0.6B | multilingual | -0.043 | 17.8% | +0.045 | — | 0% / 48% / 52% |
| gte-multilingual-base | multilingual | -0.047 | 15.3% | +0.025 | — | 0% / 50% / 50% |
| e5-base-v2 | English | -0.132 | 31.8% | +0.185 | `reference_regressions` | 0% / 0% / 100% |
| gte-base | English | -0.141 | 34.4% | +0.200 | `reference_regressions` | 0% / 0% / 100% |
| bge-base-en-v1.5 | English | -0.170 | 38.2% | +0.220 | `reference_regressions` | 0% / 0% / 100% |
| bge-small-en-v1.5 | English | -0.177 | 38.2% | +0.260 | `reference_regressions` | 0% / 0% / 100% |
| paraphrase-multilingual-MiniLM-L12-v2 | multilingual | -0.196 | 34.4% | +0.215 | `reference_regressions` | 0% / 0% / 100% |
| all-mpnet-base-v2 | English | -0.244 | 46.5% | +0.325 | `reference_regressions` | 0% / 0% / 100% |
| all-MiniLM-L6-v2 | English | -0.257 | 46.5% | +0.330 | `reference_regressions` | 0% / 0% / 100% |
| glove.6B.300d | English | -0.387 | 79.0% | +0.595 | `reference_regressions` | 0% / 0% / 100% |

Portuguese:

| candidate | languages | nDCG@10 change | agreed top hits lost | net drop | label-free | labeled gate pass / inconclusive / fail |
|---|---|---|---|---|---|---|
| KURE-v1 | multilingual | +0.048 | 0.6% | -0.200 | — | 78% / 22% / 0% |
| Qwen3-Embedding-0.6B | multilingual | +0.034 | 6.4% | -0.075 | — | 45% / 55% / 0% |
| gte-multilingual-base | multilingual | +0.027 | 5.7% | -0.040 | — | 40% / 60% / 0% |
| multilingual-e5-large | multilingual | -0.037 | 12.7% | +0.070 | `reference_regressions` | 1% / 50% / 49% |
| gte-base | English | -0.062 | 22.9% | +0.100 | `reference_regressions` | 0% / 42% / 58% |
| e5-base-v2 | English | -0.070 | 17.8% | +0.095 | `reference_regressions` | 0% / 18% / 82% |
| bge-small-en-v1.5 | English | -0.074 | 26.1% | +0.120 | `reference_regressions` | 0% / 28% / 72% |
| bge-base-en-v1.5 | English | -0.076 | 26.1% | +0.135 | `reference_regressions` | 0% / 23% / 77% |
| all-mpnet-base-v2 | English | -0.113 | 30.6% | +0.150 | `reference_regressions` | 0% / 2% / 98% |
| paraphrase-multilingual-MiniLM-L12-v2 | multilingual | -0.125 | 24.2% | +0.120 | `reference_regressions` | 0% / 2% / 98% |
| all-MiniLM-L6-v2 | English | -0.128 | 29.9% | +0.170 | `reference_regressions` | 0% / 2% / 98% |
| glove.6B.300d | English | -0.322 | 74.5% | +0.575 | `reference_regressions` | 0% / 0% / 100% |

## What it shows

- **The label-free signal ranks the candidates.** The net drop grew with the loss: 0.095 to
  0.63 for the English-only models, 0.12 to 0.22 for paraphrase-multilingual-MiniLM-L12-v2,
  below zero for every improvement. Every candidate that lost 0.06 nDCG@10 or more was
  flagged.
- **It flags no improvement here.** Every candidate that improved nDCG@10 had a net drop
  below zero, even those that lost 5.7% to 6.4% of the agreed top hits
  (multilingual-e5-large on Polish, Qwen3-Embedding-0.6B and gte-multilingual-base on
  Portuguese): a new model loses some of the old model's agreed hits and finds more of the
  reference's first results elsewhere.
- **It misses small losses between multilingual models.** Of the three multilingual
  candidates that lost 0.037 to 0.047, it flagged one (multilingual-e5-large on Portuguese,
  net drop 0.070); Qwen3-Embedding-0.6B and gte-multilingual-base on Polish (net drop 0.045
  and 0.025, within 2 standard errors) were neither flagged nor listed for review. Held-out
  data shows the same: 1 of 13 such losses flagged. A clean label-free result is not
  evidence that nDCG@10 held.
- **The labeled gate almost never got the direction wrong.** With 100 labeled queries, a
  candidate that lost 0.03 or more passed in at most 1% of the subsets, and an improvement
  failed in at most 3% (Qwen3-Embedding-0.6B on Italian, +0.012). Small differences were
  often inconclusive: Polish e5-large (+0.030) passed in 56% of the subsets, and the two
  Polish losses the net drop missed failed in about half and were inconclusive in the rest,
  never passed. An inconclusive run reports how many labeled queries would settle it
  (`retrieval_confidence_checks`).
- **KURE-v1 looks perfect to the reference because it is the reference's fine-tune.** It lost
  0.0% to 0.6% of the agreed top hits on every dataset of the set. It was also the best
  candidate, but the agreed top hits cannot show that: they cannot judge a model from the
  reference's own family.
- **The absolute verdict is not a reason to switch.** multilingual-e5-base was at `fail` on
  all three sets (`ood_queries`, `collapse` or `query_gap`) and is still among the better
  models here; the reference model and the lowering rules did not clear those findings. The
  baseline diff decides a change, not one run's verdict.

## What to do

1. Embed the same rows and queries with each candidate and with a reference from another
   model family.
2. Audit the current model once with the reference and `--result-sample`, and keep the JSON.
3. Drop candidates in `reference_regressions`: on this data each of them lost 0.037 nDCG@10
   or more. Keep the others as candidates, not as safe: two unflagged Polish candidates lost
   0.043 and 0.047.
4. Decide among the rest with labeled queries and `--retrieval-confidence 0.95`. If the
   gate is inconclusive, label the number of queries it asks for before switching.

## What this does not show

- XPQA stands in for product search: one public source, three languages, short answers.
  Production query logs and catalogs were not measured.
- The baseline is one model, multilingual-e5-base.
- The labeled gate assumes the labeled queries are a random sample of the traffic.
