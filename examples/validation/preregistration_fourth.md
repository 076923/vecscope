# Fourth held-out set: preregistration

Written and committed on 2026-11-26, before any dataset below was downloaded or embedded.
The commit that adds this file freezes the code, the thresholds, the rules and the criteria.
The results are reported whatever they are; nothing is tuned on this set.

## Why a fourth set

The third set ([preregistration_third.md](preregistration_third.md), docs/verification.md)
left two limits:

1. The overlap with a reference model's results, read as a drop of the whole top-k overlap,
   flagged 44% of the harmless faults at 0.02 (P5b failed): truncated and quantized vectors
   moved away from the reference without costing nDCG@10.
2. A hub finding a good model makes stays at `fail` with the reference rule: 6 good
   pipelines on the second set and 6 on the third.

Two changes were designed on the first three sets, all of which are now development data:

1. **Agreed top hits** (`reference_top_hit`, `reference_regressions` at
   `--reference-tolerance 0.05`, `vecscope/results.py`). Of the sampled queries whose
   baseline top-10 holds the reference's first result, the share whose new top-10 no longer
   does; flagged beyond the tolerance and 2 standard errors. Only the baseline needs the
   reference. On 725 injected faults of the three sets (duplicates left out), its z
   separated harmful from harmless faults at AUC 0.91 (0.87 to 0.96 per set), where the
   overlap drop did at 0.86 and the changed share at 0.51. At 0.05 it flagged 72% to 83% of
   the faults above 0.02 per set and 5% to 8% of the harmless ones; the overlap drop at 0.02
   flagged 24% to 44% of the harmless ones. The reference's top 2, top 3, top 5 and top 10,
   tolerances from 0.03 to 0.06 and z of 3 were also tried; the first result at 0.05 had the
   fewest false alarms with at least 70% of the large drops caught on every set.
2. **Retrieval agreement for hub findings** (`reference.agreement`, `AGREEMENT = 0.40`,
   `vecscope/reference.py`). With queries and reference queries, the mean share of this
   pipeline's top-10 the reference returns too, over 200 sampled queries. A hub finding
   (hubness, query_hubness, antihubs) still at `fail` after the reference rule is lowered to
   `warn` when the agreement is 0.40 or more. On 490 pipelines of the three sets the
   agreement separated good from bad pipelines at AUC 0.999. No bad pipeline at `fail`
   reached 0.32; of the 12 good pipelines left at `fail` with the reference on the second
   and third sets, 9 agreed at 0.40 or more and the rule leaves them without a `fail`.

The `query_hubness` chance rule, the labeled gate judged against noise and the reference rule
for findings are unchanged and are reported without predictions.

## Data (all public, none seen before)

| name | source | language | domain | license | queries | role in the test |
|---|---|---|---|---|---|---|
| cqadupstack-android | mteb/cqadupstack-android | en | community QA | Apache-2.0 | all (699) | P8, P9 |
| feedbackqa | mteb/FeedbackQARetrieval | en | public health and government FAQ | Apache-2.0 | all | P8, P9 |
| hunsum2 | mteb/HunSum2AbstractiveRetrieval | hu | news | CC-BY-4.0 | all | P8, P9 |
| wikipedia-it | mteb/WikipediaRetrievalMultilingual it | it | wikipedia | CC-BY-SA-3.0 | all | P8, P9 |
| wikipedia-fa | mteb/WikipediaRetrievalMultilingual fa | fa | wikipedia | CC-BY-SA-3.0 | all | P8, P9 |
| webfaq-msa | mteb/WebFAQRetrieval msa | ms | web FAQ | CC-BY-4.0 | all (643) | P8, P9 |
| webfaq-heb | mteb/WebFAQRetrieval heb | he | web FAQ | CC-BY-4.0 | 1000 seeded | P9 |
| webfaq-ell | mteb/WebFAQRetrieval ell | el | web FAQ | CC-BY-4.0 | 1000 seeded | P9 |
| webfaq-hun | mteb/WebFAQRetrieval hun | hu | web FAQ | CC-BY-4.0 | 1000 seeded | P9 |
| rubq | mteb/RuBQRetrieval | ru | wikipedia QA | CC-BY-SA-4.0 | 1000 seeded | P9 |
| litsearch | mteb/LitSearchRetrieval | en | science | MIT | 120 seeded | P9 |

The first six have at most 25k documents, the limit of the injected faults. WebFAQ is where
good models made the hubs left at `fail`; Hebrew, Greek and Hungarian were not used before.
Models: the same 14 public models (registry.py). Labels: good = nDCG@10 at least 0.9 x the
best model on the dataset, bad = at most 0.5 x the best. Only the dataset sizes and column
names were read from the Hub's metadata before this commit.

## Procedure

```bash
python examples/validation/run.py --datasets cqadupstack-android feedbackqa hunsum2 \
    wikipedia-it wikipedia-fa webfaq-msa webfaq-heb webfaq-ell webfaq-hun rubq litsearch
python examples/validation/fourth.py
```

Reference model: bge-m3 for every pipeline, multilingual-e5-base for bge-m3 itself, with the
reference's own query embeddings. Audits use the defaults (k=10, sample 100k, seed 0).
Faults: the six of faults.py, one at a time, on the good pipelines with at most 25k
documents. For the duplicates fault the reference embeds the duplicated rows too; for the
others the reference is the clean one. Each faulted run and its clean baseline are audited
with `reference_embeddings`, `reference_queries`, `result_sample=200` and the qrels, and
diffed with `retrieval_tolerances={"ndcg_at_k": 0.01}`, `retrieval_confidence=0.95` and
`reference_tolerance=0.05`. Harmful = full-set exact nDCG@10 drop above 0.01. The overlap-drop
rule of the third set is recomputed on the same diffs from `result_agreement.queries.reference`
(drop above 0.02 and z above 2).

## Predictions and pass criteria

- **P8, agreed top hits as a sign of harm**, on the faults other than duplicates.
  (a) The z of the lost share on the queries side separates harmful from harmless faults at
  AUC at least 0.85. (b) `reference_regressions` at 0.05 flags at least 65% of the faults
  whose nDCG@10 drop exceeds 0.02 and at most 15% of the harmless ones. (c) It flags harmless
  faults at most half as often as the overlap-drop rule at 0.02 on the same faults. Not
  testable with fewer than 10 harmless faults or 10 faults above 0.02. Duplicates are
  reported apart; the prediction for them is that the rule does not see them.
- **P9, retrieval agreement for hub findings.** (a) Of the good pipelines at `fail` with the
  reference before the agreement rule, at least half end without a `fail`. Not testable with
  fewer than 4. (b) Of the bad pipelines at `fail` with the reference, at least 95% stay at
  `fail`. Not testable with fewer than 10. (c) Of the harmful faults the findings catch
  before the agreement rule (the run at `fail` where the clean run was not, or a finding
  worse than on the clean run, both with the reference), at least 95% are still caught with
  it. Not testable with fewer than 10.
- **P10, determinism.** For one good pipeline per fault dataset, the same audit with
  reference, result sample and qrels run twice gives identical JSON.

A prediction that fails is reported as failed. A bug found in the analysis script after this
commit is fixed and named in the report, with the criteria unchanged.
