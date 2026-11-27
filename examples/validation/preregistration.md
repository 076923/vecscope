# Second held-out set: preregistration

Written and committed on 2026-11-24, before any dataset below was downloaded or embedded.
The commit that adds this file freezes the code, the thresholds, the rules and the criteria.
The results are reported whatever they are; nothing is tuned on this set. Anything learned
from it goes into a later version and needs another new set to confirm.

## Why a second set

The first held-out set (19 datasets, `role="heldout"`, docs/validation.md) was read before
four changes, so it cannot confirm them:

1. `query_hubness` is judged against the skewness a hub-free index shows with the same
   number of queries (`details.null_skewness`); levels chance alone reaches are not judged.
2. `--reference`: findings a reference model reaches on the same rows are attributed to the
   rows (`vecscope/reference.py`). On the first set, with bge-m3 as reference, good
   pipelines at fail went from 28 to 21 and bad pipelines at fail stayed at 54 of 67. That
   rule was designed after reading this set.
3. `--result-sample` / `--result-tolerance`: label-free change of the sampled top-k
   against a baseline (`vecscope/results.py`). On the first set's 594 injected faults it
   did not separate harmful from harmless changes (AUC 0.58), so it is offered only as a
   change detector.
4. A labeled-query budget: on the first set's faults, with 100 labeled queries a 0.01 nDCG
   tolerance caught 92% of harmful faults (100% of those dropping more than 0.05), with
   false alarms on 24% of harmless ones (`label_budget.py`).

## Data (all public, none seen before)

| name | source | language | domain | license | queries |
|---|---|---|---|---|---|
| webfaq-kor | mteb/WebFAQRetrieval kor | ko | web FAQ | CC-BY-4.0 | 1000 seeded |
| webfaq-tha | mteb/WebFAQRetrieval tha | th | web FAQ | CC-BY-4.0 | 1000 seeded |
| webfaq-vie | mteb/WebFAQRetrieval vie | vi | web FAQ | CC-BY-4.0 | 1000 seeded |
| jagovfaqs | mteb/JaGovFaqsRetrieval | ja | government FAQ | CC-BY-4.0 | all |
| gerdalir-small | mteb/GerDaLIRSmall | de | legal | MIT | 1000 seeded |
| cqadupstack-webmasters | mteb/cqadupstack-webmasters | en | community QA | Apache-2.0 | all |
| turhistquad | mteb/TurHistQuadRetrieval | tr | history | MIT | all |

Two corpora have more than 100k documents (WebFAQ kor 102k, vie 124k), two languages
(Thai, Turkish) and three scripts are new to the validation. Models: the same 14 public
models as the first set (registry.py). Labels: good = nDCG@10 at least 0.9 x the best
model on the dataset, bad = at most 0.5 x the best, as before.

## Procedure

```bash
python examples/validation/run.py --datasets webfaq-kor webfaq-tha webfaq-vie jagovfaqs \
    gerdalir-small cqadupstack-webmasters turhistquad
python examples/validation/faults.py --role confirm            # corpora up to 25k documents
python examples/validation/label_budget.py --role confirm
python examples/validation/confirm.py                          # scores P1-P4
```

Reference model: bge-m3 for every pipeline, multilingual-e5-base for bge-m3 itself, with
the reference's own query embeddings. Audits use the defaults (k=10, sample 100k, seed 0).

## Predictions and pass criteria

- **P1, query_hubness against chance.** Good pipelines flagged by query_hubness with the
  null rule are no more than with the thresholds alone, and of the bad pipelines the
  thresholds alone put at fail on query_hubness, the null rule keeps at least 90% at fail.
- **P2, reference model.** (a) Good pipelines at fail with the reference are at most 75% of
  those without it. (b) Bad pipelines at fail with the reference are at least 95% of those
  without it. Reported as well: flagged and all-ok counts, which findings were explained,
  and the good pipelines still at fail.
- **P3, labeled-query budget.** On the faults of the good pipelines with up to 25k
  documents, with 100 labeled queries the 0.01 nDCG tolerance catches at least 85% of the
  harmful faults and at least 90% of those whose full-set drop exceeds 0.02.
- **P4, label-free result change.** Every injected fault changes the sampled top-k results
  (changed share above 0 on the queries or the rows), and an identical re-run changes none.

A prediction that fails is reported as failed. A bug found in the analysis scripts after
this commit is fixed and named in the report, with the criteria unchanged.
