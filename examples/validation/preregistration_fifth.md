# Fifth held-out set: preregistration

Written and committed on 2026-11-26, before any dataset below was downloaded or embedded.
The commit that adds this file freezes the code, the thresholds, the rules and the criteria.
The results are reported whatever they are; nothing is tuned on this set.

## Why a fifth set

The fourth set ([preregistration_fourth.md](preregistration_fourth.md), docs/verification.md)
passed every prediction, but left three limits:

1. The agreed top hits at 0.05 miss small losses below the reference's first result: 5 of
   20 harmful truncations and 2 of 11 harmful quantizations were flagged.
2. Two good pipelines stayed at `fail` on `effective_rank` (LitSearch), where the reference
   was at `warn`; the reference rule only explains a finding the reference reaches at the
   same level.
3. Every set so far used one reference model, bge-m3, and the faults were injected only into
   corpora of at most 25k documents.

Two changes were designed on the first four sets, all of which are now development data:

1. **A review tier for agreed top hits** (`reference_review`, `REFERENCE_REVIEW = 0.03`,
   `vecscope/results.py`). Below the tolerance, a lost share of agreed top hits above 0.03
   and 2 standard errors is listed for review; it never fails a gate, and `change_decision`
   says "a small loss is possible, check with labels". On the faults of the four sets
   (duplicates left out), it listed 80 of the 117 faults costing more than 0.02 nDCG@10 that
   0.05 did not flag, and 29 of the 244 harmless faults (per set 17%, 28%, 0% and 0%). The
   reference's top 2 and top 3, and the drop of the overlap with its top 5 and top 10, were
   also tried at 0.02 to 0.05; the first result at 0.03 sent the fewest harmless faults to
   review for the share of missed losses it caught.
2. **One step for `effective_rank`** (`PARTLY_EXPLAINED`, `vecscope/reference.py`). An
   `effective_rank` at `fail` where the reference is at `warn` is lowered to `warn`, not to
   `ok`: a narrow corpus lowers the effective rank of every model. The row-level findings
   (collapse, outliers, hubs) keep the strict rule. On the four sets it lowered the finding on
   10 good pipelines and 17 bad ones, and left 1 bad pipeline without a `fail` (on the first
   set the count is from the verdicts without the reference, which that set did not store).

The agreed top hits at 0.05, the retrieval agreement for hub findings (0.40) and the reference
rule for findings are unchanged. Their test here is a new reference model.

## Data (all public, none seen before)

| name | source | language | domain | license | documents | queries | role in the test |
|---|---|---|---|---|---|---|---|
| cqadupstack-mathematica | mteb/cqadupstack-mathematica | en | community QA | Apache-2.0 | 16,705 | all (804) | P11 to P13 |
| wikipedia-pt | mteb/WikipediaRetrievalMultilingual pt | pt | wikipedia | CC-BY-SA-3.0 | 13,500 | all (1,500) | P11 to P13 |
| wikipedia-fi | mteb/WikipediaRetrievalMultilingual fi | fi | wikipedia | CC-BY-SA-3.0 | 13,500 | all (1,500) | P11 to P13 |
| webfaq-est | mteb/WebFAQRetrieval est | et | web FAQ | CC-BY-4.0 | 12,891 | 1,000 seeded | P11 to P13 |
| webfaq-srp | mteb/WebFAQRetrieval srp | sr | web FAQ | CC-BY-4.0 | 5,824 | all (583) | P11 to P13 |
| codesearchnet-ruby | mteb/CodeSearchNetRetrieval ruby | code | code | MIT | 1,000 | all (1,000) | P11 to P13 |
| webfaq-ron | mteb/WebFAQRetrieval ron | ro | web FAQ | CC-BY-4.0 | 59,892 | 1,000 seeded | P13 |
| miracl-es | mteb/MIRACLRetrievalHardNegatives es | es | wikipedia | CC-BY-SA-4.0 | 146,750 | all (648) | P13 |

The first six have at most 25k documents, the limit of the injected faults; the last two are
larger, one about six times the largest corpus of the fourth set's faults. Models: the same 14
public models (registry.py). Labels: good = nDCG@10 at least 0.9 x the best model on the
dataset, bad = at most 0.5 x the best. Only the dataset sizes were read from the Hub's
metadata before this commit.

## Procedure

```bash
python examples/validation/run.py --datasets cqadupstack-mathematica wikipedia-pt \
    wikipedia-fi webfaq-est webfaq-srp codesearchnet-ruby webfaq-ron miracl-es
python examples/validation/fifth.py
```

Reference model: **gte-multilingual-base** for every pipeline, bge-m3 for
gte-multilingual-base itself, with the reference's own query embeddings. The earlier sets all
used bge-m3. Audits use the defaults (k=10, sample 100k, seed 0). Faults: the six of
faults.py, one at a time, on the good pipelines with at most 25k documents. For the
duplicates fault the reference embeds the duplicated rows too; for the others the reference
is the clean one. Each faulted run and its clean baseline are audited with
`reference_embeddings`, `reference_queries`, `result_sample=200` and the qrels, and diffed
with `retrieval_tolerances={"ndcg_at_k": 0.01}`, `retrieval_confidence=0.95` and
`reference_tolerance=0.05`. Harmful = full-set exact nDCG@10 drop above 0.01.

## Predictions and pass criteria

- **P11, agreed top hits with another reference**, on the faults other than duplicates.
  (a) The z of the lost share on the queries side separates harmful from harmless faults at
  AUC at least 0.85. (b) `reference_regressions` at 0.05 flags at least 65% of the faults
  whose nDCG@10 drop exceeds 0.02 and at most 15% of the harmless ones. Not testable with
  fewer than 10 harmless faults or 10 faults above 0.02.
- **P12, the review tier**, on the same faults. (a) Of the faults above 0.02 that
  `reference_regressions` did not flag, at least half are in `reference_review`. Not testable
  with fewer than 10. (b) At most 25% of the harmless faults are in `reference_review`. Not
  testable with fewer than 10.
- **P13, the two lowering rules with another reference** (retrieval agreement for hub
  findings and one step for `effective_rank`, together). (a) Of the good pipelines at `fail`
  with the reference before the two rules, at least half end without a `fail`. Not testable
  with fewer than 4. (b) Of the bad pipelines at `fail` with the reference before the two
  rules, at least 95% stay at `fail`. Not testable with fewer than 10. (c) Of the harmful
  faults the findings catch before the two rules (the run at `fail` where the clean run was
  not, or a finding worse than on the clean run, both with the reference), at least 95% are
  still caught with them. Not testable with fewer than 10.
- **P14, determinism.** For one good pipeline per fault dataset, the same audit with
  reference, result sample and qrels run twice gives identical JSON.

Reported without predictions: what each lowering rule does apart, the larger corpora, the
`query_hubness` chance rule and the labeled gate judged against noise.

A prediction that fails is reported as failed. A bug found in the analysis script after this
commit is fixed and named in the report, with the criteria unchanged.
