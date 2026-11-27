# Sixth held-out set: preregistration

Written and committed on 2026-11-27, before any dataset below was downloaded or embedded.
The commit that adds this file freezes the code, the thresholds, the rules and the criteria.
The results are reported whatever they are; nothing is tuned on this set.

## Why a sixth set

The fifth set ([preregistration_fifth.md](preregistration_fifth.md), docs/verification.md)
passed every prediction, but three questions were left:

1. **Small counts.** Some criteria passed on few cases. With 95% intervals
   (`intervals.py`, Wilson for counts, a pipeline bootstrap for the faults):
   - P12a, 12 of 23 missed losses listed for review: 52%, interval 32% to 70%. It is not
     shown to be above 50%.
   - P13a, 4 of 5 good pipelines cleared: interval 38% to 96%.
   - P13b, 17 of 17 bad pipelines kept at `fail`: interval 82% to 100%. Pooled with the
     fourth set's 38 of 38, 55 of 55: 93% to 100%.
   - P11b, 80 of 103 large losses flagged: 69% to 86% (bootstrap); 1 of 47 harmless: 0% to
     7%.
2. **Faults only in small corpora.** Every fault so far went into corpora of at most 25k
   documents.
3. **Only injected faults.** The most common change a team makes is replacing the embedding
   model. It was never judged against labels.

**No rule changed after the fifth set.** This set replicates the shipped rules on new data.
The reference model goes back to bge-m3 (the reference of sets 2 to 4), so the rules are
tested with both reference models used so far.

## Data (all public, none seen before)

| name | source | language | domain | license | documents | queries |
|---|---|---|---|---|---|---|
| xpqa-it | mteb/XPQARetrieval ita-ita | it | product answers | CDLA-Sharing-1.0 | 1,272 | all (663) |
| xpqa-pl | mteb/XPQARetrieval pol-pol | pl | product answers | CDLA-Sharing-1.0 | 1,579 | all (785) |
| xpqa-pt | mteb/XPQARetrieval por-por | pt | product answers | CDLA-Sharing-1.0 | 1,622 | all (800) |
| wikipedia-nl | mteb/WikipediaRetrievalMultilingual nl | nl | wikipedia | CC-BY-SA-3.0 | 13,500 | all (1,500) |
| wikipedia-sv | mteb/WikipediaRetrievalMultilingual sv | sv | wikipedia | CC-BY-SA-3.0 | 13,500 | all (1,500) |
| webfaq-lit | mteb/WebFAQRetrieval lit | lt | web FAQ | CC-BY-4.0 | 18,363 | 1,000 seeded |
| codesearchnet-php | mteb/CodeSearchNetRetrieval php | code | code | MIT | 1,000 | all (1,000) |
| webfaq-ces | mteb/WebFAQRetrieval ces | cs | web FAQ | CC-BY-4.0 | 72,308 | 1,000 seeded |
| miracl-zh | mteb/MIRACLRetrievalHardNegatives zh | zh | wikipedia | CC-BY-SA-4.0 | 81,309 | all (393) |

The query counts above are the Hub's metadata for the query configs. A run uses only the
judged queries, so it may use fewer. Only the sizes were read from the Hub before this
commit. XPQA (Amazon product questions and answers) stands in for short product text; the
other XPQA languages were used in the first set, and kor-kor in the design data.

Models: the same 14 public models (registry.py). Labels: good = nDCG@10 at least 0.9 x the
best model on the dataset, bad = at most 0.5 x the best.

## Procedure

```bash
python examples/validation/run.py --datasets xpqa-it xpqa-pl xpqa-pt wikipedia-nl \
    wikipedia-sv webfaq-lit codesearchnet-php webfaq-ces miracl-zh
python examples/validation/sixth.py
```

Reference model: **bge-m3**, gte-multilingual-base for bge-m3 itself. Audits, faults and
diffs exactly as in the fifth set, but the faults go into every good pipeline of the set,
the two corpora over 25k documents included. Harmful = full-set exact nDCG@10 drop above
0.01.

Model change (P18): on each dataset, the baseline is multilingual-e5-base, a common
multilingual default. Each other model is a candidate, except the reference. The candidate
is audited with the same reference, queries, qrels and result sample, and diffed against
the baseline as in the fifth set. The full-label nDCG@10 difference is the truth.

## Predictions and pass criteria

Each result is reported with its 95% interval (`intervals.py`). The pass criteria apply to
the point values, as in the earlier sets.

- **P15, the faults in corpora of at most 25k documents** (duplicates left out). These are
  the criteria of P11 and P12:
  - (a) The z of the lost share of agreed top hits separates harmful from harmless faults at
    AUC at least 0.85.
  - (b) `reference_regressions` at 0.05 flags at least 65% of the faults whose drop exceeds
    0.02, and at most 15% of the harmless ones.
  - (c) `reference_review` lists at least half of the faults above 0.02 that (b) missed.
  - (d) `reference_review` lists at most 25% of the harmless faults.
  - Not testable with fewer than 10 cases in a group.
- **P16, the faults in the two corpora over 25k documents.** The same four criteria as
  P15, on webfaq-ces and miracl-zh, with the same testability limits.
- **P17, the two lowering rules** (retrieval agreement for hub findings, one step for
  `effective_rank`):
  - (a) Of the good pipelines at `fail` with the reference before the two rules, at least
    half end without a `fail`. Not testable with fewer than 4.
  - (b) Of the bad pipelines at `fail` with the reference before the two rules, at least 95%
    stay at `fail`. Not testable with fewer than 10.
  - (c) Of the harmful faults the findings catch before the two rules, at least 95% are
    still caught with them. Not testable with fewer than 10. This covers all faults, the
    large corpora included.
- **P18, a model change**, multilingual-e5-base replaced by each other model except the
  reference:
  - (a) `reference_regressions` at 0.05 flags at least 65% of the changes that lose more
    than 0.02 nDCG@10, and at most 15% of the changes that lose 0.01 or less (improvements
    included).
  - (a-multi) The same criteria on the multilingual candidates alone. English-only models on
    non-English data lose a lot and are easy to flag; this sub-criterion is the harder
    test. Not testable with fewer than 10 cases in a group.
  - (b) With 100 seeded labeled queries and `retrieval_confidence=0.95` (200 subsets each),
    harmful changes pass at most 3% of the time, and at most half as often as with the mean
    alone.
  - (c) Harmless changes fail at most 0.75 times as often as with the mean alone.
  - (b) and (c) are not testable with fewer than 10 harmful or 10 harmless changes.
  - These criteria were carried over from the faults (P11b, P6) and not fitted on model
    changes. The only model-change numbers seen before this commit come from a dry run of
    `sixth.py` on one development dataset (webfaq-srp, fifth set), run to check the code:
    8 of 8 changes losing more than 0.02 were flagged, 0 of 3 harmless ones. Seven of the 8
    were English-only models.
- **P19, determinism.** For one good pipeline per dataset, the same audit with reference,
  result sample and qrels run twice gives identical JSON.

Reported without predictions:
- the intervals
- each lowering rule apart
- the labeled gate on the faults
- the `query_hubness` chance rule
- per dataset, the model-change rows, which are the data for a case page on XPQA

A prediction that fails is reported as failed. A bug found in the analysis script after this
commit is fixed and named in the report, with the criteria unchanged.
