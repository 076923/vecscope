# Third held-out set: preregistration

Written and committed on 2026-11-25, before any dataset below was downloaded or embedded.
The commit that adds this file freezes the code, the thresholds, the rules and the criteria.
The results are reported whatever they are; nothing is tuned on this set.

## Why a third set

The second set ([preregistration.md](preregistration.md), docs/verification.md) left three
things open, and two changes were designed after reading the first set again:

1. `query_hubness` judged against chance was never tested: every dataset of the second set
   had 1,000 or more queries, so chance stayed below warn.
2. The overlap with a reference model's results (`reference_overlap`, `--reference-tolerance`,
   `vecscope/results.py`). On the first set's injected faults, duplicates left out, the paired
   drop of that overlap over 200 sampled queries separated harmful from harmless changes at
   AUC 0.875, where the share of changed results did at 0.56. At a 0.02 tolerance it flagged
   79% of the faults that cost more than 0.02 nDCG@10 and 24% of the harmless ones. Designed
   on the first set.
3. The labeled gate judged against the noise of its queries (`--retrieval-confidence`,
   `diff_evidence(retrieval_confidence=...)`). On the first set's faults with 100 labeled
   queries at 0.95: harmful changes passed 1% of the time (8% for the mean alone) and
   harmless ones failed 14% (24%); the rest were inconclusive. Designed on the first set.

The reference-model rule for findings (`--reference`) is unchanged since its prediction P2a
failed on the second set; it is reported here without a prediction.

## Data (all public, none seen before)

| name | source | language | domain | license | queries | role in the test |
|---|---|---|---|---|---|---|
| miracl-ko | mteb/MIRACLRetrievalHardNegatives ko | ko | wikipedia | CC-BY-SA-4.0 | all (213) | P1 |
| miracl-th | mteb/MIRACLRetrievalHardNegatives th | th | wikipedia | CC-BY-SA-4.0 | 200 seeded | P1 |
| scidocs | mteb/scidocs | en | science | CC-BY-SA-4.0 | 60 seeded | P1 |
| webfaq-ind | mteb/WebFAQRetrieval ind | id | web FAQ | CC-BY-4.0 | 150 seeded | P1 |
| arguana | mteb/arguana | en | arguments | CC-BY-SA-4.0 | all | P5, P6 |
| stackoverflowqa | mteb/StackOverflowQA | en | code QA | MIT | all | P5, P6 |
| wikipedia-bn | mteb/WikipediaRetrievalMultilingual bn | bn | wikipedia | CC-BY-SA-3.0 | all | P5, P6 |
| wikipedia-cs | mteb/WikipediaRetrievalMultilingual cs | cs | wikipedia | CC-BY-SA-3.0 | all | P5, P6 |
| autorag-ko | mteb/AutoRAGRetrieval | ko | five domains | MIT | all (114) | P5, P6 |

The first four have few queries over 25k to 117k documents, so chance alone gives
query_hubness a skewness above warn (4). The other five have at most 25k documents, the
limit of the injected faults. Models: the same 14 public models (registry.py). Labels: good =
nDCG@10 at least 0.9 x the best model on the dataset, bad = at most 0.5 x the best.

## Procedure

```bash
python examples/validation/run.py --datasets miracl-ko miracl-th scidocs webfaq-ind arguana \
    stackoverflowqa wikipedia-bn wikipedia-cs autorag-ko
python examples/validation/third.py
```

Reference model: bge-m3 for every pipeline, multilingual-e5-base for bge-m3 itself, with the
reference's own query embeddings. Audits use the defaults (k=10, sample 100k, seed 0).
Faults: the six of faults.py, one at a time, on the good pipelines with at most 25k
documents. For the duplicates fault the reference embeds the duplicated rows too (the fault
is in the rows, which a reference made from the same texts shares); for the others the
reference is the clean one (the fault is in the pipeline under test). Each faulted run and
its clean baseline are audited with `reference_embeddings`, `result_sample=200` and the
qrels, and diffed with `retrieval_tolerances={"ndcg_at_k": 0.01}`,
`retrieval_confidence=0.95` and `reference_tolerance=0.02`. Harmful = full-set exact nDCG@10
drop above 0.01.

## Predictions and pass criteria

- **P1, query_hubness against chance**, on the pipelines whose chance skewness
  (`details.null_skewness`) is at least warn. (a) Good pipelines flagged by query_hubness
  with the rule are at most half of those flagged by the thresholds alone. (b) Of the bad
  pipelines the thresholds alone put at fail on query_hubness, the rule keeps at least 90%
  at fail. A part with fewer than 5 cases on the thresholds-alone side is reported as not
  testable.
- **P5, overlap with the reference as a sign of harm**, on the faults other than duplicates.
  (a) The paired z of the drop on the queries side separates harmful from harmless faults at
  AUC at least 0.80, and at least 0.10 above the AUC of the share of changed results.
  (b) `reference_regressions` at tolerance 0.02 flags at least 70% of the faults whose
  nDCG@10 drop exceeds 0.02 and at most 35% of the harmless ones. Not testable with fewer
  than 10 harmless faults or 10 faults above 0.02. Duplicates are reported apart; the
  prediction for them is that the overlap does not see them.
- **P6, labeled gate against the noise of 100 queries**, on every fault of a pipeline with
  at least 100 judged queries, 200 seeded subsets each. (a) With `retrieval_confidence=0.95`
  harmful faults pass at most 3% of the time and at most half as often as with the mean
  alone. (b) Harmless faults fail at most 0.75 times as often as with the mean alone. Not
  testable with fewer than 10 harmful or 10 harmless faults.
- **P7, determinism.** For one good pipeline per fault dataset, the same audit with
  reference, result sample and qrels run twice gives identical JSON.

A prediction that fails is reported as failed. A bug found in the analysis script after this
commit is fixed and named in the report, with the criteria unchanged.
