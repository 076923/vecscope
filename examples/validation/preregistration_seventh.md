# Seventh held-out set: preregistration

Written and committed on 2026-11-27, before any dataset below was downloaded or embedded.
The commit that adds this file freezes the code, the thresholds, the rules and the criteria.
The results are reported whatever they are; nothing is tuned on this set.

## Why a seventh set

On the sixth set ([preregistration_sixth.md](preregistration_sixth.md)) the agreed top hits
failed on model changes (P18a): at a lost share of 0.05 they flagged 5 of 27 changes that
improved nDCG@10. The diff counted the agreed top hits a new run lost, never the reference's
first results it gained, and a new model loses some agreed hits while finding others.

**One rule changed after the sixth set.** The diff now also counts, per sampled query, the
reference's first result gained (in the new top-k, not in the baseline's), and the rules
judge the **net drop**: the mean over every compared query of (in the baseline's top-k) minus
(in the new top-k). `reference_regressions` lists a side whose net drop is above the
tolerance (default 0.03) and 2 standard errors; `reference_review` one above 0.015 and 2
standard errors, below the tolerance.

It was designed on data that is no longer blind (`net_design.py`, `net_design.json`): 595
injected faults of the fourth to sixth sets (duplicates left out) and 1,488 model changes on
the 63 datasets of sets 1 to 6, from two baselines (multilingual-e5-base and
paraphrase-multilingual-MiniLM-L12-v2), 200 sampled queries each.

| design data | rule | flagged, drop above 0.02 | flagged, harmless |
|---|---|---|---|
| faults | lost share above 0.05 (shipped until now) | 258 of 351 (74%) | 3 of 163 (1.8%) |
| faults | net drop above 0.03 | 286 of 351 (81%) | 1 of 163 (0.6%) |
| changes from e5-base | lost share above 0.05 | 434 of 471 (92%) | 94 of 265 (35%) |
| changes from e5-base | net drop above 0.03 | 414 of 471 (88%) | 13 of 265 (4.9%) |
| changes from MiniLM-L12 | lost share above 0.05 | 256 of 257 (100%) | 225 of 495 (45%) |
| changes from MiniLM-L12 | net drop above 0.03 | 245 of 257 (95%) | 3 of 495 (0.6%) |

Thresholds 0.02 to 0.05 were compared on that data; 0.03 was the lowest at which the harmless
faults stayed at 1 of 163. By fault, the net rule flagged more zero-row faults (82 of 119
against 48) and slightly fewer truncations (6 of 58 against 8) and quantizations (4 of 36
against 5).

KURE-v1, a fine-tune of bge-m3, lost almost no agreed top hits against bge-m3 on every sixth-set
dataset, whatever its nDCG@10 did. Candidates from the reference's own family are left out of
the criteria and reported apart.

Nothing else changed: the findings, thresholds, the reference rule and the two lowering
rules are those of the sixth set.

## Data (all public, none seen before)

| name | source | language | domain | license | documents | queries |
|---|---|---|---|---|---|---|
| xpqa-ta | mteb/XPQARetrieval tam-tam | ta | product answers | CDLA-Sharing-1.0 | 1,275 | all (782) |
| wikipedia-da | mteb/WikipediaRetrievalMultilingual da | da | wikipedia | CC-BY-SA-3.0 | 13,500 | all (1,500) |
| wikipedia-no | mteb/WikipediaRetrievalMultilingual no | no | wikipedia | CC-BY-SA-3.0 | 13,500 | all (1,500) |
| wikipedia-ro | mteb/WikipediaRetrievalMultilingual ro | ro | wikipedia | CC-BY-SA-3.0 | 13,500 | all (1,500) |
| wikipedia-bg | mteb/WikipediaRetrievalMultilingual bg | bg | wikipedia | CC-BY-SA-3.0 | 13,500 | all (1,500) |
| wikipedia-de | mteb/WikipediaRetrievalMultilingual de | de | wikipedia | CC-BY-SA-3.0 | 13,500 | all (1,500) |
| webfaq-hrv | mteb/WebFAQRetrieval hrv | hr | web FAQ | CC-BY-4.0 | 5,545 | all (555) |
| codesearchnet-javascript | mteb/CodeSearchNetRetrieval javascript | code | code | MIT | 1,000 | all (1,000) |
| webfaq-slk | mteb/WebFAQRetrieval slk | sk | web FAQ | CC-BY-4.0 | 31,522 | 1,000 seeded |
| webfaq-bul | mteb/WebFAQRetrieval bul | bg | web FAQ | CC-BY-4.0 | 34,734 | 1,000 seeded |

The sizes are the Hub's metadata, the only thing read from the Hub before this commit. A run
uses only the judged queries. Models: the same 14 public models (registry.py). Labels: good =
nDCG@10 at least 0.9 x the best model on the dataset, bad = at most 0.5 x the best.

## Procedure

```bash
python examples/validation/run.py --datasets xpqa-ta wikipedia-da wikipedia-no wikipedia-ro \
    wikipedia-bg wikipedia-de webfaq-hrv codesearchnet-javascript webfaq-slk webfaq-bul
python examples/validation/seventh.py
```

Reference model: **bge-m3**, gte-multilingual-base for bge-m3 itself. The six faults go into
every good pipeline, as in the sixth set. Model changes: on each dataset, each of the two
baselines (multilingual-e5-base, paraphrase-multilingual-MiniLM-L12-v2) is replaced by each
other model except the reference; the candidate is audited with the same reference, queries,
qrels and result sample (200) and diffed against the baseline at the shipped defaults
(`reference_tolerance` 0.03). Harmful = full-set exact nDCG@10 drop above 0.01. The lost-share
rule of sets 4 to 6 (above 0.05 and 2 standard errors) is computed from the same diff for
comparison.

## Predictions and pass criteria

Each result is reported with its 95% interval. The pass criteria apply to the point values.
A criterion is not testable with fewer than 10 cases in a group it needs.

- **P20, the net rule on the faults** (duplicates left out):
  - (a) It flags at least 70% of the faults whose drop exceeds 0.02, and at most 10% of the
    harmless ones.
  - (b) It flags at least as many of the faults above 0.02 as the lost-share rule.
- **P21, the net rule on model changes** (candidates of the reference's family left out):
  - (a) It flags at least 65% of the changes losing more than 0.02 nDCG@10, and at most 10%
    of those losing 0.01 or less (improvements included).
  - (a-multi) On the multilingual candidates alone, at least 60% and at most 10%.
  - (b) It flags at most half as many harmless changes as the lost-share rule.
- **P22, the two lowering rules** (unchanged; the criteria of P17):
  - (a) Of the good pipelines at `fail` with the reference before the two rules, at least
    half end without a `fail`. Not testable with fewer than 4.
  - (b) Of the bad pipelines at `fail` with the reference before the two rules, at least 95%
    stay at `fail`.
  - (c) Of the harmful faults the findings catch before the two rules, at least 95% are
    still caught with them.
- **P23, the labeled gate on the model changes** (the criteria of P18b and c): with 100
  seeded labeled queries and `retrieval_confidence=0.95` (200 subsets each), (a) harmful
  changes pass at most 3% of the time and at most half as often as with the mean alone; (b)
  harmless changes fail at most 0.75 times as often as with the mean alone.
- **P24, determinism.** For one good pipeline per dataset, the same audit with reference,
  result sample and qrels run twice gives identical JSON.

The only seventh-set code run before this commit is a dry run of `seventh.py` on one
development dataset of the fifth set (webfaq-srp), to check the code: the net rule flagged 14
of 16 faults above 0.02 (lost share: 9) and 0 of 8 harmless; on 22 model changes, 14 of 15
above 0.02 (lost share: 15) and 0 of 6 harmless (lost share: 2). The AUC of the net z there
was 0.80 against 0.94 for the lost z.

Reported without predictions:
- the review tier at 0.015 on faults and model changes
- the AUC of the net z and of the lost z between harmful and harmless faults
- by fault, by corpus size and by baseline
- the candidates of the reference's family
- the labeled gate on the faults and the `query_hubness` chance rule

A prediction that fails is reported as failed. A bug found in the analysis script after this
commit is fixed and named in the report, with the criteria unchanged.
