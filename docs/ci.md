**English** · [한국어](ko/ci.md)

# Use in CI

The JSON output is stable across runs and carries `schema_version`, so commit it as a baseline
and diff it on every index rebuild.

`--baseline` prints each finding's change and exits 1 when any verdict got worse (or
`--fail-on` trips), so the pipeline fails on a regression, not on a `warn` that was already
there.

```bash
vecscope audit embeddings.npy --json > baseline.json          # once, reviewed and committed
vecscope audit embeddings.npy --baseline baseline.json        # on every rebuild
```

```text
  baseline baseline.json: verdict ok -> fail  (n, dim, k, sampling, thresholds or near-dup cutoff differ: values and verdicts are not directly comparable)
    hubness                1.0454 -> 0.8594  (-0.1860)  ok -> ok
    antihubs               0.0383 -> 0.0105  (-0.0278)  ok -> ok
    anisotropy             0.0105 -> 0.0127  (+0.0022)  ok -> ok
    collapse               0.0000 -> 0.8000  (+0.8000)  ok -> fail  WORSE
    ...
  regressions: collapse
```

The example diffs `healthy.npy` against `collapsed.npy`, which have different row counts,
hence the note on the first line.

With `--json` the diff is attached as `baseline_diff`; the library call is
`diff_evidence(new, old)`. A baseline written with a different `schema_version` is rejected with
a message to re-create it.

Audit every row (`--sample 0`) for both the baseline and the check, so the two runs see the
same rows. When both runs pass `--queries`, the baseline's query distances also give
`query_shift` (see [Queries and clusters](usage.md#queries-and-clusters)); it appears under
`only_in_new` because the baseline had no reference.

## Tolerances for drift

A value can drift toward the threshold for months without crossing it and changing the
verdict. Every change row therefore carries:

- `worsened`: the value moved in the bad direction
- `worsening`: by how much, in the metric's units, positive = worse

`--tolerances` gates on that drift, independent of the verdict bands:

```bash
# tolerances.json: {"collapse": 0.01, "ann_recall": 0.02, "hubness": 0.3}
vecscope audit embeddings.npy --baseline baseline.json --tolerances tolerances.json --require-comparable
```

- A finding whose value worsens by more than its tolerance is listed under
  `value_regressions` (text: `VALUE WORSE by 0.0300 > 0.01`) and exits 1 even when its
  verdict is unchanged.
- Findings without a tolerance are judged by verdict alone.
- A value that improves is never a regression.

The tolerance is a per-rebuild budget against the committed baseline, so re-commit the
baseline after a change is reviewed and accepted.

`comparable` is false when n, dim, k, sampling, thresholds or the near-duplicate cutoff
differ from the baseline. `--require-comparable` makes that an error (exit 2) instead of a
note, which is the right policy for a gate: a diff across different settings measures the
settings, not the corpus. The library call is `diff_evidence(new, old, tolerances={...})`.

## Retrieval quality as a gate

Findings describe the geometry of the space, not whether search returns the right documents.
A reindex that loses the row-to-document mapping, a model update that ranks worse on your
queries or a filter that drops relevant documents can break retrieval without moving any
finding. With labeled queries, gate on them directly:

```bash
vecscope audit embeddings.npy --queries queries.npy --qrels qrels.json --sample 0 --json > baseline.json
# after the change
vecscope audit embeddings.npy --queries queries.npy --qrels qrels.json --sample 0 \
  --baseline baseline.json --retrieval-tolerance ndcg_at_k=0.01 --retrieval-tolerance recall_at_k=0.02
```

- `--retrieval-tolerance METRIC=DELTA` (repeatable) is the drop a metric may take against the
  baseline: `ndcg_at_k`, `recall_at_k`, `mrr_at_k`, `hit_rate_at_k`, `duplicate_result_share`
  (a rise is the bad direction) or `overlap_with_exact`. A bare name covers exact search and,
  when both runs passed `--ranked`, the supplied ranking; `exact.` or `ranked.` limits it to one.
- A metric that drops by more is listed in `retrieval_regressions` (as `exact.ndcg_at_k`,
  text: `WORSE by 0.0213 > 0.01`) and exits 1, whatever the findings say.
- A tolerance that cannot be checked also fails the run and is listed in
  `retrieval_unchecked`: the baseline has no retrieval block, the qrels or k differ
  (`retrieval_comparable` false), or the tolerance names `ranked.` and a run has no ranking.
  With `--require-comparable` it is an error (exit 2) instead.
- The flag needs `--baseline` and `--qrels` on the run.

Row ids in the qrels must mean the same documents in both runs. A change that drops or
reorders rows (dedup, a rebuild from another export) needs qrels rewritten for the new rows;
otherwise the stale qrels show up as exactly the collapse this gate exists to catch. The
library call is `diff_evidence(new, old, retrieval_tolerances={"ndcg_at_k": 0.01})`.

### How many labeled queries

The gate reads the mean over the labeled queries, so a small set is noisy. On 594 injected
faults over 99 public pipelines (`examples/validation/label_budget.py`), a 0.01 nDCG
tolerance on a random subset of the labeled queries, judged against the full set's answer,
did this:

| labeled queries | harmful faults caught | drop 0.02 to 0.05 caught | drop above 0.05 caught | harmless changes failed |
|---|---|---|---|---|
| 25 | 83% | 80% | 97% | 33% |
| 50 | 89% | 89% | 100% | 29% |
| 100 | 92% | 95% | 100% | 24% |
| 200 | 95% | 99% | 100% | 17% |
| 400 | 97% | 100% | 100% | 10% |

The 400 row covers the pipelines with at least 400 labeled queries. On preregistered held-out
public data, 100 labeled queries caught 87% of 95 harmful faults (97% of drops above 0.02)
and failed 26% of 25 harmless changes
([Verification](verification.md#the-labeled-gate-against-noise)). About 100 labeled queries
catch nearly every fault that costs more than two nDCG points; drops between 0.01 and 0.02
need several hundred. The false alarms are changes whose full-set drop stayed just under
0.01, which a small set cannot tell from a real 0.01 drop. A failed gate on a small set means
"re-check on more queries", not "broken".

### Judging a drop against the noise of the labeled queries

`--retrieval-confidence LEVEL` lets the gate judge that noise itself. Both runs keep each
judged query's values (`retrieval.per_query`), and the diff pairs them by query row:

```bash
vecscope audit embeddings.npy --queries queries.npy --qrels qrels.json --sample 0 \
  --baseline baseline.json --retrieval-tolerance ndcg_at_k=0.01 --retrieval-confidence 0.95
```

- **fail** (`retrieval_regressions`): the mean drop is beyond the tolerance and above zero
  by more than the one-sided interval at that level.
- **pass**: the interval's upper end is within the tolerance.
- **inconclusive** (`retrieval_inconclusive`): anything else. The run exits 1, the text
  says `INCONCLUSIVE`, and `retrieval_confidence_checks` gives the interval and about how
  many labeled queries would settle it. `change_decision` is `needs_review`, not `reject`.

On the first held-out set's injected faults with 100 labeled queries, this rule passed 1% of
harmful changes (mean alone: 8%) and failed 14% of harmless ones (mean alone: 24%); 17% of
harmful and 62% of harmless changes were inconclusive. Most harmless faults there cost just
under 0.01, which 100 labels cannot settle. On preregistered held-out data 0.4% of harmful
changes passed (3.9% with the mean alone), 10.1% of harmless ones failed (23.7%), and 12% of
harmful and 68% of harmless changes were inconclusive
([Verification](verification.md#the-labeled-gate-against-noise)). A baseline without
per-query values falls back to the mean.

## Results that must not change

Without labels, no statistic of the space tells a harmful change from a harmless one: on the
same injected faults, the share of top-k results a change moved separated them no better
than chance (AUC 0.58). It does show that results moved, even when no finding does: a lost id
mapping changes no finding but moves the results of every query that hit a moved row. For a
pipeline meant to return the same results (a re-index, a refactor, a library or hardware
upgrade), gate on the results themselves:

```bash
vecscope audit embeddings.npy --queries queries.npy --result-sample 500 --json > baseline.json
# after the change
vecscope audit embeddings.npy --queries queries.npy --result-sample 500 \
  --baseline baseline.json --fail-on none --result-tolerance 0
```

- `--result-sample M` stores the exact top-k of M queries and M corpus rows in the JSON. Rows
  are chosen by a hash of the row number, so appended rows take only part of the sample and
  a grown corpus shares most of it with the baseline.
- The diff carries `result_agreement` per side (`queries`, `rows`): `changed_share` is the
  share of top-k slots that hold another row, `unchanged_items` the share of items whose
  top-k is identical.
- `--result-tolerance SHARE` fails the run when a side changed more (`result_regressions`)
  or either run has no sample (`result_unchecked`). It needs `--baseline` and
  `--result-sample`.
- A change meant to move results (a new model, a transform) is not judged by this number.
  Without labels its `change_decision` is `needs_review` and cites the changed share; the
  labeled gate above decides whether it is better.

### Without labels: did the change lose the top hits a reference model agrees on?

A reference model can stand in for labels. Embed the same rows and queries with a second,
independently good public model, and pass it when you write the baseline:

```bash
vecscope audit embeddings.npy --queries queries.npy --result-sample 200 \
  --reference ref.npy --reference-queries ref_q.npy --json > baseline.json
# after the change: only the baseline needs the reference
vecscope audit embeddings.npy --queries queries.npy --result-sample 200 \
  --baseline baseline.json --fail-on none --reference-tolerance 0.03
```

- The baseline stores, per sampled query and row, the reference's first result
  (`result_sample.reference_first`). Where that result is also in the pipeline's top-k, the
  two models agree on it: an agreed top hit.
- The diff gives, per side, `result_agreement.<side>.reference_top_hit`: how many items had
  an agreed top hit (`agreed`), how many of those the new run no longer returns (`lost`), how
  many items without one the new run now has (`gained`), and the **net drop**: the mean over
  every compared item of "in the baseline's top-k" minus "in the new top-k", with its z
  (`net_drop`, `net_z`), and the lost share with its z. With the reference on both runs,
  `result_agreement.<side>.reference` also gives the overlap with the reference before and
  after.
- `--reference-tolerance SHARE` lists a side in `reference_regressions` when the net drop is
  beyond the tolerance and beyond 2 standard errors. The queries side decides when there are
  queries. A baseline without the reference is `reference_unchecked`. Both exit 1.
- Below the tolerance, a net drop above 0.015 and 2 standard errors is listed in
  `reference_review` (text: `REVIEW: a small loss is possible, check with labels`). It never
  fails the gate; `change_decision` says a small loss is possible.
- Without a tolerance, `change_decision` still says whether the agreed top hits were lost (a
  net drop above 0.03), lost a little (above 0.015) or kept.

A harmless change such as truncated or quantized vectors reshuffles the lower ranks, where the
two models never agreed; a harmful one loses the results both models put first. The net drop
counts gains as well as losses because a new model loses some agreed top hits while finding
others the old model missed, so losses alone would flag better models too. On a
preregistered held-out set ([Verification](verification.md#agreed-top-hits)):

| | flagged, drop above 0.02 nDCG@10 | flagged, harmless |
|---|---|---|
| injected faults | 137 of 154 (89%) | 0 of 81 |
| model swaps | 116 of 133 (87%) | 1 of 85 |

`reference_review` listed 5 of the 17 large fault losses the tolerance let through, and 3 of
the 81 harmless faults. Read a net drop as "likely harmful, check with labels", not as a
measured loss:

- It misses small losses spread over the lower ranks: it flagged 1 of 20 harmful truncations
  and 1 of 8 harmful quantizations.
- It caught 47 of 52 lost-row faults (5% zero vectors); the findings catch those too.
- It cannot see rows that are wrong for both models. Duplicated rows show up in the
  reference too; `collapse` finds those.
- The reference must be made from the source texts, independently of the pipeline under
  test. A reference built from the same faulty export shares the fault.
- It is not a verdict on a new embedding model. It rarely flags a swap that improves
  retrieval, but misses small losses between models of similar quality: of 13 multilingual
  swaps losing 0.02 to 0.09 nDCG@10 it flagged 1. Use it to drop candidates that lose many
  agreed hits, and decide with `--retrieval-confidence` (0.07% of the harmful swaps passed
  and 0.65% of the harmless ones failed with 100 labeled queries). A candidate from the
  reference's own family (a fine-tune of it) agrees with it by construction and cannot be
  judged this way. A worked example: [Changing the embedding model](case_model_change.md).

## Which rule failed the run

Two kinds of rule can fail a run, and the output keeps them apart:

- The absolute rule is the verdict against the thresholds (`--fail-on fail` by default,
  `--fail-on warn` to be strict).
- The baseline rules are `regressions` (a verdict got worse), `value_regressions` (a value
  worsened beyond its tolerance), `retrieval_regressions` (a labeled retrieval metric dropped
  beyond its tolerance), `retrieval_unchecked` (a retrieval tolerance could not be checked),
  `result_regressions` (sampled results changed beyond `--result-tolerance`),
  `result_unchecked` (a run had no `--result-sample`), `retrieval_inconclusive` (the
  labeled queries could not settle a tolerance under `--retrieval-confidence`),
  `reference_regressions` (a net drop of agreed top hits beyond `--reference-tolerance`) and
  `reference_unchecked` (the baseline had no reference results).

The defaults are calibrated on public models. If a healthy corpus in your domain sits at
`warn` or `fail` on some finding, do not gate on the absolute rule:
`--fail-on none --baseline baseline.json` judges only the change against the reviewed
baseline.

Every run ends with a `gate` block (JSON) or line (text) that names the deciding rule:

```text
  gate: FAIL; verdict fail reaches --fail-on fail; verdict worse than baseline: collapse
  gate: pass (baseline)
```

Treat the gate as a trend tracker first: how much change is too much depends on how the
findings relate to retrieval quality on your own queries. The
[retrieval quality study](quality.md#do-the-findings-predict-retrieval-quality) shows that
relation for public datasets, and
[calibrating thresholds](quality.md#calibrating-thresholds-to-your-domain) says how to
measure it on yours before blocking merges on it.

## GitHub Action

The repository is also a GitHub Action. It installs vecscope from the tag the workflow names,
runs `vecscope audit --json`, writes the findings table to the job summary and fails the job
with the CLI's exit code.

```yaml
- uses: 076923/vecscope@v0          # or pin a release: @v0.13.0
  with:
    path: embeddings.npy
    baseline: baseline.json         # committed --json output, optional
    tolerances: tolerances.json     # optional
    queries: queries.npy            # optional, with qrels: labeled retrieval quality
    qrels: qrels.json
    retrieval-tolerance: ndcg_at_k=0.01 recall_at_k=0.02
    sample: "0"                     # audit every row when diffing against a baseline
    require-comparable: "true"
```

| Input | Default | CLI option |
|---|---|---|
| `path` | required | the audited file, index or store URI |
| `baseline`, `tolerances`, `thresholds`, `queries`, `qrels` | empty | `--baseline`, `--tolerances`, `--thresholds`, `--queries`, `--qrels` |
| `retrieval-tolerance` | empty | one `--retrieval-tolerance` per `METRIC=DELTA`, separated by spaces or commas |
| `fail-on` | `fail` | `--fail-on` |
| `k`, `sample` | `10`, `100000` | `-k`, `--sample` |
| `require-comparable` | `false` | `--require-comparable` |
| `args` | empty | further options, separated by spaces |
| `output` | `vecscope-evidence.json` | where the evidence JSON is written |
| `extras` | empty | extras to install, e.g. `index` for faiss files or `qdrant` |
| `python-version` | `3.13` | empty uses the Python already on the runner |

Outputs: `verdict` (`ok`, `warn`, `fail`, or `error` when no evidence was written),
`exit-code`, `regressions` (comma-separated finding names), `retrieval-regressions`
(comma-separated metrics such as `exact.ndcg_at_k`, then `unchecked:<metric>` for a tolerance
that could not be checked) and `evidence` (the JSON path). The job summary adds a retrieval
table when both runs had labels. Upload `evidence` as an artifact to keep it, or commit it as
the next baseline after review.

`@v0` moves to every 0.x release, and a 0.x minor release may change findings or thresholds,
so pin the full version (`@v0.13.0`) in a gate that must stay stable.
