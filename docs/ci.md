**English** · [한국어](ko/ci.md)

# Use in CI

The JSON output is stable across runs and carries `schema_version`, so it can be committed
as a baseline and diffed on every index rebuild.

`--baseline` prints each finding's change and exits 1 when any verdict got worse (or when
`--fail-on` trips). The pipeline then fails on a regression, not on a `warn` that was
already there.

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
hence the note on the first line; a rebuild of the same corpus with the same settings prints
no note.

With `--json` the diff is attached as `baseline_diff`; the library call is
`diff_evidence(new, old)`. A baseline written by an older `schema_version` is rejected with
a message to re-create it.

Audit every row (`--sample 0`) for the baseline and the check, so the two runs see the same
rows. When both runs pass `--queries`, the baseline's query distances also give
`query_shift` (see [Queries and clusters](usage.md#queries-and-clusters)); it shows up under
`only_in_new` in the diff because the baseline itself had no reference.

## Tolerances for drift

A verdict is a coarse signal: a value can drift toward the threshold for months without
crossing it. Every change row therefore carries:

- `worsened`: the value moved in the bad direction
- `worsening`: by how much, in the metric's units, positive = worse

`--tolerances` turns that drift into a gate independent of the verdict bands:

```bash
# tolerances.json: {"collapse": 0.01, "ann_recall": 0.02, "hubness": 0.3}
vecscope audit embeddings.npy --baseline baseline.json --tolerances tolerances.json --require-comparable
```

- A finding whose value worsens by more than its tolerance is listed under
  `value_regressions` (text: `VALUE WORSE by 0.0300 > 0.01`) and exits 1 even when its
  verdict is unchanged.
- Findings without a tolerance are judged by verdict alone.
- A value that improves is never a regression.

The tolerance is a budget per rebuild against the committed baseline, so re-commit the
baseline when a change was reviewed and accepted.

The run also reports `comparable`: false when n, dim, k, sampling, thresholds or the
near-duplicate cutoff differ from the baseline.

`--require-comparable` makes that an error (exit 2) instead of a note. That is the right
policy for a gate, since a diff across different settings measures the settings, not the
corpus. The library call is `diff_evidence(new, old, tolerances={...})`.

## Retrieval quality as a gate

Findings describe the geometry of the space, not whether search returns the right documents.
A change can leave every finding where it was and still break retrieval: a reindex that loses
the row-to-document mapping, a model update that ranks worse on your queries, a filter that
drops relevant documents. When labeled queries exist, gate on them directly:

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
- A tolerance that cannot be checked fails the run too and is listed in
  `retrieval_unchecked`: the baseline has no retrieval block, the qrels or k differ
  (`retrieval_comparable` false), or the tolerance names `ranked.` and a run has no ranking.
  A requested check that did not run is not a pass. With `--require-comparable` it is an
  error (exit 2) instead.
- The flag needs `--baseline` and `--qrels` on the run.

Row ids in the qrels must mean the same documents in both runs. A change that drops or
reorders rows (dedup, a rebuild from another export) needs qrels rewritten for the new rows;
a qrels file that silently points at other documents shows up as exactly the collapse this
gate exists to catch. The library call is
`diff_evidence(new, old, retrieval_tolerances={"ndcg_at_k": 0.01})`.

### How many labeled queries

The gate reads the mean over the labeled queries, so a small set is noisy. On 594 injected
faults over 99 public pipelines (`examples/validation/label_budget.py`), a 0.01 nDCG
tolerance on a random subset of the labeled queries did this against the answer of the full
set:

| labeled queries | harmful faults caught | drop 0.02 to 0.05 caught | drop above 0.05 caught | harmless changes failed |
|---|---|---|---|---|
| 25 | 83% | 80% | 97% | 33% |
| 50 | 89% | 89% | 100% | 29% |
| 100 | 92% | 95% | 100% | 24% |
| 200 | 95% | 99% | 100% | 17% |
| 400 | 97% | 100% | 100% | 10% |

The 400 row covers the pipelines with at least 400 labeled queries. On a second set of public
data preregistered before it was embedded, 100 labeled queries caught 87% of 95 harmful faults
(97% of drops above 0.02) and failed 26% of 25 harmless changes
([Verification](verification.md#a-second-held-out-set-preregistered)). About 100 labeled
queries catch nearly every fault that costs more than two nDCG points.
Drops between 0.01 and 0.02 need several hundred. The false alarms are changes whose
full-set drop stayed just under 0.01; a small set cannot tell them from a real 0.01 drop, so
a failed gate on a small set means "re-check on more queries", not "broken".

### Judging a drop against the noise of the labeled queries

`--retrieval-confidence LEVEL` turns that advice into the gate's own answer. Both runs keep
each judged query's values (`retrieval.per_query`), and the diff pairs them by query row:

```bash
vecscope audit embeddings.npy --queries queries.npy --qrels qrels.json --sample 0 \
  --baseline baseline.json --retrieval-tolerance ndcg_at_k=0.01 --retrieval-confidence 0.95
```

- **fail** (`retrieval_regressions`): the mean drop is beyond the tolerance and above zero
  by more than the one-sided interval at that level, so the drop is real.
- **pass**: the interval's upper end is within the tolerance.
- **inconclusive** (`retrieval_inconclusive`): anything else. The run exits 1, the text
  says `INCONCLUSIVE`, and `retrieval_confidence_checks` gives the interval and about how
  many labeled queries would settle it. `change_decision` is `needs_review`, not `reject`.

On the first held-out set's injected faults with 100 labeled queries, this rule let 1% of
harmful changes pass, where the mean alone let 8% through. It failed 14% of harmless
changes, where the mean alone failed 24%. The rest were inconclusive: 17% of harmful and
62% of harmless changes. Most harmless faults there cost just under 0.01, so 100 labels
cannot settle them. The rule was designed on that set and confirmed on the third held-out
set, frozen before it was embedded: 0.4% of harmful changes passed (3.9% with the mean
alone), 10.1% of harmless ones failed (23.7%), and 12% of harmful and 68% of harmless
changes were inconclusive ([Verification](verification.md#a-third-held-out-set-preregistered)).
A baseline written
before this version has no per-query values, and its tolerances fall back to the mean.

## Results that must not change

Without labels, no statistic of the space tells a harmful change from a harmless one. On the
same injected faults, the share of top-k results a change moved separated harmful from
harmless changes no better than chance (AUC 0.58). It does show that results moved, also
when no finding does: a lost id mapping changes no finding and moves the results of every
query that hit a moved row. So for a pipeline that is meant to return the same results (a
re-index, a refactor, a library or hardware upgrade), gate on the results themselves:

```bash
vecscope audit embeddings.npy --queries queries.npy --result-sample 500 --json > baseline.json
# after the change
vecscope audit embeddings.npy --queries queries.npy --result-sample 500 \
  --baseline baseline.json --fail-on none --result-tolerance 0
```

- `--result-sample M` stores the exact top-k of M queries and M corpus rows in the JSON. The
  rows are chosen by a hash of the row number, so appended rows only take part of the sample
  and a grown corpus still shares most of it with the baseline.
- The diff carries `result_agreement` per side (`queries`, `rows`): `changed_share` is the
  share of top-k slots that hold another row, `unchanged_items` the share of items whose
  top-k is identical.
- `--result-tolerance SHARE` fails the run when a side changed more (`result_regressions`),
  and when either run has no sample (`result_unchecked`). It needs `--baseline` and
  `--result-sample`.
- A change that is meant to move results (a new model, a transform) is not judged by this
  number. Without labels its `change_decision` is `needs_review` and cites the changed
  share; the labeled gate above decides whether it is better.

### Without labels: did the change lose the top hits a reference model agrees on?

A reference model can stand in for labels. Embed the same rows and queries with a second,
independently good public model, and pass it when you write the baseline:

```bash
vecscope audit embeddings.npy --queries queries.npy --result-sample 200 \
  --reference ref.npy --reference-queries ref_q.npy --json > baseline.json
# after the change: only the baseline needs the reference
vecscope audit embeddings.npy --queries queries.npy --result-sample 200 \
  --baseline baseline.json --fail-on none --reference-tolerance 0.05
```

- The baseline stores, per sampled query and row, the reference's first result
  (`result_sample.reference_first`). Where that result is also in the pipeline's top-k,
  both models agree on it: an agreed top hit.
- The diff gives, per side, `result_agreement.<side>.reference_top_hit`: how many items had
  an agreed top hit, how many of those the new run no longer returns, the lost share and
  its z. With the reference on both runs, `result_agreement.<side>.reference` also gives the
  overlap with the reference before and after, for information.
- `--reference-tolerance SHARE` lists a side in `reference_regressions` when the lost share
  is beyond the tolerance and beyond 2 standard errors. The queries side decides when there
  are queries. A baseline without the reference is `reference_unchecked`. Both exit 1.
- Below the tolerance, a lost share above 0.03 and 2 standard errors is listed in
  `reference_review` (text: `REVIEW: a small loss is possible, check with labels`). It never
  fails the gate; `change_decision` says a small loss is possible. On the faults of four
  held-out sets it listed 80 of the 117 changes costing more than 0.02 nDCG@10 that 0.05 let
  through, and 29 of 244 harmless changes. On a fifth set, frozen before it was embedded and
  run with another reference model
  ([Verification](verification.md#a-fifth-held-out-set-preregistered)), it listed 12 of the
  23 such changes and 3 of 47 harmless ones; on a sixth, 13 of 28 and 4 of 61, below the half
  it was expected to list.
- Without a tolerance, `change_decision` still says whether the agreed top hits were lost (at
  0.05), lost a little (above 0.03) or kept.

Why the top hits and not the whole list: a harmless change such as truncated or quantized
vectors reshuffles the lower ranks, where the two models never agreed, and a harmful one
loses the results both models put first. An earlier version gated on the drop of the whole
overlap; on the preregistered third held-out set it flagged 44% of the harmless faults
([Verification](verification.md#a-third-held-out-set-preregistered)). On the injected faults
of the three held-out sets (duplicates left out), the lost share of agreed top hits separated
harmful from harmless changes at AUC 0.91, where the drop of the whole overlap did at 0.86
and the share of changed results at 0.51. At 0.05 it flagged 72% to 83% of the faults that
cost more than 0.02 nDCG@10 per set and 5% to 8% of the harmless ones. Those sets were
the design data. On a fourth held-out set, frozen before it was embedded
([Verification](verification.md#a-fourth-held-out-set-preregistered)), it separated them at
AUC 0.97 and flagged 72% of the faults costing more than 0.02 and 1 of 48 harmless ones
(2%), where the whole-overlap drop flagged 21 of 48. On a fifth, with gte-multilingual-base
instead of bge-m3 as the reference
([Verification](verification.md#a-fifth-held-out-set-preregistered)), it separated them at
AUC 0.93 and flagged 78% of the faults costing more than 0.02 and 1 of 47 harmless ones; on
a sixth, with bge-m3 again
([Verification](verification.md#a-sixth-held-out-set-preregistered)), AUC 0.93, 74% and 1
of 61. Read a loss as "likely harmful,
check with labels", not as a measured loss:

- It misses small losses spread over the lower ranks: most truncation and quantization
  faults that cost 0.01 to 0.02 nDCG@10. `reference_review` lists some of them for review.
- It catches about half of the lost-row faults (5% zero vectors); the findings catch those.
- It cannot see rows that are wrong for both models. Duplicated rows show up in the
  reference too; `collapse` finds those.
- The reference must be made from the source texts, independently of the pipeline under
  test. A reference built from the same faulty export shares the fault.
- It counts the agreed top hits a change loses, not the ones it gains, so it is not a verdict
  on a new embedding model. A new model loses some agreed hits even when it finds more
  elsewhere: on the preregistered sixth set it flagged 72 of 80 model swaps losing more than
  0.02 nDCG@10, but also 5 of 27 swaps that improved it
  ([Verification](verification.md#a-sixth-held-out-set-preregistered)). Use it to drop
  candidates that lose many agreed hits, and decide with `--retrieval-confidence` (0.04% of
  the harmful swaps passed, 0.07% of the harmless ones failed with 100 labeled queries). A
  candidate from the reference's own family (a fine-tune of it) agrees with it by
  construction and cannot be judged this way. A worked example:
  [Changing the embedding model](case_model_change.md).

## Which rule failed the run

Two different rules can fail a run, and the output keeps them apart:

- The absolute rule is the verdict against the thresholds (`--fail-on fail` by default,
  `--fail-on warn` to be strict).
- The baseline rules are `regressions` (a verdict got worse), `value_regressions` (a value
  worsened beyond its tolerance), `retrieval_regressions` (a labeled retrieval metric dropped
  beyond its tolerance), `retrieval_unchecked` (a retrieval tolerance could not be checked),
  `result_regressions` (sampled results changed beyond `--result-tolerance`),
  `result_unchecked` (a run had no `--result-sample`), `retrieval_inconclusive` (the
  labeled queries could not settle a tolerance under `--retrieval-confidence`),
  `reference_regressions` (agreed top hits lost beyond `--reference-tolerance`) and
  `reference_unchecked` (the baseline had no reference results).

The defaults are calibrated on public models, so a domain whose healthy corpus sits at `warn`
or `fail` on some finding should not be blocked by the absolute rule at all.
`--fail-on none --baseline baseline.json` judges only the change against the reviewed
baseline.

Every run ends with a `gate` block (JSON) or line (text) that names the rule that decided:

```text
  gate: FAIL; verdict fail reaches --fail-on fail; verdict worse than baseline: collapse
  gate: pass (baseline)
```

Treat the gate as a trend tracker first. Which tolerance is "too much" depends on how the
findings relate to retrieval quality on your own queries.

The [retrieval quality study](quality.md#do-the-findings-predict-retrieval-quality) shows that relation
for public datasets, and [calibrating thresholds](quality.md#calibrating-thresholds-to-your-domain) says
how to measure it on yours before blocking merges on it.

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
table when both runs had labels.
Upload `evidence` as an artifact to keep it, or commit it as the next baseline after review.

`@v0` moves to every 0.x release, and a 0.x minor release may change findings or thresholds,
so pin the full version (`@v0.13.0`) in a gate that must stay stable.
