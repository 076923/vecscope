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

## Which rule failed the run

Two different rules can fail a run, and the output keeps them apart:

- The absolute rule is the verdict against the thresholds (`--fail-on fail` by default,
  `--fail-on warn` to be strict).
- The baseline rules are `regressions` (a verdict got worse) and `value_regressions` (a value
  worsened beyond its tolerance).

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
    sample: "0"                     # audit every row when diffing against a baseline
    require-comparable: "true"
```

| Input | Default | CLI option |
|---|---|---|
| `path` | required | the audited file, index or store URI |
| `baseline`, `tolerances`, `thresholds`, `queries` | empty | `--baseline`, `--tolerances`, `--thresholds`, `--queries` |
| `fail-on` | `fail` | `--fail-on` |
| `k`, `sample` | `10`, `100000` | `-k`, `--sample` |
| `require-comparable` | `false` | `--require-comparable` |
| `args` | empty | further options, separated by spaces |
| `output` | `vecscope-evidence.json` | where the evidence JSON is written |
| `extras` | empty | extras to install, e.g. `index` for faiss files or `qdrant` |
| `python-version` | `3.13` | empty uses the Python already on the runner |

Outputs: `verdict` (`ok`, `warn`, `fail`, or `error` when no evidence was written),
`exit-code`, `regressions` (comma-separated finding names) and `evidence` (the JSON path).
Upload `evidence` as an artifact to keep it, or commit it as the next baseline after review.

`@v0` moves to every 0.x release, and a 0.x minor release may change findings or thresholds,
so pin the full version (`@v0.13.0`) in a gate that must stay stable.
