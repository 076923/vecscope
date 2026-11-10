**English** · [한국어](ko/performance.md)

# Performance and how vecscope works

## Performance

Exact k-NN is quadratic in the number of rows: doubling the rows roughly quadruples the time,
and a full audit (`--sample 0`) of 100k rows takes minutes on a laptop.

The k-NN pass works in row blocks sized to keep its distance matrix near 1 GiB whatever the
row count. The other passes need a few float32 copies of the matrix, so the peak grows with
rows x dims at roughly four times the matrix size.

Measured on an 8-vCPU AMD EPYC 7R13 with 8 BLAS threads (NumPy only, single process,
clustered synthetic corpus, k=10). Memory is the peak the audit itself allocates; the input
matrix comes on top. A three-run median in separate processes agreed with these single runs
within 2%, and the OS peak memory within 0.04 GiB.

| rows | dim | seconds | peak memory (input excluded) |
|---|---|---|---|
| 10,000 | 384 | 1.3 | 0.40 GiB |
| 10,000 | 768 | 2.0 | 0.41 GiB |
| 10,000 | 1,536 | 3.6 | 0.44 GiB |
| 50,000 | 384 | 17 | 1.63 GiB |
| 50,000 | 768 | 23 | 1.70 GiB |
| 50,000 | 1,536 | 36 | 1.84 GiB |
| 100,000 | 384 | 65 | 1.71 GiB |
| 100,000 | 768 | 86 | 1.85 GiB |
| 100,000 | 1,536 | 130 | 3.45 GiB |
| 200,000 | 768 | 338 | 3.46 GiB |
| 200,000, default 100k sample | 768 | 85 | 2.14 GiB |

- Time grows close to n² (3.5 to 3.9x per doubling from 50k up).
- Memory is flat from 50k rows at 384 and 768 dims because the k-NN block shrinks with n,
  and steps up at 1,536 dims and at 200k rows.
- The default 100k-row sample keeps any corpus near the 100k rows.
- `--queries` adds one more pass over the corpus per query batch, and `metric_mismatch` adds
  a second k-NN pass only when vector norms vary.

To re-measure on your machine at the sizes you care about:

```bash
python examples/benchmark_scale.py --rows 50000 100000 --dim 768   # the audit table above
python examples/benchmark_sweep.py                                 # the index sweep
```

Both take `--corpus` (clustered, isotropic, anisotropic, duplicates, lowrank, zipf) and
record the machine in their JSON.

### Sampled audits

Blocking bounds memory, not work: the exact pass still costs about n² × dim multiply-adds,
so a million rows at 768 dims is roughly 100 times the 100k-row figure.

For corpora that size, audit the default sample (or several with `--sample-repeats`) and
reserve `--sample 0` for the findings that need every row. A sampled audit is a statement
about the sample. Which findings need every row:

| finding | what a sample gives | needs every row |
|---|---|---|
| anisotropy, effective_rank, rogue_dimensions, norms, metric_mismatch, distance_concentration | a distribution estimate: within 2.3% of the full value on 200k x 384 with 100k, 50k and 20k samples, and the verdict holds | no |
| hubness, antihubs | a *lower bound*: k-occurrence concentrates with density, so the same corpus scores less hubness on a smaller sample (skewness 2.6 `warn` on all 200k rows, 1.15 on a quarter, 0.16 on a tenth). Verdicts other than `fail` are marked inconclusive | to measure the corpus's own hubness |
| collapse | the duplicate *share* seen inside the sample: a pair counts only when both rows were drawn, so the sampled share is 0.96 to 1.03 x (sample fraction x full share); the verdict uses the estimated full share, at least the exact copies counted over every row | to count or list every near-duplicate |
| outliers | an estimate that over-counts on small samples (3.9x at a tenth); verdicts other than `ok` are marked inconclusive | to find every outlier |
| connectivity | the islands of the sample's own k-NN graph, which miss most islands of the corpus (37% of rows in islands scored 0.08% on a quarter sample); verdicts other than `fail` are marked inconclusive | yes, to judge connectivity at all |
| ann_recall | not available on a sample: ids must line up with the ANN file | yes (`--sample 0`) |
| query_gap, query_shift, query_hubness, ood_queries | reference distances from the sample; every query is scored. query_gap is a ratio of two distances that both grow on a smaller sample; compare query_shift only against a baseline with the same sample size | no |

A sampled audit therefore:

- sets `evidence_scope` to `sampled` (`full` when every row was audited) and lists in
  `needs_full_audit` the findings whose verdict cannot stand for the corpus: the inconclusive
  ones and, with `--sample-repeats`, the unstable ones. Anything consuming the JSON can gate
  on these two keys without reading the per-finding caveats
- lists `inconclusive_findings` (the verdicts the table marks inconclusive)
- flags them `INCONCLUSIVE (sampled)` in the text output and names them in the gate line
- takes the collapse verdict from `details.estimated_full_share` (see below)
- repeats each caveat in the finding's explanation next to the number

`estimated_full_share` is the sampled share divided by the sample fraction, or the share of
rows with an exact copy counted over every row (`details.full_corpus_exact_copy_share`) when
that is larger. Counting exact copies over the whole corpus is cheap (one digest per row), so
rare copies the sample missed still count. Near-duplicates that are not exact copies are only
seen in the sample.

The 200k-row figures come from the
[verification](https://github.com/076923/vecscope/blob/main/docs/verification.md);
`tests/test_regression_injection.py` checks the same properties at 6k rows on every run.

Every number is exact for the rows it saw; vecscope never substitutes an approximate
neighbor search for the exact one. The only approximation is the choice of rows.

## How vecscope works

`vecscope` computes exact k-NN once over the matrix and derives every diagnostic from it:

- the k-occurrence distribution gives hubness and anti-hubs
- nearest-neighbor cosines give collapse, outliers and distance concentration
- the k-NN graph gives connectivity
- random pairs and the covariance spectrum give anisotropy, effective rank and rogue
  dimensions
- vector norms give the norm spread
- a second inner-product pass (skipped when norms are uniform) gives metric mismatch
- the ANN neighbor file, if present, is compared against the exact neighbors for recall

Query embeddings, if present, get their own exact search against the corpus for the
query-side findings.

Everything is deterministic: fixed seeds, exact neighbors, no learned components, no network.
Two runs on the same file give the same JSON.

Exact k-NN is quadratic in the number of rows, so inputs above 100,000 rows are audited on a
seeded random sample unless you pass `--sample 0`.
