# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-10-29

### Added

- `--baseline old.json` / `diff_evidence(new, old)`: per-finding value and verdict changes
  against an earlier run, `regressions` for verdicts that got worse, exit 1 on regression.
  Evidence JSON carries `schema_version` (1) and `near_dup_cos`; a baseline from another
  schema is rejected, and one made with a different near-duplicate cutoff or threshold set is
  marked not `comparable` like a different k, so a changed setting is not read as a regression
- Reference tests (`tests/test_reference.py`): the k-NN kernel against a naive full distance
  matrix for all metrics, hubness skewness against SciPy, effective rank of rank-r data,
  hand-computed ann_recall and collapse ratios, and audit invariance to row order, rotation
  and uniform scale; golden evidence for the example matrices (`tests/golden/`, regenerated
  by `examples/make_examples.py`, which now exposes `make_examples()`)
- MCP `sweep_index` takes `n_queries` like the CLI `--n-queries`
- `ann_recall` reports `rows_with_fewer_than_k_ids_ratio` and says in the explanation when the
  ANN file supplied fewer than k ids after dropping self ids, which caps recall below 1;
  `effective_rank` details carry `max_attainable_rank` and the explanation says when the row
  count, not the dimension, bounds the rank; `effective_rank` and `query_hubness` carry their
  numbers in `details`

### Changed

- The exact k-NN pass sizes its row block from the row count so peak memory stays near 1 GiB
  for the distance matrix at any size (100k x 768: 1.9 GiB instead of 4.1 GiB); results do
  not depend on the block size except among exactly tied distances. The README performance
  table lists memory only
- `anisotropy.mean_random_cosine` (and the value after centering) is now the exact mean over
  all pairs, computed in closed form from the sum vector, instead of a 20,000-pair sample.
  Values move by at most the old sampling error (about 0.003) and no longer depend on row
  order; calibration tables are unchanged at three decimals. The noise baseline of
  `distance_concentration` uses the same estimator (the old one was biased by up to 2% below
  200 rows)
- Exact-copy grouping compares rows by direction and by length relative to the data's scale
  (log scale, rounded) instead of at an absolute 1e-6, so vectors of size 1e-7 are no longer
  one duplicate group and short rows with their own direction are not copies of each other.
  Rows that differ by float32 noise can still split on a rounding boundary; they remain
  near-duplicates through the cosine cutoff, only the exact-copy statistics undercount them
- `compare(sample=0)` means every row, as in `audit`; both share one `resolve_sample`
  validator. The CLI and MCP already mapped 0 to None, only the library call differed
- The text output aligns the finding name column to the longest name
  (`distance_concentration`)

### Fixed

Verdicts and thresholds:

- A threshold override with `warn == fail` kept the default direction only by accident: for
  lower-is-worse keys (`ann_recall.recall_at_k`, `metric_mismatch.topk_overlap`,
  `anisotropy.effective_rank_ratio`, `distance_concentration.contrast_over_noise`) it flipped
  the verdict. The direction now always comes from the defaults
- `verify_by` for antihubs and rogue_dimensions names the metric the verdict uses
  (`antihub_excess`, `top1_dim_variance_ratio`); the `distance_concentration` explanation
  quotes the contrast measured on `baseline_rows`, which is the number the verdict uses
  (`relative_contrast_at_baseline_rows` in details); `distance_concentration` called directly
  without `x` on more than 4000 rows raises instead of mislabelling its baseline rows
- `anisotropy.effective_rank` of a constant input is 0, not 1 (float32 rounding left a
  non-zero spectrum)

ANN recall and sweep:

- `ann_recall` and `vecscope sweep` count hits by distance instead of by id: both sides are
  recomputed in float64 from the vectors and returned ids are assigned to exact neighbor
  slots by a maximum matching within a tie tolerance relative to the neighbor distance.
  Exact neighbors at one distance are interchangeable, so duplicate groups larger than k no
  longer cap recall (an exact index on a corpus with 8 copies per item scored 0.72 and FAIL,
  now 1.0), while a missed closer neighbor stays a miss
- `sweep` validates the corpus and the queries before casting to float32 (NaN, Inf and values
  beyond float32 get the "rescale" message instead of garbage recall); `--device gpu` without
  a GPU build prints a one-line error and exits 2 instead of a traceback
- An ANN neighbor array with zero columns is rejected with a clear error instead of
  producing NaN recall

Duplicates and degenerate corpora:

- `collapse.near_duplicate_ratio` counts every exact copy of another row, including zero
  vectors, whose cosine with their own copies is 0; an all-zero matrix used to pass collapse
  as ok. Their `sample_pairs` partner is another member of the exact group. The per-cluster
  `near_duplicate_ratio` and `compare` use the same rule, applied in float64, so the cluster
  map shows where zero-vector duplicates live. `nn_cosine_median` / `nn_cosine_p95` are
  clamped to 1
- `outliers` and `distance_concentration` report 0 with `distinct_neighbors: false` when every
  row's nearest neighbor is a copy of it, instead of measuring float32 rounding noise;
  `ood_queries` reports 0 with `corpus_has_distinct_neighbors: false` when every corpus row
  has an exact copy (there is no reference distance), instead of flagging every query
- `compare` treats two nearest partners that are exact copies of each other in both spaces
  as the same partner, so copy groups no longer show up as disagreements; it computes the
  copy groups of each space once instead of twice

Row ids and determinism:

- Row ids in `details.sample_ids`, `details.sample_pairs`, the cluster report and
  `compare`'s `sample_disagreements` are ids of the input rows when the audit was sampled;
  they used to index the subsample, so an agent opened the wrong rows
- `audit(seed=...)` is forwarded to the anisotropy SVD subset and the distance-concentration
  subsample as well as to row sampling; ordered id lists (`sample_ids`, `rogue_dims`,
  `ood_sample_ids`, cluster samples) use stable sorts; among exactly tied neighbor distances
  the selection is NumPy's, identical between runs on one NumPy build
- `audit(near_dup_cos=np.float32(...))` no longer leaves a NumPy scalar in the JSON

Input validation and loaders:

- Values above 1e16 in magnitude are rejected with a message to rescale: they are finite,
  but their squares overflow float32 and used to give silently wrong neighbors. The check
  runs before the float32 cast in `audit`, `compare` and `sweep`, and JSON/JSONL/parquet/text
  vectors are parsed as float64, so float64 values beyond float32 get the "rescale" message
  instead of "NaN or Inf"; the int64 minimum no longer slips past the check; vector norms are
  computed in float64 so large-but-safe values do not overflow during normalization
- Query vectors are validated before the corpus k-NN pass (a bad query file fails at once);
  a query file with zero rows is rejected (it produced NaN values and invalid JSON)
- Input with zero columns is rejected in `audit`, `compare`, `sweep` and the loaders (it
  crashed with a ZeroDivisionError, and `compare` reported perfect agreement); `compare`
  rejects `k` >= rows and `near_dup_cos` outside (0, 1] like `audit`; a threshold pair given
  as a string, and a malformed finding passed to `diff_evidence`, are rejected with a message
- Neighbor files: ids are read as exact integers everywhere (parquet ids above 2**24 were
  rounded; `.json`/`.jsonl` ids with a fractional part are rejected instead of truncated;
  `12.0` in CSV/TSV loads as an integer); `null`, `inf`, boolean arrays and unsigned ids
  beyond the int64 range are rejected as non-integer instead of failing with a TypeError
- Embedding files: an empty `.npy` or a truncated `.npz` is reported as a user error (exit 2 /
  ToolError) instead of an EOFError or BadZipFile traceback; a zip archive named `.npy` (or an
  array named `.npz`) is a clear error instead of an AttributeError; string or datetime
  dtypes, scalar JSON rows, nested lists, objects inside rows, booleans, empty JSON and empty
  CSV/TSV/TXT files get clear errors instead of a TypeError, a 3-D array or a NumPy warning
- A missing neighbor file and a `--baseline` / `--thresholds` file that is not JSON report a
  clear error (exit 2); the `--clusters` cap (one per 10 rows) is stated in the help texts
- `examples/make_examples.py` run outside the repository writes only the matrices: it checks
  for the vecscope source tree instead of any `tests/` directory next to it, so a downloaded
  copy no longer writes `tests/golden/` into a stranger's project

## [0.3.0] - 2026-10-28

### Added

- `sample_repeats` (`audit(sample_repeats=...)`, `--sample-repeats`, MCP `sample_repeats`):
  when sampling applies, re-audits independent seeded samples and attaches `value_range` and
  `verdict_stable` to each finding plus a top-level `unstable_findings` list
- Retrieval-quality study (`docs/retrieval_quality.md`, `examples/retrieval_quality.py`):
  every finding against nDCG@10 on BEIR SciFact and NFCorpus for eight public models and for
  controlled defects; README section summarizing what the findings do and do not predict
- `collapse` details now describe the duplicate structure: `near_duplicate_groups`,
  `largest_near_duplicate_groups`, `largest_exact_duplicate_groups` and `sample_pairs`
  (row, row, cosine; one pair per group, largest groups first), so an agent can tell
  duplicate items (many small groups) from an embedding failure (a few huge groups) without
  extra scripts. Groups link every neighbor within the cutoff among the k audited, so a
  jittered boilerplate cluster is one group rather than many nearest-neighbor pairs. The
  explanation and the suggested action say which is which; the action previously pointed at
  a `sample_ids` field that collapse never had
- `vecscope compare A B` / `compare()` / MCP `compare_embeddings`: agreement between two
  embeddings of the same rows (nearest-neighbor and top-k overlap, near-duplicate share in
  each space and in both, same duplicate partner, largest disagreements); no verdicts
- Sweep recommendations carry `margin` (recall minus target) and are marked `tight` with a
  `safer_alternative` when the margin is under `--margin` (default 0.02); results carry
  `caveats` (in-corpus queries, recall falls with corpus size, qps is relative) which the CLI
  prints and the MCP docstring tells the agent to relay. qps is the median of
  `--timing-repeats` runs (default 3) instead of a single timing
- Project metadata: description, keywords and Changelog/Documentation URLs on PyPI
- README: troubleshooting for an MCP server that stops connecting after the Python
  environment is rebuilt, with a `python -m vecscope.mcp_server` registration

### Changed

- `anisotropy` thresholds (0.3, 0.6) -> (0.5, 0.85): the best models on labeled data sit at
  0.58 to 0.81 with no ranking cost, and a shared offset up to 0.98 left nDCG@10 unchanged
- `distance_concentration` now reports `contrast_over_noise`: the relative contrast divided
  by that of seeded isotropic noise at the same dimension (`details` keep `relative_contrast`
  and `noise_relative_contrast`). The raw contrast of pure noise falls from 1.43 at 128-d to
  1.10 at 1536-d, so no fixed threshold worked across models; the ratio is 1.0 for noise at
  any width. Thresholds (1.4, 1.1): noise that halves nDCG@10 gives 1.05 and noise that
  costs 4% gives 1.27; sixteen real model/corpus pairs sit at 1.43 to 2.9. Threshold key renamed to `distance_concentration.contrast_over_noise`; an override
  that still uses the old key gets an error naming the new one. The nearest-neighbor
  distance also shrinks as rows are added, so the data is measured on a seeded subsample of
  the same size as the noise baseline (4000 rows; `details.baseline_rows`): pure noise is
  1.0 at any n instead of drifting to 1.1 at 100k rows
- `ood_queries` drops duplicate corpus rows with the same relative cutoff as the other
  findings instead of a fixed 1e-4
- MCP `sweep_index` exposes `margin` and `timing_repeats`
- `examples/healthy.npy` and the healthy test scenarios now have cluster structure; pure
  isotropic noise is reported as having no neighbor signal, which is correct

### Fixed

- Text loaders (`.csv`, `.tsv`, `.txt`) read UTF-8 with a BOM and always return 2-D arrays
  (a single-row file used to load as 1-D)

## [0.2.0] - 2026-10-27

### Added

- `thresholds` override (`audit(thresholds=...)`, `--thresholds file.json`, MCP `thresholds`);
  effective thresholds are echoed in the JSON result
- Every finding carries `likely_causes`, `suggested_action` and `verify_by`; the text output
  adds a `try:` line for findings that are not `ok`
- Scenario test suite (`tests/test_scenarios.py`): 13 synthetic defects, 11 degenerate inputs,
  determinism
- README: scope statement (vector search implementation, not relevance), threshold caveats,
  sampling caveats, performance table
- MCP: server instructions with the diagnose/change/re-verify loop, read-only tool
  annotations, `sample` and `verbose` parameters, a one-line `summary` first in every response,
  and a `describe_findings` tool

### Changed

- Agent skill rewritten around a diagnose → smallest change → re-audit → compare loop
- `antihubs` now reports the excess over the random baseline e^-k (`antihub_excess`), so small
  k no longer trips it on healthy data; `rogue_dimensions` now reports the top dimension's
  variance as a multiple of the uniform 1/dim share (`top1_dim_variance_ratio`, warn 10x,
  fail 50x) instead of an absolute share that always warned on low-dimensional vectors
- `ann_neighbors` combined with sampling now raises instead of being silently dropped

### Fixed

- `connectivity` capped label propagation at 1000 rounds and reported one long-chain component
  as thousands of islands; it now converges fully (pointer jumping)
- `vecscope sweep` removed "self" by position, so duplicate or zero rows capped recall below
  1.0; self is now removed by id for both exact and approximate lists
- ANN neighbor files are validated (shape, integer ids, id range); ids that point at the item
  itself are ignored; ids are loaded as integers so values above 2^24 survive
- Parquet loader picks a list-of-numbers column by type (no longer grabs `embedding_model`)
- JSON/JSONL loader skips non-numeric lists (tags), rejects empty files and ragged rows with
  clear errors
- `audit` validates `k`, `sample`, `clusters`, `near_dup_cos`, `thresholds` (type, finiteness)
  and rejects complex/object dtypes with clear errors instead of crashing deep inside NumPy
- `sweep` validates `k < n`, `n_queries >= 1`, `target_recall`, query dimensionality
- MCP tools raise `ToolError`, so the real message (missing file, bad k) reaches the agent
  instead of "Error executing tool"
- CLI prints a one-line error and exits 2 on bad input instead of a traceback

## [0.1.1] - 2026-10-26

### Fixed

- MCP server failed to import with `mcp` 2.x, which renamed `FastMCP` to `MCPServer`; both
  1.x and 2.x are now supported and the server is smoke-tested in CI

### Added

- Python 3.14 classifier and a `.gitattributes` that normalizes line endings

## [0.1.0] - 2026-10-25

### Added

- `vecscope audit` CLI with text and `--json` output, `--fail-on` gate and non-zero exit on FAIL
- Findings: hubness, antihubs, anisotropy, effective rank, collapse, rogue dimensions, norms,
  metric mismatch, distance concentration, outliers, k-NN graph connectivity, ANN recall
- Seeded sampling (`--sample`, default 100k rows) so large matrices stay tractable
- MCP server (`vecscope-mcp`) exposing `audit_embeddings`
- Query-side findings (`--queries`): out-of-distribution queries and query hubness
- Per-cluster report with sample ids (`--clusters N`)
- `vecscope sweep` / MCP `sweep_index`: HNSW and IVF parameter sweep with recall, build time and
  QPS, plus a recommended configuration (optional `faiss-cpu` extra); IVF and the exact
  baseline run on the GPU automatically with a faiss GPU build (`--device auto|cpu|gpu`)
- Loaders for `.npy`, `.npz`, `.parquet`, `.jsonl`, `.json`, `.csv`
- Agent skill file describing when to run the audit and how to act on verdicts
- Each finding reports `healthy_range`, `warn_at`, `fail_at`; `anisotropy` also reports the
  mean-vector norm and the random-pair cosine after centering
- Thresholds calibrated on public embeddings from seven models over eight datasets (OpenAI ada-002 and 3-large,
  Cohere embed-v3, all-MiniLM-L6-v2, bge-m3, mxbai-embed-large-v1, gte-base-en-v1.5);
  `examples/real_models.py` reproduces the table in the README
