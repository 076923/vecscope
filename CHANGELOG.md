# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
  clear errors; text loaders read UTF-8 with BOM and always return 2-D arrays
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
- Thresholds calibrated on public embeddings from eight models (OpenAI ada-002 and 3-large,
  Cohere embed-v3, all-MiniLM-L6-v2, bge-m3, mxbai-embed-large-v1, gte-base-en-v1.5);
  `examples/real_models.py` reproduces the table in the README
