# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
