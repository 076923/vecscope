# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
