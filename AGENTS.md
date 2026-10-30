# Repository guidelines

`vecscope` gives agents deterministic evidence about embedding spaces and vector indexes.
These rules apply to any agent or person changing this repository.

## Product rules

- Every finding reports a measured value, the verdict, the threshold that produced it, and an
  explanation that states what the number means and what healthy looks like.
- Output is deterministic: fixed seeds, exact neighbors, no learned components, no network.
  Two runs on the same input must produce identical JSON.
- The core depends on NumPy only. Loaders and the MCP server may use optional extras.
- Never guess at embedding quality in docs or messages. Point at the finding.

## Layout

- `src/vecscope/diagnostics/`: one pure function per finding, returning plain Python numbers
- `src/vecscope/evidence.py`: thresholds, `Finding`/`Evidence` models, the `audit` entry point
- `src/vecscope/sweep.py`: HNSW/IVF parameter sweep on top of faiss (optional extra)
- `src/vecscope/cli.py`, `src/vecscope/mcp_server.py`: thin adapters over `audit` and `sweep`
- `.agents/skills/vecscope/SKILL.md`: what an agent should do with each verdict

## Checks before a change is done

```bash
pytest -q
ruff check src tests examples
```

Update `README.md`, `docs/ko/README.md`, the skill file and `CHANGELOG.md` when the
CLI, MCP tool, findings or verdicts change. Keep the English and Korean READMEs in sync,
and every `docs/*.md` with its `docs/ko/*.md` translation.
