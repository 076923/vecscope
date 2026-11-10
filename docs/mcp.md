**English** · [한국어](ko/mcp.md)

# Use with an agent

`vecscope` ships an MCP server with four read-only tools: `audit_embeddings`, `sweep_index`,
`compare_embeddings` and `describe_findings`. Register it with any MCP client:

```bash
pip install vecscope
claude mcp add vecscope -- vecscope-mcp --allow-dir /data/embeddings     # Claude Code
claude mcp add vecscope -- uvx vecscope mcp --allow-dir /data/embeddings # or without installing
```

`vecscope mcp` and `vecscope-mcp` are the same server with the same options. It is listed in
the [MCP Registry](https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.076923/vecscope)
as `io.github.076923/vecscope`.

```json
{ "mcpServers": { "vecscope": { "command": "vecscope-mcp", "args": ["--allow-dir", "/data/embeddings"] } } }
```

The JSON form goes in `.cursor/mcp.json` for Cursor and in `.mcp.json` for a Claude Code
project. Codex reads TOML from `~/.codex/config.toml`:

```toml
[mcp_servers.vecscope]
command = "vecscope-mcp"
args = ["--allow-dir", "/data/embeddings"]
```

VS Code reads `.vscode/mcp.json` in the workspace, under `servers` instead of `mcpServers`.
Use the full path of `vecscope-mcp` (`where vecscope-mcp` on Windows, `which vecscope-mcp`
elsewhere), since VS Code may not see the PATH of your shell:

```json
{
  "servers": {
    "vecscope": {
      "type": "stdio",
      "command": "/path/to/vecscope-mcp",
      "args": ["--allow-dir", "${workspaceFolder}/data"]
    }
  }
}
```

Start it from the **Start** link above the entry or with `MCP: List Servers`, and use the
tools in the Chat view in Agent mode.

Then ask:

```text
Audit the embeddings in data/chunks.npy before we tune the retriever, and tell me
which finding to fix first.
```

## Tools and responses

- `audit_embeddings` runs the audit, with the same inputs as the CLI.
- `sweep_index` runs the [index sweep](usage.md#index-sweep).
- `compare_embeddings` runs the [comparison](usage.md#compare-two-embeddings).
- `describe_findings` lists every finding with its thresholds, causes and actions.

Responses start with a one-line `summary` and an `audit_id`. The summary names the non-ok
findings, retrieval nDCG, the first plan step and, on a re-audit, the comparison with the
baseline.

Pass that id as `baseline` when re-auditing after a change. The response then carries
`baseline_diff` (verdict and value changes, `regressions`, `improvements`, retrieval
changes), as `--baseline` does in the CLI, without the agent saving the earlier response.
The server keeps its last 32 audits in memory; a saved response or `--json` file works too.

`verbose=false` drops the details, explanations and actions of findings that are ok. Agents
without MCP can run the CLI with `--json` and read the same evidence.

The agent skill in [`.agents/skills/vecscope/SKILL.md`](https://github.com/076923/vecscope/blob/main/.agents/skills/vecscope/SKILL.md)
tells the agent when to run the audit and what to do about each verdict: center or swap the
model, deduplicate, re-rank, or rebuild the index. The server also carries those
instructions, so an agent that only has the MCP connection follows the same loop.

## What the server may read

The tools are read-only, but they read whatever path the agent names, so the server limits
where that can be. A path outside the allowed directories (after resolving `..` and
symlinks) is refused before it is opened.

- By default, only the directory it was started in and below.
- `--allow-dir DIR` names the directories instead, as in the setup above. It can be
  repeated; the environment variable `VECSCOPE_ALLOWED_DIRS` (paths separated by `:` on Unix,
  `;` on Windows) does the same.
- `--allow-any` (or `VECSCOPE_ALLOW_ANY=1`) lifts the restriction, so every file the server
  process can read is reachable. Use it only on your own machine.
- Network vector stores need `--allow-network` on top. Read-only does not make reaching them
  harmless, so grant it only where the server's network access is what you intend.

Each file is read through one handle that is checked after opening: it must be a regular
file, and the same file the path resolves to inside the allowed directories. A symlink
swapped between the check and the read, a FIFO or a directory is refused.

Local database directories (`qdrant+local://` and the like) are checked by path only,
because their client opens the files itself.

One request cannot exhaust the server's memory:

- A file larger than `--max-read-mb` (default 4096; `VECSCOPE_MAX_READ_MB`; 0 for no limit)
  is refused before it is read.
- A store stops being read once its vectors pass the same size.
- `--max-rows` and `--max-dim` (`VECSCOPE_MAX_ROWS`, `VECSCOPE_MAX_DIM`; default 0 for no
  limit) refuse a matrix with more rows or columns than that, whatever its byte size; the
  queries file is checked the same way, and a store stops being read at `--max-rows`.
- Files are parsed straight from the checked handle, never copied into memory first. On a
  500,000 x 384 `.npy` file (0.72 GiB) an MCP read peaks at 0.79 GiB, as the CLI does.
- A `MemoryError` while reading or auditing is returned to the agent as a tool error that
  names the ways out (a sample, a smaller input, the limits above) instead of ending the
  call silently.

Calls in parallel, or with expensive settings, are bounded too:

- The server runs each tool call in its own thread. Audits, sweeps and compares take one of
  `--max-concurrent` slots (default 1; `VECSCOPE_MAX_CONCURRENT`), and further calls wait.
  An agent that fires three audits at once therefore holds one corpus in memory, not three.
- Settings that multiply a call's cost are capped per call: `k` and `clusters` at 1,000,
  `sample_repeats` and `timing_repeats` at 10, `n_queries` at 100,000 and `latency_queries`
  at 10,000. The CLI has no caps.

Each audit, sweep and compare writes one line to the server's stderr, which MCP clients keep
in their logs. It names the tool and every file or store it read (credentials redacted), the
verdict or error, and the time taken, including any wait for a slot:

```text
vecscope-mcp: audit_embeddings path=/data/run/corpus.npy qrels_path=/data/run/qrels.tsv: fail in 2.1 s
```

The recent audits that `baseline` refers to are kept behind a lock. A stdio server serves the
one client that started it; agents that run in parallel each start their own server.

## Tested clients

The diagnose, change and re-verify loop is tested over the MCP stdio protocol
(`tests/test_agent_workflow.py`), and was run end to end by Claude Code in headless mode
(`claude -p` with the server above).

It audited a corpus with a shared offset, normalized, centered and re-audited as the plan
said, and stopped at the remaining step it was not asked to fix.

GitHub Copilot in VS Code (Agent mode, `.vscode/mcp.json` above) was run by hand on the
example matrices from `examples/make_examples.py`:

- `healthy.npy`, `anisotropic.npy` and `collapsed.npy` came back ok, warn and fail with the
  values of `vecscope audit`, and the agent recommended the first step of `plan`.
- After centering and re-normalizing `anisotropic.npy`, it re-audited with the first
  `audit_id` as `baseline` and reported anisotropy and antihubs as improvements.
- A file outside `--allow-dir` was refused by the server, and the agent did not try to work
  around the refusal.
- Copilot moves a response of this size (about 13,000 to 15,000 characters with `verbose`
  on) to a file and reads it in parts, which costs a few extra tool calls per audit but did
  not change any answer.

Cursor and Codex use the same protocol and were not run.

[docs/agent_case.md](https://github.com/076923/vecscope/blob/main/docs/agent_case.md) records
two cases on real SciFact embeddings, each run twice:

- **Two ingestion faults**: a batch ingested twice and a worker that skipped normalization.
  From the symptom alone, the agent found both, fixed the corpus and re-audited against its
  first `audit_id`. Production nDCG@10 went from 0.100 to 0.713.
- **An IVF index searched with nprobe=1**: the audit found the embeddings healthy and the
  production ranking 0.29 nDCG below exact search. The agent swept the index, chose nprobe,
  and production nDCG@10 went from 0.422 to 0.700 and 0.711.

## If the server stops connecting

"vecscope-mcp not found" usually means the Python environment it was installed into has
been replaced. This happens on hosted notebooks and clusters that rebuild their environment.

Check with `pip show vecscope` in the interpreter the client uses and reinstall there. Then
register the server through that interpreter, so the error names the right environment:

```json
{ "mcpServers": { "vecscope": { "command": "/path/to/env/bin/python", "args": ["-m", "vecscope.mcp_server"] } } }
```
