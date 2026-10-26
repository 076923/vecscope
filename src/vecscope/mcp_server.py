"""MCP server exposing vecscope as tools. Run: `vecscope-mcp` or `python -m vecscope.mcp_server`.

Works with mcp 1.x (FastMCP) and 2.x (MCPServer). All tools are read-only: they read local
files and return JSON; nothing is written or sent anywhere.
"""

from __future__ import annotations

import inspect

try:  # mcp >= 2.0 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

try:
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # mcp 1.x
    try:
        from mcp.server.fastmcp.exceptions import ToolError
    except ImportError:  # pragma: no cover
        ToolError = ValueError  # type: ignore[misc,assignment]

from vecscope.evidence import ACTIONS, THRESHOLDS, audit
from vecscope.io import load_neighbors, load_vectors

_USER_ERRORS = (ValueError, TypeError, FileNotFoundError, ImportError, OSError)


def _user_error(e: Exception) -> ToolError:
    """Return a ToolError whose message reaches the agent (plain exceptions are masked)."""
    return ToolError(f"{type(e).__name__}: {e}")

INSTRUCTIONS = """vecscope gives deterministic evidence about an embedding matrix and a vector
index: per-finding measurements, ok/warn/fail verdicts, thresholds, likely causes, the smallest
change to try, and how to verify it. It diagnoses the vector search implementation (model
geometry, corpus hygiene, metric, index, query pipeline), not relevance.

Workflow: (1) call audit_embeddings and report the failing findings with their values and
ranges verbatim; (2) separate causes using likely_causes and the other findings; (3) apply the
smallest change from suggested_action, applying any vector transform to queries too; (4)
re-audit and compare retrieval on real queries (verify_by says which number should move);
(5) adopt only confirmed improvements. Thresholds are calibrated defaults, not a universal
bar: a warn on a good model is common. Use sweep_index before choosing HNSW/IVF parameters."""


def _annotations():
    """Read-only, idempotent tool annotations across mcp versions (field names differ)."""
    try:
        from mcp.types import ToolAnnotations
    except ImportError:  # pragma: no cover - very old mcp
        return None
    for kwargs in (
        {"read_only_hint": True, "idempotent_hint": True, "open_world_hint": False},
        {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
    ):
        try:
            return ToolAnnotations(**kwargs)
        except (TypeError, ValueError):
            continue
    return None  # pragma: no cover


def _server():
    kwargs = {"name": "vecscope"}
    if "instructions" in inspect.signature(_Server.__init__).parameters:
        kwargs["instructions"] = INSTRUCTIONS
    return _Server(**kwargs)


mcp = _server()
_TOOL_KW = {}
if "annotations" in inspect.signature(mcp.tool).parameters and _annotations() is not None:
    _TOOL_KW["annotations"] = _annotations()


def _summary(ev: dict) -> str:
    bad = [f for f in ev["findings"] if f["verdict"] != "ok"]
    head = f"verdict={ev['verdict']} n={ev['n']} dim={ev['dim']} k={ev['k']}"
    if ev.get("sampled"):
        head += f" (sampled from {ev['n_total']} rows)"
    if not bad:
        return head + "; all findings ok"
    parts = [f"{f['name']}={f['value']:.3g} {f['verdict']} ({f['healthy_range']})" for f in bad]
    return head + "; " + "; ".join(parts)


@mcp.tool(**_TOOL_KW)
def audit_embeddings(
    path: str,
    k: int = 10,
    ann_path: str | None = None,
    queries_path: str | None = None,
    clusters: int = 0,
    near_dup_cos: float = 0.99,
    sample: int = 100_000,
    thresholds: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Audit an embedding file: hubness, anisotropy, collapse, rogue dimensions, norms,
    metric mismatch, distance concentration, outliers, k-NN graph islands, and optionally
    ANN recall and query-side findings. Read-only; the file never leaves the machine.

    `path`: local .npy/.npz/.parquet/.jsonl/.json/.csv with one vector per row.
    `ann_path`: optional (n x k) integer array of neighbor ids from an ANN index (requires
    `sample=0` so row ids line up).
    `queries_path`: optional (m x dim) file of query embeddings; adds ood_queries and
    query_hubness.
    `clusters` > 0 attaches a per-cluster map with sample ids (no verdicts).
    `sample`: rows above this are audited on a seeded random subset (0 = audit every row;
    exact k-NN is quadratic, ~3 min for 100k x 768 on a laptop).
    `thresholds`: override default (warn, fail) pairs, e.g. {"hubness.skewness": [2.5, 5]}.
    `verbose=False` drops details, causes and actions for findings that are ok, which cuts
    the response roughly in half.

    The response starts with `summary` (one line) and `verdict`. Each finding carries value,
    healthy_range, warn_at, fail_at, explanation, details, likely_causes, suggested_action and
    verify_by. Treat actions as hypotheses: apply the smallest change, re-audit, and compare
    retrieval on real queries before adopting.
    """
    try:
        x = load_vectors(path)
        ann = load_neighbors(ann_path) if ann_path else None
        q = load_vectors(queries_path) if queries_path else None
        ev = audit(
            x, k=k, ann_neighbors=ann, near_dup_cos=near_dup_cos, sample=sample or None,
            queries=q, clusters=clusters, thresholds=thresholds,
        ).to_dict()
    except _USER_ERRORS as e:
        raise _user_error(e) from e
    if not verbose:
        for f in ev["findings"]:
            if f["verdict"] == "ok":
                f["details"] = {}
                f["likely_causes"], f["suggested_action"], f["verify_by"] = [], "", ""
    return {"summary": _summary(ev), "verdict": ev["verdict"], **ev}


@mcp.tool(**_TOOL_KW)
def sweep_index(
    path: str,
    k: int = 10,
    queries_path: str | None = None,
    target_recall: float = 0.95,
    device: str = "auto",
) -> dict:
    """Sweep HNSW (M, ef_search) and IVF (nprobe) parameters on an embedding file and report
    recall@k against exact k-NN, build time and queries per second for each setting, plus
    the fastest configuration that reaches `target_recall`. Needs the faiss extra
    (`pip install "vecscope[index]"`). Read-only; timings are relative to this machine.
    `device` is "auto" (GPU when the faiss build has one), "cpu" or "gpu"; HNSW is CPU-only.
    """
    from vecscope.sweep import sweep

    try:
        x = load_vectors(path)
        q = load_vectors(queries_path) if queries_path else None
        res = sweep(x, k=k, queries=q, target_recall=target_recall, device=device).to_dict()
    except (*_USER_ERRORS, RuntimeError) as e:
        raise _user_error(e) from e
    r = res["recommendation"]
    params = ", ".join(f"{a}={b}" for a, b in r["params"].items())
    res["summary"] = (f"recommend {r['index']} ({params}): recall@{k}={r['recall_at_k']:.3f}, "
                      f"{r['queries_per_second']:.0f} qps on {res['device']}; {r['reason']}")
    return res


@mcp.tool(**_TOOL_KW)
def describe_findings() -> dict:
    """List every finding vecscope can report with its metric key, default (warn, fail)
    thresholds and direction, likely causes, suggested first action and verification step.
    Use it to explain a verdict to the user or to decide which thresholds to override.
    """
    from vecscope.evidence import _FINDING_KEYS

    out = {}
    for name, key in _FINDING_KEYS.items():
        warn, fail = THRESHOLDS[key]
        causes, action, verify = ACTIONS[name]
        out[name] = {
            "threshold_key": key,
            "warn_at": warn,
            "fail_at": fail,
            "lower_is_worse": warn > fail,
            "likely_causes": list(causes),
            "suggested_action": action,
            "verify_by": verify,
        }
    return out


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
