"""MCP server exposing vecscope as a tool. Run: `python -m vecscope.mcp_server`."""

from __future__ import annotations

try:  # mcp >= 2.0 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from vecscope.evidence import audit
from vecscope.io import load_vectors

mcp = _Server("vecscope")


@mcp.tool()
def audit_embeddings(
    path: str,
    k: int = 10,
    ann_path: str | None = None,
    queries_path: str | None = None,
    clusters: int = 0,
    near_dup_cos: float = 0.99,
) -> dict:
    """Audit an embedding file: hubness, anisotropy, collapse, rogue dimensions, norms,
    metric mismatch, distance concentration, outliers, k-NN graph islands, and optionally
    ANN recall and query-side findings.

    `path` is a local .npy/.npz/.parquet/.jsonl/.json/.csv with one vector per row.
    `ann_path` is an optional (n x k) integer array of neighbor ids from an ANN index.
    `queries_path` is an optional (m x dim) file of query embeddings.
    `clusters` > 0 attaches a per-cluster report with sample ids.
    Returns deterministic evidence with ok/warn/fail verdicts, thresholds and explanations.
    """
    x = load_vectors(path)
    ann = load_vectors(ann_path).astype(int) if ann_path else None
    q = load_vectors(queries_path) if queries_path else None
    return audit(x, k=k, ann_neighbors=ann, near_dup_cos=near_dup_cos, queries=q,
                 clusters=clusters).to_dict()


@mcp.tool()
def sweep_index(
    path: str,
    k: int = 10,
    queries_path: str | None = None,
    target_recall: float = 0.95,
    device: str = "auto",
) -> dict:
    """Sweep HNSW (M, ef_search) and IVF (nprobe) parameters on an embedding file and report
    recall@k against exact k-NN, build time and queries per second for each setting, plus
    the fastest configuration that reaches `target_recall`. Needs the faiss extra.
    `device` is "auto" (GPU when the faiss build has one), "cpu" or "gpu"; HNSW is CPU-only.
    """
    from vecscope.sweep import sweep

    x = load_vectors(path)
    q = load_vectors(queries_path) if queries_path else None
    return sweep(x, k=k, queries=q, target_recall=target_recall, device=device).to_dict()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
