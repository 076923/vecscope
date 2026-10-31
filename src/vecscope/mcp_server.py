"""MCP server exposing vecscope as tools. Run: `vecscope-mcp` or `python -m vecscope.mcp_server`.

Works with mcp 1.x (FastMCP) and 2.x (MCPServer). All tools are read-only: they read local
files and return JSON; nothing is written or sent anywhere. Read-only is not the same as
harmless: the tools read whatever path the agent names, so they are confined to directories.

- By default only the directory the server was started in (and below) is readable.
- ``--allow-dir DIR`` (repeatable) or ``VECSCOPE_ALLOWED_DIRS`` (paths separated by
  ``os.pathsep``) names the directories instead.
- ``--allow-any`` (or ``VECSCOPE_ALLOW_ANY=1``) lifts the restriction: every file the
  process can read is reachable, which is fine on a developer machine and not on a shared one.

A file is read through one handle that is checked to be a regular file inside those
directories after it is opened, so a symlink swapped between the check and the read is
refused. A local database directory (``qdrant+local://`` and the like) is checked by path
only, because its client opens the files itself. Vector stores reached over the network
(qdrant://, chroma://, milvus://, weaviate://) are refused unless the server is started with
``--allow-network`` (or ``VECSCOPE_ALLOW_NETWORK=1``); they are then read, never written,
with a per-request timeout (``timeout=`` in the URI, default 60 s).

A file larger than ``--max-read-mb`` (default 4096, ``VECSCOPE_MAX_READ_MB``, 0 = no limit)
is refused before it is read, and a store stops being read once its vectors pass that size,
so one request cannot exhaust the server's memory. The recent-audit memory that ``baseline``
refers to is guarded by a lock.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
import sys
import threading
from collections import OrderedDict
from pathlib import Path

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

from vecscope.evidence import ACTIONS, THRESHOLDS, audit, diff_evidence, same_corpus
from vecscope.io import load_neighbors, load_vectors

_USER_ERRORS = (ValueError, TypeError, FileNotFoundError, ImportError, OSError)
ALLOWED_DIRS_ENV = "VECSCOPE_ALLOWED_DIRS"
ALLOW_NETWORK_ENV = "VECSCOPE_ALLOW_NETWORK"
ALLOW_ANY_ENV = "VECSCOPE_ALLOW_ANY"
MAX_READ_ENV = "VECSCOPE_MAX_READ_MB"
DEFAULT_MAX_READ_MB = 4096
# recent audits of this server process by audit_id, so a re-audit can name its baseline
# without the agent writing the earlier response to a file; held in memory only
_RECENT: OrderedDict[str, dict] = OrderedDict()
_RECENT_MAX = 32
_RECENT_LOCK = threading.Lock()


def _audit_id(ev: dict) -> str:
    """Content hash of the evidence: the same input and settings give the same id."""
    blob = json.dumps(ev, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _remember(audit_id: str, ev: dict) -> None:
    with _RECENT_LOCK:
        _RECENT[audit_id] = ev
        _RECENT.move_to_end(audit_id)
        while len(_RECENT) > _RECENT_MAX:
            _RECENT.popitem(last=False)


def _load_baseline(ref: str) -> dict:
    """An audit_id returned earlier by this server, or a JSON evidence file."""
    with _RECENT_LOCK:
        if ref in _RECENT:
            return _RECENT[ref]
    if len(ref) == 16 and all(c in "0123456789abcdef" for c in ref) and not Path(ref).exists():
        raise ValueError(f"baseline {ref!r} is not an audit_id of this server session (the "
                         f"server keeps the last {_RECENT_MAX}); pass the evidence JSON path")
    base = _json_file(ref, "baseline")
    if not isinstance(base, dict) or not isinstance(base.get("findings"), list):
        raise TypeError(f"baseline {ref}: not vecscope evidence (no findings list)")
    return base


def _user_error(e: Exception) -> ToolError:
    """Return a ToolError whose message reaches the agent (plain exceptions are masked)."""
    return ToolError(f"{type(e).__name__}: {e}")


def allowed_dirs() -> list[Path]:
    """Directories the tools may read from, resolved: --allow-dir / VECSCOPE_ALLOWED_DIRS when
    set, otherwise the directory the server runs in. Empty (every readable file) only with
    --allow-any / VECSCOPE_ALLOW_ANY=1 and no --allow-dir."""
    raw = os.environ.get(ALLOWED_DIRS_ENV, "")
    dirs = [Path(p).resolve() for p in raw.split(os.pathsep) if p.strip()]
    if dirs:
        return dirs
    if os.environ.get(ALLOW_ANY_ENV) == "1":
        return []
    return [Path.cwd().resolve()]


def max_read_bytes() -> int | None:
    """The most one file or store may contribute, from VECSCOPE_MAX_READ_MB (0 = no limit)."""
    raw = os.environ.get(MAX_READ_ENV, "")
    try:
        mb = float(raw) if raw.strip() else DEFAULT_MAX_READ_MB
    except ValueError:
        raise ValueError(f"{MAX_READ_ENV}={raw!r} is not a number of MiB") from None
    return None if mb <= 0 else int(mb * 2**20)


def read_checked(path: str, what: str = "path") -> bytes:
    """The content of ``path`` read through one handle that is verified to be a regular file
    inside the allowed directories.

    The path is checked, then opened, then the open handle is compared (device and inode)
    with the file the path resolves to now. A symlink swapped between the check and the open
    (time-of-check to time-of-use) makes the two differ and the read is refused, so the bytes
    returned always come from a file inside the allowed directories."""
    check_path(path, what)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)  # O_NONBLOCK: a FIFO cannot stall the server
    except FileNotFoundError:
        raise FileNotFoundError(f"{what} not found: {path}") from None
    except (IsADirectoryError, PermissionError) as e:
        raise ValueError(f"{what} {path!r} cannot be read as a file ({e.strerror})") from None
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{what} {path!r} is not a regular file")
        limit = max_read_bytes()
        if limit is not None and opened.st_size > limit:
            raise ValueError(f"{what} {path!r} is {opened.st_size / 2**20:.0f} MiB, over the "
                             f"{limit / 2**20:.0f} MiB this server reads; start it with "
                             "--max-read-mb to raise the limit")
        if allowed_dirs():
            check_path(path, what)
            now = os.stat(Path(path).resolve(strict=True))
            if (now.st_dev, now.st_ino) != (opened.st_dev, opened.st_ino):
                raise PermissionError(f"{what} {path!r} changed while it was being opened; "
                                      "refusing to read it")
        with os.fdopen(fd, "rb") as fh:
            fd = None
            data = fh.read() if limit is None else fh.read(limit + 1)
        if limit is not None and len(data) > limit:  # grew after the size check
            raise ValueError(f"{what} {path!r} grew past {limit / 2**20:.0f} MiB while being read")
        return data
    finally:
        if fd is not None:
            os.close(fd)


def _json_file(path: str, what: str) -> dict:
    try:
        return json.loads(read_checked(path, what).decode("utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"{what} {path}: not valid JSON ({e})") from None


def _vectors(path: str, what: str = "path"):
    return load_vectors(path, read_checked(path, what))


def _neighbors(path: str | None, what: str):
    return load_neighbors(path, read_checked(path, what)) if path else None


def _open_source(path: str):
    """A faiss index or vector store named by ``path``: local ones go through the directory
    check, network ones need the server started with --allow-network."""
    from vecscope.adapters import load_source, local_path, scheme_of

    scheme = scheme_of(path)
    if scheme in ("", "faiss"):  # one index file: read through a verified handle
        return load_source(path, read_checked(local_path(path) if scheme else path),
                           max_bytes=max_read_bytes())
    if scheme.endswith("+local"):
        # a local database directory: its client opens the files itself, so only the
        # directory is checked
        check_path(local_path(path))
    elif os.environ.get(ALLOW_NETWORK_ENV) != "1":
        raise PermissionError(f"{scheme} sources reach the network; start the server with "
                              f"--allow-network (or {ALLOW_NETWORK_ENV}=1) to read them")
    return load_source(path, max_bytes=max_read_bytes())


def check_path(path: str | None, what: str = "path") -> str | None:
    """Refuse a path outside the allowed directories (symlinks resolved) before it is opened.
    Returns the path unchanged when it is allowed or when no restriction is configured."""
    if path is None:
        return None
    roots = allowed_dirs()
    if not roots:
        return path
    resolved = Path(path).resolve()
    for root in roots:
        try:
            resolved.relative_to(root)
            return path
        except ValueError:
            continue
    raise PermissionError(
        f"{what} {path!r} is outside the directories this server may read "
        f"({', '.join(str(r) for r in roots)}); start it with --allow-dir DIR (or "
        f"{ALLOWED_DIRS_ENV}), or with --allow-any to read every file the process can"
    )

INSTRUCTIONS = """vecscope gives deterministic evidence about an embedding matrix and a vector
index: per-finding measurements, ok/warn/fail verdicts, thresholds, likely causes, the smallest
change to try, and how to verify it. It diagnoses the vector search implementation (model
geometry, corpus hygiene, metric, index, query pipeline), not relevance.

Workflow: (1) call audit_embeddings and report the failing findings with their values and
ranges verbatim; (2) follow the response's plan, which groups the non-ok findings by shared
cause in fix order, one step at a time; (3) apply that step's first_change, applying any
vector transform to queries too; (4) re-audit with baseline set to the earlier audit_id
and qrels_path on both runs when labels exist; judge the change by baseline_diff, the step's
re_measure numbers and retrieval.ndcg_at_k; (5) adopt only confirmed improvements. Thresholds are calibrated defaults, not a universal
bar: a warn on a good model is common, and neither a fail nor all-ok is a statement about
relevance; say so. Use sweep_index before choosing HNSW/IVF parameters, and
compare_embeddings when two models embedded the same items (or the same model before and
after a change) to tell item problems from model problems; quote its neighbor_scope, because
a sampled compare without supplied neighbors describes the sample's neighbor graph only."""


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
    if ev.get("unstable_findings"):
        head += f"; verdict unstable across sample repeats: {', '.join(ev['unstable_findings'])}"
    if ev.get("inconclusive_findings"):
        head += (f"; inconclusive on this sample (not evidence of health): "
                 f"{', '.join(ev['inconclusive_findings'])}")
    if not bad:
        head += "; all findings ok"
    else:
        head += "; " + "; ".join(f"{f['name']}={f['value']:.3g} {f['verdict']}" for f in bad)
    r = ev.get("retrieval")
    if r:
        head += f"; retrieval nDCG@{r['k']}={r['exact']['ndcg_at_k']:.3f} exact"
        if r.get("ranked"):
            head += f", {r['ranked']['ndcg_at_k']:.3f} ranked"
    if ev.get("plan"):
        step = ev["plan"][0]
        head += (f"; plan step 1 of {len(ev['plan'])}: {step['group']} "
                 f"({', '.join(f['name'] for f in step['findings'])}): {step['first_change']}")
    d = ev.get("baseline_diff")
    if d:
        head += f"; vs baseline: verdict {d['baseline_verdict']} -> {d['verdict']}"
        head += f", regressions: {', '.join(d['regressions']) or 'none'}"
        head += f", improved: {', '.join(d['improvements']) or 'none'}"
        for c in d.get("retrieval_changes") or []:
            if c["metric"] == "ndcg_at_k":
                head += f", {c['ranking']} nDCG {c['baseline_value']:.3f} -> {c['value']:.3f}"
        if not d["comparable"]:
            head += " (settings differ from the baseline: not comparable)"
    if ev.get("query_reference_note"):
        head += "; " + ev["query_reference_note"]
    return head


@mcp.tool(**_TOOL_KW)
def audit_embeddings(
    path: str,
    k: int = 10,
    ann_path: str | None = None,
    queries_path: str | None = None,
    clusters: int = 0,
    near_dup_cos: float = 0.99,
    sample: int = 100_000,
    sample_repeats: int = 1,
    thresholds: dict | None = None,
    verbose: bool = True,
    query_baseline_path: str | None = None,
    qrels_path: str | None = None,
    ranked_path: str | None = None,
    ann_from_source: bool = False,
    baseline: str | None = None,
    tolerances: dict | None = None,
) -> dict:
    """Audit an embedding file: hubness, anisotropy, collapse, rogue dimensions, norms,
    metric mismatch, distance concentration, outliers, k-NN graph islands, and optionally
    ANN recall and query-side findings. Read-only; the file never leaves the machine.

    `path`: local .npy/.npz/.parquet/.jsonl/.json/.csv with one vector per row, a faiss index
    file, or a vector store URI (qdrant://, chroma://, milvus://, weaviate://, or their
    +local forms with a directory). When the server was started with --allow-dir or
    VECSCOPE_ALLOWED_DIRS, paths outside those directories are refused; network stores need
    the server started with --allow-network. Store rows are in ascending id order and
    findings carry `sample_source_ids` next to `sample_ids`.
    `ann_from_source`: query the store's own index for every stored vector and score it as
    `ann_path` (needs `sample=0` above the sample size).
    `qrels_path`: relevance judgments for the queries (.json/.jsonl/.tsv of query row, corpus
    row, grade); adds `retrieval` with nDCG, recall, MRR and duplicate share of exact search.
    `ranked_path`: (m x k) corpus row ids per query from the production index, scored with
    `qrels_path` next to exact search.
    `ann_path`: optional (n x k) integer array of neighbor ids from an ANN index (requires
    `sample=0` so row ids line up).
    `queries_path`: optional (m x dim) file of query embeddings; adds query_gap,
    query_hubness and the deprecated ood_queries.
    `query_baseline_path`: optional JSON evidence (this tool's response or `vecscope audit
    --json`) from an earlier audit of the same corpus with healthy queries; with
    `queries_path` it adds query_shift, the share of queries farther from the corpus than
    that reference's median + 3 MAD.
    `clusters` > 0 attaches a per-cluster map with sample ids (no verdicts); capped at one
    cluster per 10 rows.
    `sample`: rows above this are audited on a seeded random subset (0 = audit every row;
    exact k-NN is quadratic, minutes for 100k rows on a laptop; peak memory about 2 GiB).
    `sample_repeats`: when sampling applies, re-audit this many independent samples; each
    finding then carries `value_range` [min, max] (the spread between samples, not an
    interval for the full-corpus value) and `verdict_stable`, and the response lists
    `unstable_findings`. Costs that many times the audit time. A sampled audit also lists
    `inconclusive_findings`: hubness, antihubs and connectivity verdicts other than fail,
    and outlier verdicts other than ok, which the sample systematically mis-states; do not
    report them as healthy. The collapse verdict uses the estimated full share
    (`details.estimated_full_share`).
    `thresholds`: override default (warn, fail) pairs, e.g. {"hubness.skewness": [2.5, 5]}.
    `baseline`: the `audit_id` of an earlier call in this session, or a JSON evidence file;
    adds `baseline_diff` (per-finding value and verdict changes, `regressions`,
    `improvements`, `retrieval_changes` when both runs had the same qrels) and a "vs
    baseline" part in `summary`. With `queries_path`, a baseline audited with queries also
    supplies the reference for query_shift, but only when it was audited on the same corpus
    or an unchanged prefix of it (`query_reference_note` says when it was skipped). `tolerances` maps a finding to the worsening it
    may accumulate, e.g. {"collapse": 0.01}; beyond it the finding is a value regression.
    `verbose=False` drops details, causes, actions and explanations of findings that are ok
    and the thresholds table; it saves most when most findings are ok.

    The response starts with `summary` (one line: non-ok findings, retrieval, the first plan
    step and the baseline comparison), `verdict` and `audit_id`. Each finding carries value,
    healthy_range, warn_at, fail_at, explanation, details, likely_causes, suggested_action and
    verify_by. Treat actions as hypotheses: apply the smallest change, re-audit, and compare
    retrieval on real queries before adopting.
    """
    try:
        from vecscope.adapters import annotate, is_source
        from vecscope.retrieval import load_qrels

        src = _open_source(path) if is_source(path) else None
        x = src.vectors if src is not None else _vectors(path)
        if ann_from_source and src is None:
            raise ValueError("ann_from_source needs a faiss index or store as path")
        ann = _neighbors(ann_path, "ann_path")
        if ann_from_source:
            ann = src.neighbors(k)
        if src is not None:
            src.close()
        qrels = (load_qrels(qrels_path, read_checked(qrels_path, "qrels_path"))
                 if qrels_path else None)
        ranked = _neighbors(ranked_path, "ranked_path")
        q = _vectors(queries_path, "queries_path") if queries_path else None
        if tolerances is not None and baseline is None:
            raise ValueError("tolerances need baseline")
        base = _load_baseline(baseline) if baseline else None
        qbase, note = None, None
        if query_baseline_path:
            qbase = _json_file(query_baseline_path, "query_baseline_path")
        elif q is not None and base is not None and any(
                f.get("name") == "query_gap" for f in base["findings"]):
            if same_corpus(base, x):
                qbase = base
            else:
                note = ("query_shift skipped: the baseline was audited on a different corpus "
                        "(rows changed or transformed), so its query distances are not a "
                        "reference here")
        ev = audit(
            x, k=k, ann_neighbors=ann, near_dup_cos=near_dup_cos, sample=sample or None,
            queries=q, clusters=clusters, thresholds=thresholds, sample_repeats=sample_repeats,
            query_baseline=qbase, qrels=qrels, ranked=ranked,
        )
        if src is not None:
            annotate(ev, src)
        ev = ev.to_dict()
        audit_id = _audit_id(ev)
        _remember(audit_id, ev)
        if base is not None:
            ev["baseline_diff"] = diff_evidence(ev, base, tolerances=tolerances)
        if note:
            ev["query_reference_note"] = note
    except _USER_ERRORS as e:
        raise _user_error(e) from e
    summary = _summary(ev)
    if not verbose:
        ev = {**ev, "findings": [dict(f) for f in ev["findings"]]}
        ev.pop("thresholds", None)
        for f in ev["findings"]:
            if f["verdict"] == "ok":
                f["details"] = {}
                f["likely_causes"], f["suggested_action"], f["verify_by"] = [], "", ""
                f["explanation"] = ""
    return {"summary": summary, "verdict": ev["verdict"], "audit_id": audit_id, **ev}


@mcp.tool(**_TOOL_KW)
def sweep_index(
    path: str,
    k: int = 10,
    queries_path: str | None = None,
    target_recall: float = 0.95,
    device: str = "auto",
    margin: float = 0.02,
    timing_repeats: int = 3,
    n_queries: int = 1000,
    latency_queries: int = 200,
) -> dict:
    """Sweep HNSW (M, ef_search) and IVF (nprobe) parameters on an embedding file and report
    recall@k against exact k-NN, build time, queries per second, single-query latency
    percentiles (p50/p95/p99 ms) and the serialized index size for each setting, plus the
    fastest configuration that reaches `target_recall`. The recommendation carries
    `margin` (recall minus target) and is marked `tight` with a `safer_alternative` when the
    margin is under `margin`, because recall at fixed parameters falls as the corpus grows;
    relay `caveats` to the user. qps is the median of `timing_repeats` runs. Needs the faiss
    extra (`pip install "vecscope[index]"`). Read-only; timings are relative to this machine
    and process, not a serving latency.
    `device` is "auto" (GPU when the faiss build has one), "cpu" or "gpu"; HNSW is CPU-only.
    `n_queries` caps the queries timed (sampled from the corpus, or from `queries_path`);
    pass real query traffic as `queries_path` whenever it exists, because the best setting
    on corpus-drawn queries can differ. `latency_queries` are searched one at a time per
    setting for the percentiles (0 = skip).
    """
    from vecscope.sweep import sweep

    try:
        x = _vectors(path)
        q = _vectors(queries_path, "queries_path") if queries_path else None
        res = sweep(x, k=k, queries=q, n_queries=n_queries, target_recall=target_recall,
                    device=device, margin=margin, timing_repeats=timing_repeats,
                    latency_queries=latency_queries).to_dict()
    except (*_USER_ERRORS, RuntimeError) as e:
        raise _user_error(e) from e
    r = res["recommendation"]
    params = ", ".join(f"{a}={b}" for a, b in r["params"].items())
    lat = (f", p95 {r['latency_ms_p95']:.3f} ms single-query"
           if r.get("latency_ms_p95") is not None else "")
    size = f", index {r['index_bytes'] / 2**20:.1f} MiB" if r.get("index_bytes") is not None else ""
    res["summary"] = (f"recommend {r['index']} ({params}): recall@{k}={r['recall_at_k']:.3f} "
                      f"(margin {r['margin']:+.3f}{', tight' if r.get('tight') else ''}), "
                      f"{r['queries_per_second']:.0f} qps on {res['device']}{lat}{size}; "
                      f"{r['reason']}")
    return res


@mcp.tool(**_TOOL_KW)
def compare_embeddings(
    path_a: str,
    path_b: str,
    k: int = 10,
    near_dup_cos: float = 0.99,
    sample: int = 100_000,
    neighbors_a_path: str | None = None,
    neighbors_b_path: str | None = None,
) -> dict:
    """Compare two embedding files whose rows are the same items in the same order (two
    models, or one model before and after a transform). Reports how often the nearest
    neighbor and the top-k lists agree, the near-duplicate share in each space and in both,
    whether both spaces pick the same duplicate partner, and the rows that disagree most.
    No verdicts: duplicates flagged by both spaces with the same partner belong to the items;
    duplicates flagged by one space only point at that model. Above `sample` rows a seeded
    subset is compared (0 = all rows).

    Read `neighbor_scope` before quoting the agreement: "sample" means the neighbors were
    recomputed inside the sampled rows, so the numbers describe the sample's own neighbor
    graph and are usually higher than the full corpus's. To compare full-corpus rankings,
    pass `neighbors_a_path` and `neighbors_b_path` (both): (n x >= k) neighbor ids per row
    computed on the full corpus in each space, exact or from the production index; the
    rows may still be sampled but their neighbors are not (`neighbor_scope` = "full",
    `neighbor_source` = "supplied").
    """
    from vecscope.compare import compare

    try:
        if (neighbors_a_path is None) != (neighbors_b_path is None):
            raise ValueError("neighbors_a_path and neighbors_b_path go together")
        na = _neighbors(neighbors_a_path, "neighbors_a_path")
        nb = _neighbors(neighbors_b_path, "neighbors_b_path")
        res = compare(_vectors(path_a, "path_a"),
                      _vectors(path_b, "path_b"), k=k, near_dup_cos=near_dup_cos,
                      sample=sample or None, neighbors_a=na, neighbors_b=nb).to_dict()
    except _USER_ERRORS as e:
        raise _user_error(e) from e
    return {"summary": res["explanation"], **res}


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


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="vecscope-mcp", description="vecscope MCP server (stdio)")
    p.add_argument("--allow-dir", action="append", default=[], metavar="DIR",
                   help="only read files under this directory (repeatable); adds to "
                        f"{ALLOWED_DIRS_ENV}. Without it only the current directory is readable")
    p.add_argument("--allow-any", action="store_true",
                   help="read any file the process can read (no directory restriction); "
                        f"same as {ALLOW_ANY_ENV}=1. Ignored when --allow-dir is given")
    p.add_argument("--allow-network", action="store_true",
                   help="let the tools read network vector stores (qdrant://, chroma://, "
                        f"milvus://, weaviate://); same as {ALLOW_NETWORK_ENV}=1")
    p.add_argument("--max-read-mb", type=float, default=None, metavar="MB",
                   help="refuse files larger than this and stop reading a store past it "
                        f"(default {DEFAULT_MAX_READ_MB}; 0 = no limit); same as {MAX_READ_ENV}")
    args = p.parse_args(argv)
    if args.max_read_mb is not None:
        if args.max_read_mb < 0:
            p.error("--max-read-mb must be 0 (no limit) or positive")
        os.environ[MAX_READ_ENV] = str(args.max_read_mb)
    if args.allow_network:
        os.environ[ALLOW_NETWORK_ENV] = "1"
    if args.allow_any:
        os.environ[ALLOW_ANY_ENV] = "1"
    if args.allow_dir:
        for d in args.allow_dir:
            if not Path(d).is_dir():
                p.error(f"--allow-dir {d}: not a directory")
        current = os.environ.get(ALLOWED_DIRS_ENV, "")
        os.environ[ALLOWED_DIRS_ENV] = os.pathsep.join(
            [*([current] if current else []), *(str(Path(d).resolve()) for d in args.allow_dir)])
    roots = allowed_dirs()
    limit = max_read_bytes()
    print(("vecscope-mcp: reading files under " + ", ".join(map(str, roots)) if roots else
           "vecscope-mcp: reading any file this process can read (--allow-any)")
          + (f"; at most {limit / 2**20:.0f} MiB per file or store" if limit else "; no size limit"),
          file=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
