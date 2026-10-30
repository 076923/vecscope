"""Index-side evidence: how do HNSW and IVF parameters trade recall for speed on this corpus?

Requires the optional `faiss-cpu` dependency (`pip install "vecscope[index]"`), or a GPU
build of faiss (`faiss-gpu-cu12`), in which case the IVF index and the exact baseline run on
the GPU automatically. HNSW has no GPU implementation in faiss and always runs on the CPU.
Recall is measured against exact k-NN for a seeded sample of queries drawn from the corpus
(or a supplied query file). Timings are wall-clock on this machine and only meaningful
relative to each other; each setting is timed ``timing_repeats`` times and the median kept.

The recommendation is the fastest setting that reaches ``target_recall``. When its margin
over the target is below ``margin`` the result also names the next safer setting, because
recall at fixed parameters falls as the corpus grows: a setting that just clears the target
on a sample usually misses it on the full corpus.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from vecscope._knn import check_magnitude, normalize
from vecscope.diagnostics.ann_recall import tie_aware_hits
from vecscope.diagnostics.queries import query_knn

HNSW_GRID = {"M": (8, 16, 32), "ef_construction": (100,), "ef_search": (16, 32, 64, 128, 256)}
IVF_NPROBE = (1, 2, 4, 8, 16, 32, 64)
RECALL_MARGIN = 0.02  # below this margin over target_recall the recommendation is marked tight
LATENCY_QUERIES = 200  # queries timed one at a time per setting for the latency percentiles


@dataclass
class SweepPoint:
    index: str
    params: dict
    recall_at_k: float
    build_seconds: float
    queries_per_second: float
    device: str = "cpu"
    latency_ms_p50: float | None = None  # single-query search, median over latency_queries
    latency_ms_p95: float | None = None
    latency_ms_p99: float | None = None
    index_bytes: int | None = None  # serialized index size, the memory the index needs at rest


@dataclass
class SweepResult:
    tool: str
    version: str
    n: int
    dim: int
    k: int
    n_queries: int
    target_recall: float
    points: list[SweepPoint]
    recommendation: dict = field(default_factory=dict)
    device: str = "cpu"
    gpu_name: str | None = None
    caveats: list[str] = field(default_factory=list)
    latency_queries: int = 0  # queries searched one at a time for the latency percentiles

    def to_dict(self) -> dict:
        return asdict(self)


def gpu_available(faiss) -> bool:
    """True when this faiss build has GPU support and at least one GPU is visible."""
    needed = ("StandardGpuResources", "GpuParameterSpace", "index_cpu_to_gpu", "get_num_gpus")
    if not all(hasattr(faiss, name) for name in needed):
        return False  # CPU wheels (notably on macOS) ship partial GPU stubs
    try:
        if faiss.get_num_gpus() <= 0:
            return False
        faiss.StandardGpuResources()  # fails on a stub build or a broken driver
    except (RuntimeError, AttributeError, TypeError):
        return False
    return True


def _resolve_device(faiss, device: str) -> str:
    if device not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be 'auto', 'cpu' or 'gpu'")
    has_gpu = gpu_available(faiss)
    if device == "gpu" and not has_gpu:
        raise RuntimeError(
            "device='gpu' but this faiss build has no GPU support or no GPU is visible; "
            "install a GPU build (e.g. faiss-gpu-cu12) or use device='auto'"
        )
    return "gpu" if (device == "gpu" or (device == "auto" and has_gpu)) else "cpu"


def _drop_self(idx: np.ndarray, qids: np.ndarray | None, k: int) -> np.ndarray:
    """Keep the first k ids per row after removing the query's own id (and faiss -1 fillers)."""
    out = np.full((idx.shape[0], k), -1, dtype=np.int64)
    for i, row in enumerate(idx):
        keep = row[row >= 0]
        if qids is not None:
            keep = keep[keep != qids[i]]
        keep = keep[:k]
        out[i, : keep.size] = keep
    return out


def _recall(approx: np.ndarray, exact: np.ndarray, q: np.ndarray, c: np.ndarray) -> float:
    """Share of the k exact neighbor slots the index filled, counted by distance so that
    tied duplicates are interchangeable while a missed closer neighbor stays a miss."""
    k = exact.shape[1]
    hits = np.empty(exact.shape[0])
    for i, (a, e) in enumerate(zip(approx, exact)):
        e, a = e[e >= 0], np.unique(a[a >= 0])
        if e.size == 0:
            hits[i] = 0.0
            continue
        qi = q[i].astype(np.float64)
        exact_d = np.sort(1.0 - c[e].astype(np.float64) @ qi)
        hits[i] = tie_aware_hits(exact_d, 1.0 - c[a].astype(np.float64) @ qi)
    return float(hits.mean() / k)


def sweep(
    x: np.ndarray,
    k: int = 10,
    queries: np.ndarray | None = None,
    n_queries: int = 1000,
    target_recall: float = 0.95,
    seed: int = 0,
    device: str = "auto",
    timing_repeats: int = 3,
    margin: float = RECALL_MARGIN,
    latency_queries: int = LATENCY_QUERIES,
) -> SweepResult:
    """Sweep HNSW and IVF settings. ``device`` is 'auto' (GPU when available), 'cpu' or 'gpu'.

    ``timing_repeats`` searches are timed per setting and the median qps is reported.
    ``margin`` is the recall headroom over ``target_recall`` below which the recommendation
    is marked tight and a safer alternative is named. ``latency_queries`` queries are also
    searched one at a time per setting for ``latency_ms_p50/p95/p99`` (0 = skip), and every
    point carries ``index_bytes``, the serialized size of its index.
    """
    try:
        import faiss
    except ImportError as e:  # pragma: no cover
        raise ImportError('index sweeps need faiss: pip install "vecscope[index]"') from e
    from vecscope import __version__

    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if n_queries < 1:
        raise ValueError("n_queries must be at least 1")
    if not (0.0 < target_recall <= 1.0):
        raise ValueError("target_recall must be in (0, 1]")
    if timing_repeats < 1:
        raise ValueError("timing_repeats must be at least 1")
    if not (0.0 <= margin < 1.0):
        raise ValueError("margin must be in [0, 1)")
    if isinstance(latency_queries, bool) or not isinstance(latency_queries, (int, np.integer)) \
            or latency_queries < 0:
        raise ValueError("latency_queries must be a non-negative integer (0 = skip)")
    x = np.asarray(x)
    if x.dtype.kind not in "iufb":
        raise ValueError(f"embeddings must be real numbers, got dtype {x.dtype}")
    if x.ndim != 2 or x.shape[0] <= k or x.shape[1] < 1:
        raise ValueError(f"need a 2-D matrix with more than k={k} rows and at least one column, "
                         f"got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("embeddings contain NaN or Inf")
    check_magnitude(x)
    x = np.asarray(x, dtype=np.float32)
    if queries is not None:
        queries = np.asarray(queries)
        if queries.dtype.kind not in "iufb":
            raise ValueError(f"queries must be real numbers, got dtype {queries.dtype}")
        if queries.ndim != 2 or queries.shape[1] != x.shape[1] or queries.shape[0] == 0:
            raise ValueError(f"queries must be shaped (m, {x.shape[1]}) with m >= 1, got "
                             f"{queries.shape}")
        if not np.isfinite(queries).all():
            raise ValueError("queries contain NaN or Inf")
        check_magnitude(queries, "queries")
        queries = np.asarray(queries, dtype=np.float32)
    dev = _resolve_device(faiss, device)
    gpu_res = faiss.StandardGpuResources() if dev == "gpu" else None
    gpu_name = None
    if dev == "gpu":
        try:
            import torch  # optional, only for a readable device name

            gpu_name = torch.cuda.get_device_name(0)
        except (ImportError, RuntimeError, AssertionError):
            gpu_name = "gpu:0"

    def to_device(index):
        return faiss.index_cpu_to_gpu(gpu_res, 0, index) if dev == "gpu" else index

    c = normalize(x)
    n, d = c.shape
    rng = np.random.default_rng(seed)
    qids = None  # row ids of the queries inside the corpus, so self can be removed by id
    if queries is None:
        qids = rng.choice(n, min(n_queries, n), replace=False)
        q = c[qids]
    else:
        q = normalize(queries)
        if q.shape[0] > n_queries:
            q = q[rng.choice(q.shape[0], n_queries, replace=False)]
    kq = min(k + 1, n) if queries is None else min(k, n)
    if dev == "gpu":  # exact baseline on the GPU too
        flat = to_device(faiss.IndexFlatIP(d))
        flat.add(c)
        _, exact_raw = flat.search(q, kq)
    else:
        exact_raw, _ = query_knn(c, q, kq)
    exact_idx = _drop_self(exact_raw, qids, k)
    points: list[SweepPoint] = []

    n_lat = min(latency_queries, q.shape[0])
    lat_q = q[:n_lat]  # the first queries of the (already seeded) set

    def measure(index, name, params, build, index_bytes, device):
        times = []
        for _ in range(timing_repeats):
            t = time.perf_counter()
            _, approx = index.search(q, kq)
            times.append(time.perf_counter() - t)
        qps = q.shape[0] / max(float(np.median(times)), 1e-9)
        p50 = p95 = p99 = None
        if n_lat:
            one = np.empty(n_lat)
            for i in range(n_lat):
                t = time.perf_counter()
                index.search(lat_q[i : i + 1], kq)
                one[i] = time.perf_counter() - t
            one *= 1000.0
            p50, p95, p99 = (float(v) for v in np.quantile(one, (0.5, 0.95, 0.99)))
        recall = _recall(_drop_self(approx, qids, k), exact_idx, q, c)
        return SweepPoint(name, params, recall, build, qps, device, p50, p95, p99, index_bytes)

    def size_of(index) -> int:
        cpu_index = faiss.index_gpu_to_cpu(index) if dev == "gpu" else index
        return int(faiss.serialize_index(cpu_index).nbytes)

    for m in HNSW_GRID["M"]:
        for efc in HNSW_GRID["ef_construction"]:
            index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = efc
            t = time.perf_counter()
            index.add(c)
            build = time.perf_counter() - t
            nbytes = size_of(index)
            for efs in HNSW_GRID["ef_search"]:
                index.hnsw.efSearch = efs
                points.append(measure(index, "hnsw",
                                      {"M": m, "ef_construction": efc, "ef_search": efs},
                                      build, nbytes, "cpu"))  # HNSW: CPU only

    nlist = int(max(1, min(4 * np.sqrt(n), n // 39 or 1)))
    quantizer = faiss.IndexFlatIP(d)
    index = to_device(faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT))
    t = time.perf_counter()
    index.train(c)
    index.add(c)
    build = time.perf_counter() - t
    nbytes = size_of(index)
    for nprobe in IVF_NPROBE:
        if nprobe > nlist:
            break
        if dev == "gpu":
            faiss.GpuParameterSpace().set_index_parameter(index, "nprobe", nprobe)
        else:
            index.nprobe = nprobe
        points.append(measure(index, "ivf", {"nlist": nlist, "nprobe": nprobe}, build, nbytes, dev))

    rec = _recommend(points, k, target_recall, margin)
    caveats = [
        (f"recall is measured on {q.shape[0]} queries drawn from the corpus itself"
         if queries is None else
         f"recall is measured on {q.shape[0]} supplied queries") + "; real query traffic "
        "can differ, pass it as queries when available",
        ("HNSW recall at fixed parameters falls as the corpus grows, so a sweep on a sample is "
         "optimistic for HNSW; IVF nlist is set from the row count (4*sqrt(n)), so its recall "
         "at a given nprobe can move either way with size. Re-run at production size before "
         "fixing parameters"),
        (f"queries_per_second is the median of {timing_repeats} runs on this machine with one "
         "query batch; it ranks settings against each other and is not a serving throughput"),
        (f"latency_ms_p50/p95/p99 time {n_lat} queries one at a time in this process, with no "
         "network, concurrency or serving stack; compare settings with them, do not quote "
         "them as service latency" if n_lat else
         "latency percentiles were skipped (latency_queries=0)"),
        ("index_bytes is the serialized index (vectors plus graph or lists); the process "
         "needs more while building, and a product-quantized index would be smaller"),
    ]
    return SweepResult("vecscope", __version__, int(n), int(d), int(k), int(q.shape[0]),
                       target_recall, points, rec, device=dev, gpu_name=gpu_name,
                       caveats=caveats, latency_queries=int(n_lat))


def _recommend(points: list[SweepPoint], k: int, target_recall: float, margin: float) -> dict:
    """Fastest point reaching the target; marked tight (with a safer alternative when one
    exists) if its headroom over the target is below ``margin``."""

    def as_rec(p):
        return {"index": p.index, "params": p.params, "recall_at_k": p.recall_at_k,
                "queries_per_second": p.queries_per_second, "latency_ms_p95": p.latency_ms_p95,
                "index_bytes": p.index_bytes}

    ok = [p for p in points if p.recall_at_k >= target_recall]
    if not ok:
        best = max(points, key=lambda p: p.recall_at_k)
        return {**as_rec(best), "margin": best.recall_at_k - target_recall, "tight": True,
                "reason": f"no configuration reached recall@{k} >= {target_recall}; "
                          "this is the best one. Expect a hard corpus (see audit findings)."}
    best = max(ok, key=lambda p: p.queries_per_second)
    rec = {**as_rec(best), "margin": best.recall_at_k - target_recall, "tight": False,
           "reason": f"fastest configuration with recall@{k} >= {target_recall}"}
    if rec["margin"] >= margin:
        return rec
    rec["tight"] = True
    safer = [p for p in ok if p.recall_at_k - target_recall >= margin]
    if safer:
        alt = max(safer, key=lambda p: p.queries_per_second)
        rec["safer_alternative"] = as_rec(alt)
        rec["reason"] += (f"; margin {rec['margin']:.3f} over the target is below {margin}, so a "
                          "corpus larger than this one will likely miss the target. The fastest "
                          f"setting with margin >= {margin} is {alt.index} "
                          f"({', '.join(f'{a}={b}' for a, b in alt.params.items())}) at recall "
                          f"{alt.recall_at_k:.3f}, {alt.queries_per_second:.0f} qps.")
    else:
        rec["reason"] += (f"; margin {rec['margin']:.3f} over the target is below {margin} and "
                          "no setting in the grid has more headroom.")
    return rec


def render_sweep(res: SweepResult) -> str:
    head = (f"vecscope {res.version}  sweep  n={res.n} dim={res.dim} k={res.k} "
            f"queries={res.n_queries} target_recall={res.target_recall}  device={res.device}")
    if res.gpu_name:
        head += f" ({res.gpu_name}; HNSW stays on CPU)"
    lines = [head]
    lines.append(f"  {'index':<6}{'params':<44}{'recall':>8}{'build s':>9}{'qps':>10}"
                 f"{'p50 ms':>8}{'p95 ms':>8}{'MiB':>8}  dev")

    def ms(v):
        return f"{v:>8.3f}" if v is not None else f"{'-':>8}"

    for p in res.points:
        params = ", ".join(f"{a}={b}" for a, b in p.params.items())
        mark = "*" if p.recall_at_k >= res.target_recall else " "
        mib = f"{p.index_bytes / 2**20:>8.1f}" if p.index_bytes is not None else f"{'-':>8}"
        lines.append(f" {mark}{p.index:<6}{params:<44}{p.recall_at_k:>8.3f}"
                     f"{p.build_seconds:>9.2f}{p.queries_per_second:>10.0f}"
                     f"{ms(p.latency_ms_p50)}{ms(p.latency_ms_p95)}{mib}  {p.device}")
    r = res.recommendation
    params = ", ".join(f"{a}={b}" for a, b in r["params"].items())
    extra = ""
    if r.get("latency_ms_p95") is not None:
        extra += f"  p95={r['latency_ms_p95']:.3f}ms"
    if r.get("index_bytes") is not None:
        extra += f"  index={r['index_bytes'] / 2**20:.1f}MiB"
    lines.append(f"  recommendation: {r['index']} ({params})  recall={r['recall_at_k']:.3f}  "
                 f"qps={r['queries_per_second']:.0f}{extra}  margin={r['margin']:+.3f}"
                 f"{'  TIGHT' if r.get('tight') else ''}  ({r['reason']})")
    for c in res.caveats:
        lines.append(f"  note: {c}")
    return "\n".join(lines)
