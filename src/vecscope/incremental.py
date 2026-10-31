"""Incremental exact k-NN: keep the neighbor lists of an audited corpus and, when rows are
appended, compute only what the new rows change.

A full audit is quadratic in the rows. When a corpus grows by appending (new documents at
the end, existing rows unchanged), the exact k-NN of n + m rows needs only the new rows
against all rows (m x (n + m)) and the old rows against the new ones (n x m): the old
neighbor lists stay valid except where a new row is closer. The cache stores the lists with
a SHA-256 fingerprint of the rows they were computed on, so a cache is reused only for the
same rows and extended only when the cached rows are an unchanged prefix of the new input.

The result equals a full ``exact_knn`` except among exactly tied distances (duplicates),
where either side may pick a different member of the tie; the findings computed from it
then agree to float32 rounding.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from vecscope._knn import auto_block, exact_knn, normalize

CACHE_VERSION = 1


def fingerprint(x: np.ndarray) -> str:
    """SHA-256 of the rows as float32, the dtype every audit runs on."""
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()


def _prepared(x: np.ndarray, metric: str) -> np.ndarray:
    return normalize(x) if metric == "cosine" else np.asarray(x, dtype=np.float32)


def _dist(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    return 1.0 - a @ b.T if metric == "cosine" else -(a @ b.T)


def extend_knn(x: np.ndarray, idx: np.ndarray, dist: np.ndarray, n_old: int, k: int,
               metric: str = "cosine", block: int | None = None):
    """Exact k-NN of all rows of ``x`` from the lists ``idx``/``dist`` of its first ``n_old``
    rows. Rows from ``n_old`` on are new; ``metric`` is "cosine" or "dot" as in exact_knn."""
    n = x.shape[0]
    k = min(k, n - 1)
    if n_old >= n:
        return idx[:, :k], dist[:, :k]
    if idx.shape[0] != n_old or idx.shape[1] < k:
        raise ValueError("cached neighbor lists do not cover the cached rows with k neighbors")
    q = _prepared(x, metric)
    new = q[n_old:]
    if block is None:
        block = auto_block(n)
    out_idx = np.empty((n, k), dtype=np.int64)
    out_dist = np.empty((n, k), dtype=np.float32)
    # old rows: merge the cached list with the k nearest new rows; on a tie the cached (lower)
    # row id stays first, as a stable sort over [cached, new] keeps it. Sorting all m new
    # distances per row would cost more than the distances themselves, so they are cut to k
    m = n - n_old
    kk = min(k, m)
    for s in range(0, n_old, block):
        e = min(s + block, n_old)
        d_new = _dist(q[s:e], new, metric).astype(np.float32)
        part = (np.argpartition(d_new, kk - 1, axis=1)[:, :kk] if m > kk
                else np.broadcast_to(np.arange(m), (e - s, m)))
        pd = np.take_along_axis(d_new, part, axis=1)
        within = np.lexsort((part, pd), axis=1)  # by distance, then row id
        cand_d = np.concatenate([dist[s:e, :k], np.take_along_axis(pd, within, axis=1)], axis=1)
        cand_i = np.concatenate([idx[s:e, :k],
                                 np.take_along_axis(part, within, axis=1) + n_old], axis=1)
        order = np.argsort(cand_d, axis=1, kind="stable")[:, :k]
        out_idx[s:e] = np.take_along_axis(cand_i, order, axis=1)
        out_dist[s:e] = np.take_along_axis(cand_d, order, axis=1)
    # new rows: against every row, self excluded
    for s in range(n_old, n, block):
        e = min(s + block, n)
        d = _dist(q[s:e], q, metric)
        d[np.arange(e - s), np.arange(s, e)] = np.inf
        part = np.argpartition(d, k, axis=1)[:, :k]
        pd = np.take_along_axis(d, part, axis=1)
        order = np.argsort(pd, axis=1, kind="stable")
        out_idx[s:e] = np.take_along_axis(part, order, axis=1)
        out_dist[s:e] = np.take_along_axis(pd, order, axis=1)
    return out_idx, out_dist


def cached_knn(x: np.ndarray, k: int, path: str | Path, metrics: tuple[str, ...] = ("cosine",)):
    """Exact k-NN of ``x`` for each metric, read from or written to the cache at ``path``.

    Returns ``({metric: (idx, dist)}, info)`` where ``info["status"]`` is "reused" (same rows),
    "extended" (the cached rows are an unchanged prefix of ``x``; ``info["new_rows"]`` were
    added) or "created" (no usable cache: no file, other rows, other k or dim). The file is
    rewritten unless it was reused as is."""
    p = Path(path)
    x = np.asarray(x, dtype=np.float32)
    n, dim = x.shape
    k_eff = min(k, n - 1)
    cache = None
    if p.exists():
        try:
            # read everything and close the file: Windows cannot replace an open file
            with np.load(p, allow_pickle=False) as z:
                cache = {key: z[key] for key in z.files}
        except (OSError, ValueError, EOFError) as e:
            raise ValueError(f"{p}: not a readable vecscope k-NN cache ({e})") from None
        if int(cache.get("version", -1)) != CACHE_VERSION:
            raise ValueError(f"{p}: k-NN cache version {cache.get('version')} is not "
                             f"{CACHE_VERSION}; delete it to rebuild")
    status, n_old = "created", 0
    if cache is not None:
        c_n, c_dim, c_k = int(cache["n"]), int(cache["dim"]), int(cache["k"])
        same_shape = c_dim == dim and c_k >= k_eff and 1 < c_n <= n
        if same_shape and str(cache["fingerprint"]) == fingerprint(x[:c_n]):
            status = "reused" if c_n == n else "extended"
            n_old = c_n
    out = {}
    for metric in metrics:
        key_i, key_d = f"{metric}_idx", f"{metric}_dist"
        if status != "created" and key_i in cache:
            idx, dist = cache[key_i], cache[key_d]
            if status == "reused":
                out[metric] = (idx[:, :k_eff], dist[:, :k_eff])
            else:
                out[metric] = extend_knn(x, idx, dist, n_old, k_eff, metric)
        else:
            out[metric] = exact_knn(x, k_eff, metric=metric)
    if status != "reused" or any(f"{m}_idx" not in cache for m in metrics):
        arrays = {"version": CACHE_VERSION, "n": n, "dim": dim, "k": k_eff,
                  "fingerprint": fingerprint(x)}
        for metric, (i, d) in out.items():
            arrays[f"{metric}_idx"], arrays[f"{metric}_dist"] = i, d
        tmp = p.with_name(p.name + ".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(p)
    info = {"path": str(p), "status": status, "cached_rows": n_old,
            "new_rows": n - n_old if status == "extended" else (0 if status == "reused" else n),
            "metrics": list(metrics)}
    return out, info


def cached_knn_for_audit(x: np.ndarray, k: int, path: str | Path):
    """``cached_knn`` with the metrics an audit needs: cosine always, and inner product when
    the norms vary enough for metric_mismatch to compare the two (the audit's own rule)."""
    from vecscope.diagnostics import norms

    metrics = ("cosine",) if norms(np.asarray(x, dtype=np.float32))["norm_cv"] < 1e-3 else (
        "cosine", "dot")
    return cached_knn(x, k, path, metrics)
