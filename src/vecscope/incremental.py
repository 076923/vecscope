"""Incremental exact k-NN: keep the neighbor lists of an audited corpus and, when rows are
appended or changed in place, compute only what those rows change.

A full audit is quadratic in the rows. When a corpus grows by appending (new documents at
the end), the exact k-NN of n + m rows needs only the new rows against all rows
(m x (n + m)) and the old rows against the new ones (n x m): the old neighbor lists stay
valid except where a new row is closer.

Rows re-embedded in place (same position, new vector) are handled the same way, with one
more step: a row whose cached list contains a changed row has lost a neighbor whose distance
is no longer known, so it is recomputed in full. Every other old row keeps its list and only
looks at the changed and appended rows.

The cache stores the lists with a SHA-256 fingerprint of all rows and a 16-byte BLAKE2b
digest per row. It is reused for the same rows, and extended or updated when the rows to
recompute (changed rows, the rows that listed one, appended rows) are at most half of all
rows. Past that an update costs about what a rebuild does, so it is rebuilt; with k = 10 a
change spreads to roughly ten times as many rows, so that point is near 5% of the rows
changed at random. It is also rebuilt when rows were removed or the dimension changed.

The result equals a full ``exact_knn`` except among exactly tied distances (duplicates),
where either side may pick a different member of the tie; the findings computed from it
then agree to float32 rounding.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from vecscope._knn import auto_block, exact_knn, normalize

CACHE_VERSION = 2
_MAX_RECOMPUTE_SHARE = 0.5


def fingerprint(x: np.ndarray) -> str:
    """SHA-256 of the rows as float32, the dtype every audit runs on."""
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()


def row_digests(x: np.ndarray) -> np.ndarray:
    """A 16-byte BLAKE2b digest of each row as float32, to find the rows that changed."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    return np.asarray([hashlib.blake2b(r.tobytes(), digest_size=16).digest() for r in x],
                      dtype="S16")


def _prepared(x: np.ndarray, metric: str) -> np.ndarray:
    return normalize(x) if metric == "cosine" else np.asarray(x, dtype=np.float32)


def _dist(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    return 1.0 - a @ b.T if metric == "cosine" else -(a @ b.T)


def extend_knn(x: np.ndarray, idx: np.ndarray, dist: np.ndarray, n_old: int, k: int,
               metric: str = "cosine", block: int | None = None):
    """Exact k-NN of all rows of ``x`` from the lists ``idx``/``dist`` of its first ``n_old``
    rows. Rows from ``n_old`` on are new; ``metric`` is "cosine" or "dot" as in exact_knn."""
    return refresh_knn(x, idx, dist, n_old, (), k, metric, block)


def refresh_knn(x: np.ndarray, idx: np.ndarray, dist: np.ndarray, n_old: int, changed, k: int,
                metric: str = "cosine", block: int | None = None):
    """Exact k-NN of all rows of ``x`` from the lists of its first ``n_old`` rows, where the
    rows at positions ``changed`` (all below ``n_old``) hold new vectors and rows from
    ``n_old`` on are appended."""
    n = x.shape[0]
    k = min(k, n - 1)
    changed = np.unique(np.asarray(changed, dtype=np.int64))
    if n_old >= n and changed.size == 0:
        return idx[:, :k], dist[:, :k]
    if idx.shape[0] != n_old or idx.shape[1] < k:
        raise ValueError("cached neighbor lists do not cover the cached rows with k neighbors")
    if changed.size and (changed[0] < 0 or changed[-1] >= n_old):
        raise ValueError("changed rows must be positions among the cached rows")
    q = _prepared(x, metric)
    if block is None:
        block = auto_block(n)
    is_changed = np.zeros(n_old, dtype=bool)
    is_changed[changed] = True
    # a changed row, or a row whose list holds one, is recomputed against every row
    lost = is_changed | is_changed[idx[:, :k]].any(axis=1)
    full = np.concatenate([np.flatnonzero(lost), np.arange(n_old, n)])
    keep = np.flatnonzero(~lost)
    dirty = np.concatenate([changed, np.arange(n_old, n)])
    out_idx = np.empty((n, k), dtype=np.int64)
    out_dist = np.empty((n, k), dtype=np.float32)
    # kept rows: merge the cached list with the k nearest dirty rows; on a tie the cached row
    # stays first, as a stable sort over [cached, dirty] keeps it. Sorting all dirty
    # distances per row would cost more than the distances themselves, so they are cut to k
    m = dirty.size
    kk = min(k, m)
    for s in range(0, keep.size, block):
        rows = keep[s:s + block]
        if m == 0:
            out_idx[rows], out_dist[rows] = idx[rows, :k], dist[rows, :k]
            continue
        d_new = _dist(q[rows], q[dirty], metric).astype(np.float32)
        part = (np.argpartition(d_new, kk - 1, axis=1)[:, :kk] if m > kk
                else np.broadcast_to(np.arange(m), (rows.size, m)))
        pd = np.take_along_axis(d_new, part, axis=1)
        ids = dirty[part]
        within = np.lexsort((ids, pd), axis=1)  # by distance, then row id
        cand_d = np.concatenate([dist[rows, :k], np.take_along_axis(pd, within, axis=1)], axis=1)
        cand_i = np.concatenate([idx[rows, :k], np.take_along_axis(ids, within, axis=1)], axis=1)
        order = np.argsort(cand_d, axis=1, kind="stable")[:, :k]
        out_idx[rows] = np.take_along_axis(cand_i, order, axis=1)
        out_dist[rows] = np.take_along_axis(cand_d, order, axis=1)
    # recomputed rows: against every row, self excluded
    for s in range(0, full.size, block):
        rows = full[s:s + block]
        d = _dist(q[rows], q, metric)
        d[np.arange(rows.size), rows] = np.inf
        part = np.argpartition(d, k, axis=1)[:, :k]
        pd = np.take_along_axis(d, part, axis=1)
        order = np.argsort(pd, axis=1, kind="stable")
        out_idx[rows] = np.take_along_axis(part, order, axis=1)
        out_dist[rows] = np.take_along_axis(pd, order, axis=1)
    return out_idx, out_dist


def cached_knn(x: np.ndarray, k: int, path: str | Path, metrics: tuple[str, ...] = ("cosine",)):
    """Exact k-NN of ``x`` for each metric, read from or written to the cache at ``path``.

    Returns ``({metric: (idx, dist)}, info)`` where ``info["status"]`` is "reused" (same rows),
    "extended" (the cached rows are an unchanged prefix of ``x``; ``info["new_rows"]`` were
    added), "updated" (``info["changed_rows"]`` of the cached rows hold new vectors, and rows
    may have been appended) or "created" (no usable cache: no file, an older cache format,
    rows removed, changes that would recompute more than half the rows, other k or dim;
    ``info["reason"]`` says which). ``info["recomputed_rows"]`` counts the rows whose lists
    were computed in full. The file is rewritten unless it was reused."""
    p = Path(path)
    x = np.asarray(x, dtype=np.float32)
    n, dim = x.shape
    k_eff = min(k, n - 1)
    cache, reason = None, "no cache file"
    if p.exists():
        try:
            # read everything and close the file: Windows cannot replace an open file
            with np.load(p, allow_pickle=False) as z:
                cache = {key: z[key] for key in z.files}
        except (OSError, ValueError, EOFError) as e:
            raise ValueError(f"{p}: not a readable vecscope k-NN cache ({e})") from None
        version = int(cache.get("version", -1))
        if version == 1:  # no row digests: rebuilt once in the current format
            cache, reason = None, "older cache format"
        elif version != CACHE_VERSION:
            raise ValueError(f"{p}: k-NN cache version {cache.get('version')} is not "
                             f"{CACHE_VERSION}; delete it to rebuild")
    digests = row_digests(x)
    status, n_old, changed = "created", 0, np.empty(0, dtype=np.int64)
    if cache is not None:
        c_n, c_dim, c_k = int(cache["n"]), int(cache["dim"]), int(cache["k"])
        if c_dim != dim or c_k < k_eff:
            reason = "other dim" if c_dim != dim else "fewer neighbors cached than k"
        elif not 1 < c_n <= n:
            reason = "rows were removed"
        else:
            changed = np.flatnonzero(cache["row_digests"] != digests[:c_n])
            is_changed = np.zeros(c_n, dtype=bool)
            is_changed[changed] = True
            lost = is_changed.copy()
            for metric in metrics:
                if f"{metric}_idx" in cache:
                    lost |= is_changed[cache[f"{metric}_idx"][:, :k_eff]].any(axis=1)
            recompute = int(lost.sum()) + n - c_n
            if changed.size and recompute > _MAX_RECOMPUTE_SHARE * n:
                reason = (f"{changed.size} changed rows would recompute {recompute} of {n} "
                          "rows, as costly as a rebuild")
                changed = np.empty(0, dtype=np.int64)
            else:
                status = ("updated" if changed.size else
                          "reused" if c_n == n else "extended")
                n_old, reason = c_n, None
    out = {}
    recomputed = 0 if status == "reused" else n if status == "created" else recompute
    for metric in metrics:
        key_i, key_d = f"{metric}_idx", f"{metric}_dist"
        if status != "created" and key_i in cache:
            idx, dist = cache[key_i], cache[key_d]
            if status == "reused":
                out[metric] = (idx[:, :k_eff], dist[:, :k_eff])
            else:
                out[metric] = refresh_knn(x, idx, dist, n_old, changed, k_eff, metric)
        else:
            out[metric] = exact_knn(x, k_eff, metric=metric)
    if status != "reused" or any(f"{m}_idx" not in cache for m in metrics):
        arrays = {"version": CACHE_VERSION, "n": n, "dim": dim, "k": k_eff,
                  "fingerprint": fingerprint(x), "row_digests": digests}
        for metric, (i, d) in out.items():
            arrays[f"{metric}_idx"], arrays[f"{metric}_dist"] = i, d
        tmp = p.with_name(p.name + ".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(p)
    info = {"path": str(p), "status": status, "cached_rows": n_old,
            "new_rows": n - n_old if status in ("extended", "updated") else
            (0 if status == "reused" else n),
            "changed_rows": int(changed.size), "recomputed_rows": recomputed,
            "metrics": list(metrics)}
    if status == "created":
        info["reason"] = reason
    return out, info


def cached_knn_for_audit(x: np.ndarray, k: int, path: str | Path):
    """``cached_knn`` with the metrics an audit needs: cosine always, and inner product when
    the norms vary enough for metric_mismatch to compare the two (the audit's own rule)."""
    from vecscope.diagnostics import norms

    metrics = ("cosine",) if norms(np.asarray(x, dtype=np.float32))["norm_cv"] < 1e-3 else (
        "cosine", "dot")
    return cached_knn(x, k, path, metrics)
