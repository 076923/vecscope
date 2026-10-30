"""Independent references for the exactness tests: float64 neighbors and a strict tie-aware
recall scorer written without looking at vecscope's, so the two can disagree."""

from __future__ import annotations

import numpy as np

TIE_TOL = 1e-6  # float rounding only


def truth(c: np.ndarray, qids: np.ndarray, k: int, block: int = 256):
    """float64 top-k (self excluded) and the float64 distance of every row to each query."""
    c64 = c.astype(np.float64)
    idx = np.empty((qids.size, k), dtype=np.int64)
    dist = np.empty((qids.size, k))
    for s in range(0, qids.size, block):
        q = qids[s:s + block]
        d = 1.0 - c64[q] @ c64.T
        d[np.arange(q.size), q] = np.inf
        part = np.argpartition(d, k, axis=1)[:, :k]
        pd = np.take_along_axis(d, part, axis=1)
        o = np.argsort(pd, axis=1, kind="stable")
        idx[s:s + block] = np.take_along_axis(part, o, axis=1)
        dist[s:s + block] = np.take_along_axis(pd, o, axis=1)
    return idx, dist


def d64(c64: np.ndarray, qids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """float64 cosine distance from each query row to each id (inf for fillers)."""
    out = np.full(ids.shape, np.inf)
    ok = ids >= 0
    rows = np.repeat(qids[:, None], ids.shape[1], axis=1)
    out[ok] = 1.0 - np.einsum("ij,ij->i", c64[rows[ok]], c64[ids[ok]])
    return out


def drop_self(idx: np.ndarray, qids: np.ndarray, k: int) -> np.ndarray:
    out = np.full((idx.shape[0], k), -1, dtype=np.int64)
    for i, row in enumerate(idx):
        keep = row[(row >= 0) & (row != qids[i])][:k]
        out[i, :keep.size] = keep
    return out


def strict_hits(exact_sorted: np.ndarray, got: np.ndarray, tol: float = TIE_TOL) -> int:
    """Tie-aware matching with float-rounding slack only: walk both sorted distance lists
    and pair a returned id with an exact slot when they differ by <= ``tol``. Greedy
    two-pointer matching is maximal for this interval structure."""
    e = np.sort(exact_sorted)
    r = np.sort(got[np.isfinite(got)])
    i = j = hits = 0
    while i < e.size and j < r.size:
        if abs(e[i] - r[j]) <= tol:
            hits += 1
            i += 1
            j += 1
        elif r[j] < e[i]:
            j += 1
        else:
            i += 1
    return hits
