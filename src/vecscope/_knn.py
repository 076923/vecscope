"""Exact k-nearest-neighbor helpers (NumPy only, blocked for memory)."""

from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def exact_knn(
    x: np.ndarray, k: int, metric: str = "cosine", block: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, distances) of the k nearest neighbors, excluding self.

    distances are cosine distance (1 - cos) or squared euclidean.
    """
    n = x.shape[0]
    k = min(k, n - 1)
    if metric == "cosine":
        q = normalize(x)
    else:  # "dot" (negative inner product) or "euclidean"
        q = np.asarray(x, dtype=np.float32)
    sq = (q * q).sum(axis=1)
    idx = np.empty((n, k), dtype=np.int64)
    dist = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, block):
        e = min(s + block, n)
        if metric == "cosine":
            d = 1.0 - q[s:e] @ q.T
        elif metric == "dot":
            d = -(q[s:e] @ q.T)
        else:
            d = sq[s:e, None] - 2.0 * (q[s:e] @ q.T) + sq[None, :]
        rows = np.arange(e - s)
        d[rows, np.arange(s, e)] = np.inf  # drop self
        part = np.argpartition(d, k, axis=1)[:, :k]
        pd = np.take_along_axis(d, part, axis=1)
        order = np.argsort(pd, axis=1)
        idx[s:e] = np.take_along_axis(part, order, axis=1)
        dist[s:e] = np.take_along_axis(pd, order, axis=1)
    return idx, dist
