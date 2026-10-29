"""Exact k-nearest-neighbor helpers (NumPy only, blocked for memory)."""

from __future__ import annotations

import numpy as np

MAX_ABS = 1e16  # above this, squares and dot products overflow float32 even at small dim


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x.astype(np.float64), axis=1, keepdims=True)  # no float32 overflow
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


def exact_copy_groups(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group rows that are the same vector: ``(first, inverse, counts)`` as ``np.unique`` gives.

    Rows are compared by direction (the unit vector, rounded to six decimals) and by length
    relative to the median non-zero row norm (log10, also rounded), so "same" scales with the
    data and a short row with its own direction is not confused with other short rows. Zero
    rows form one group. Byte-identical rows always land in one group; rows that differ by
    float32 noise usually do, but a value that sits on a rounding boundary can split a pair
    (about 2% of 1-ulp pairs, a third of pairs with 1e-7 relative noise). Such pairs still
    count as near-duplicates through the cosine cutoff; only the exact-copy statistics
    (``exact_duplicate_ratio``, ``largest_exact_duplicate_groups``) undercount them."""
    x32 = np.asarray(x, dtype=np.float32)
    n, d = x32.shape
    norms = np.sqrt(np.einsum("ij,ij->i", x32, x32, dtype=np.float64))
    scale = float(np.median(norms[norms > 0])) if (norms > 0).any() else 1.0
    # one float32 key row per vector: rounded direction, then rounded relative length; the
    # rows are compared as raw bytes, so memory stays at one extra copy of the matrix
    key = np.empty((n, d + 1), dtype=np.float32)
    np.divide(x32, np.where(norms > 0, norms, 1.0)[:, None].astype(np.float32), out=key[:, :d])
    np.round(key[:, :d], 6, out=key[:, :d])
    with np.errstate(divide="ignore"):  # zero rows: -inf, one group of their own
        key[:, d] = np.round(np.log10(norms / scale), 6)  # relative length, 2e-6 resolution
    key += 0.0  # -0.0 and 0.0 must compare equal byte-wise
    rows = np.ascontiguousarray(key).view(np.dtype((np.void, key.dtype.itemsize * (d + 1)))).ravel()
    _, first, inverse, counts = np.unique(rows, return_index=True, return_inverse=True,
                                          return_counts=True)
    return first, np.asarray(inverse).reshape(-1), counts


def check_magnitude(x: np.ndarray, what: str = "embeddings") -> None:
    """Reject values whose squares or dot products would overflow float32 arithmetic."""
    if x.size == 0:
        top = 0.0
    elif x.dtype.kind in "iu":  # np.abs wraps for the most negative int64
        top = float(max(abs(int(x.max())), abs(int(x.min()))))
    else:
        top = float(np.abs(x).max())
    if top > MAX_ABS:
        raise ValueError(
            f"{what} contain values up to {top:.3g}; above {MAX_ABS:g} float32 dot products "
            "overflow. Rescale the vectors (e.g. divide by a constant) and audit again"
        )


def resolve_sample(sample, k: int) -> int | None:
    """Shared meaning of the ``sample`` option: None or 0 = every row, else a size above k."""
    if sample is None:
        return None
    if isinstance(sample, bool) or not isinstance(sample, (int, np.integer)) or sample < 0:
        raise ValueError(f"sample must be None or a non-negative integer, got {sample!r}")
    sample = None if sample == 0 else int(sample)
    if sample is not None and sample <= k:
        raise ValueError(f"sample={sample} must exceed k={k} (0 audits every row)")
    return sample


def row_overlap(a: np.ndarray, b: np.ndarray, block: int = 4096) -> np.ndarray:
    """Per-row share of neighbor ids two (n x k) lists have in common (k = the smaller)."""
    k = min(a.shape[1], b.shape[1])
    a = np.asarray(a)[:, :k]
    b = np.asarray(b)[:, :k]
    out = np.empty(a.shape[0], dtype=np.float64)
    for s in range(0, a.shape[0], block):
        e = min(s + block, a.shape[0])
        out[s:e] = (a[s:e, :, None] == b[s:e, None, :]).any(axis=2).sum(axis=1) / k
    return out


def auto_block(n: int, budget_bytes: int = 1_000_000_000) -> int:
    """Rows per block so that the block's distance matrix and its argpartition index
    (4 + 8 bytes per entry) stay within ``budget_bytes``: 2048 rows up to ~40k rows,
    about 800 at 100k rows, never below 128."""
    return int(max(128, min(2048, budget_bytes // (12 * max(n, 1)))))


def exact_knn(
    x: np.ndarray, k: int, metric: str = "cosine", block: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, distances) of the k nearest neighbors, excluding self.

    distances are cosine distance (1 - cos) or squared euclidean. ``block`` rows are
    processed at a time; the default keeps peak memory near 1 GB whatever ``n`` is. The
    neighbor set does not depend on it except among exactly tied distances, where the
    selection is NumPy's and may differ between NumPy builds (not between runs on one build).
    """
    n = x.shape[0]
    k = min(k, n - 1)
    if block is None:
        block = auto_block(n)
    if metric == "cosine":
        q = normalize(x)
    else:  # "dot" (negative inner product) or "euclidean"
        q = np.asarray(x, dtype=np.float32)
    sq = (q * q).sum(axis=1) if metric == "euclidean" else None
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
        order = np.argsort(pd, axis=1, kind="stable")
        idx[s:e] = np.take_along_axis(part, order, axis=1)
        dist[s:e] = np.take_along_axis(pd, order, axis=1)
    return idx, dist
