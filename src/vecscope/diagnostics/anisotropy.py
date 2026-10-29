"""Anisotropy: embeddings crowd into a narrow cone, so every pair looks similar.

Evidence: mean cosine of random pairs (0 is isotropic), the norm of the mean unit vector
(the shared direction every vector carries), the same cosine after subtracting that mean
(what centering would buy), top principal-component share, effective rank (participation
ratio of the covariance spectrum).
"""

from __future__ import annotations

import numpy as np

from vecscope._knn import normalize


def _mean_pair_cosine(q: np.ndarray) -> float:
    """Exact mean cosine over all ordered pairs of distinct rows of a unit-norm matrix:
    (|sum q|^2 - sum |q_i|^2) / (n (n - 1)). No pair sampling, so the value does not depend on
    row order or on a random draw."""
    n = q.shape[0]
    if n < 2:
        return 0.0
    s = q.astype(np.float64).sum(axis=0)
    sq = float((q.astype(np.float64) ** 2).sum())  # == n for unit rows, 0 rows excepted
    return float((s @ s - sq) / (n * (n - 1)))


def anisotropy(x: np.ndarray, svd_rows: int = 50000, seed: int = 0) -> dict:
    q = normalize(x)
    n, d = q.shape
    mean_cos = _mean_pair_cosine(q)
    mean_vec = q.mean(axis=0)
    mean_cos_centered = _mean_pair_cosine(normalize(q - mean_vec))

    # the spectrum is taken on a seeded subset of at most svd_rows rows
    sub = q if n <= svd_rows else q[np.random.default_rng(seed).choice(n, svd_rows, replace=False)]
    sub = sub - sub.mean(axis=0, keepdims=True)
    s = np.linalg.svd(sub, compute_uv=False)
    var = s**2
    total = var.sum()
    if total <= 1e-10 * sub.shape[0]:  # constant input (up to float32 rounding)
        var = np.zeros_like(var)
        effective_rank = 0.0
    else:
        var = var / total
        effective_rank = float(1.0 / (var**2).sum())
    return {
        "dim": int(d),
        "mean_random_cosine": mean_cos,
        "mean_vector_norm": float(np.linalg.norm(mean_vec)),
        "mean_random_cosine_after_centering": mean_cos_centered,
        "top1_pc_variance_share": float(var[0]) if var.size else 0.0,
        "effective_rank": effective_rank,
        "max_attainable_rank": int(max(1, min(d, sub.shape[0] - 1))),
        "effective_rank_ratio": effective_rank / float(max(1, min(d, sub.shape[0] - 1))),
    }
