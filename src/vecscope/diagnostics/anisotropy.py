"""Anisotropy: embeddings crowd into a narrow cone, so every pair looks similar.

Evidence: mean cosine of random pairs (0 is isotropic), the norm of the mean unit vector
(the shared direction every vector carries), the same cosine after subtracting that mean
(what centering would buy), top principal-component share, effective rank (participation
ratio of the covariance spectrum).
"""

from __future__ import annotations

import numpy as np

from vecscope._knn import normalize


def anisotropy(x: np.ndarray, sample: int = 20000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    q = normalize(x)
    n, d = q.shape
    m = min(sample, n)
    a = rng.integers(0, n, m)
    b = rng.integers(0, n, m)
    keep = a != b
    mean_cos = float((q[a[keep]] * q[b[keep]]).sum(axis=1).mean())
    mean_vec = q.mean(axis=0)
    qc = normalize(q - mean_vec)
    mean_cos_centered = float((qc[a[keep]] * qc[b[keep]]).sum(axis=1).mean())

    sub = q[rng.choice(n, min(n, 50000), replace=False)]
    sub = sub - sub.mean(axis=0, keepdims=True)
    s = np.linalg.svd(sub, compute_uv=False)
    var = s**2
    total = var.sum()
    if total <= 0:  # constant input
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
        "effective_rank_ratio": effective_rank / float(max(1, min(d, sub.shape[0] - 1))),
    }
