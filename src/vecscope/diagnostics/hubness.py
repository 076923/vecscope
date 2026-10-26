"""Hubness: a few points appear in everyone's k-NN list (hubs), others in nobody's (anti-hubs).

High hubness means the same chunks top every query regardless of relevance.
Evidence: skewness of the k-occurrence distribution N_k, hub share, anti-hub ratio.
"""

from __future__ import annotations

import numpy as np

from vecscope._knn import exact_knn


def hubness(x: np.ndarray, k: int = 10, knn: np.ndarray | None = None) -> dict:
    n = x.shape[0]
    if knn is None:
        knn, _ = exact_knn(x, k)
    k = knn.shape[1]
    nk = np.bincount(knn.ravel(), minlength=n).astype(np.float64)
    mean, std = nk.mean(), nk.std()
    skew = float(((nk - mean) ** 3).mean() / (std**3)) if std > 0 else 0.0
    top = np.sort(nk)[::-1]
    top1pct = max(1, int(np.ceil(n * 0.01)))
    hub_share = float(top[:top1pct].sum() / nk.sum())
    antihub_ratio = float((nk == 0).mean())
    baseline = float(np.exp(-k))  # share with zero hits if neighbors were assigned at random
    return {
        "k": int(k),
        "skewness": skew,
        "max_k_occurrence": int(nk.max()),
        "expected_k_occurrence": float(k),
        "hub_share_top1pct": hub_share,
        "antihub_ratio": antihub_ratio,
        "antihub_baseline": baseline,
        "antihub_excess": max(0.0, antihub_ratio - baseline),
    }
