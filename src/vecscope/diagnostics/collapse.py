"""Collapse: many vectors are (near) identical, so retrieval cannot tell them apart.

Evidence: exact duplicate ratio, near-duplicate ratio (cos > threshold), and the
share of items whose nearest neighbor is essentially itself.
"""

from __future__ import annotations

import numpy as np

from vecscope._knn import exact_knn


def collapse(
    x: np.ndarray, near_dup_cos: float = 0.99, knn_dist: np.ndarray | None = None
) -> dict:
    n = x.shape[0]
    if knn_dist is None:
        _, knn_dist = exact_knn(x, 1)
    nn_cos = 1.0 - knn_dist[:, 0]
    exact = float((nn_cos >= 1.0 - 1e-6).mean())
    near = float((nn_cos >= near_dup_cos).mean())
    unique_rows = np.unique(np.asarray(x, dtype=np.float32).round(6), axis=0).shape[0]
    return {
        "n": int(n),
        "exact_duplicate_ratio": 1.0 - unique_rows / n,
        "near_duplicate_ratio": near,
        "nn_cosine_median": float(np.median(nn_cos)),
        "nn_cosine_p95": float(np.quantile(nn_cos, 0.95)),
        "near_dup_threshold": near_dup_cos,
        "exact_nn_match_ratio": exact,
    }
