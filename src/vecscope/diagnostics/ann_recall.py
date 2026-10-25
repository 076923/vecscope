"""ANN recall: does the approximate index actually return the true neighbors?

Compare neighbors returned by an ANN index (HNSW, IVF, ...) with exact k-NN.
Evidence: recall@k, per-query worst recall, share of queries below a floor.
"""

from __future__ import annotations

import numpy as np


def ann_recall(exact_knn_idx: np.ndarray, ann_idx: np.ndarray, floor: float = 0.8) -> dict:
    k = min(exact_knn_idx.shape[1], ann_idx.shape[1])
    exact = exact_knn_idx[:, :k]
    approx = ann_idx[:, :k]
    hits = np.array([len(set(e.tolist()) & set(a.tolist())) for e, a in zip(exact, approx)])
    per_query = hits / k
    return {
        "k": int(k),
        "recall_at_k": float(per_query.mean()),
        "recall_p05": float(np.quantile(per_query, 0.05)),
        "min_recall": float(per_query.min()),
        "queries_below_floor_ratio": float((per_query < floor).mean()),
        "floor": floor,
    }
