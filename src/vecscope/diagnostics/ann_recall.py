"""ANN recall: does the approximate index actually return the true neighbors?

Compare neighbors returned by an ANN index (HNSW, IVF, ...) with exact k-NN.
Evidence: recall@k, per-query worst recall, share of queries below a floor.
"""

from __future__ import annotations

import numpy as np


def ann_recall(exact_knn_idx: np.ndarray, ann_idx: np.ndarray, floor: float = 0.8) -> dict:
    """recall@k of ``ann_idx`` against exact neighbors. Negative ids (faiss fillers, or the
    item itself masked out by the caller) are ignored; each row contributes its first k
    valid ids, so an index that returns self first should be queried with k + 1."""
    if ann_idx.shape[0] != exact_knn_idx.shape[0]:
        raise ValueError("ann_idx and exact_knn_idx must have the same number of rows")
    k = min(exact_knn_idx.shape[1], ann_idx.shape[1])
    exact = exact_knn_idx[:, :k]
    hits = np.empty(exact.shape[0])
    for i, (e, a) in enumerate(zip(exact, ann_idx)):
        valid = a[a >= 0][:k]
        hits[i] = np.intersect1d(e, valid).size
    per_query = hits / k
    return {
        "k": int(k),
        "recall_at_k": float(per_query.mean()),
        "recall_p05": float(np.quantile(per_query, 0.05)),
        "min_recall": float(per_query.min()),
        "queries_below_floor_ratio": float((per_query < floor).mean()),
        "floor": floor,
    }
