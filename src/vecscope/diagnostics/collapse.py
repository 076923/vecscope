"""Collapse: many vectors are (near) identical, so retrieval cannot tell them apart.

Evidence: exact duplicate ratio, near-duplicate ratio (cos > threshold), the share of items
whose nearest neighbor is essentially itself, and the structure of the duplicates: how many
groups there are, how large the largest is, and sample pairs. A corpus where 10% of items sit
in pairs (the same image under two product ids) and one where 10% of items share a single
vector (an embedding failure) have the same near_duplicate_ratio but different fixes.
"""

from __future__ import annotations

import numpy as np

from vecscope._knn import exact_knn
from vecscope.diagnostics.geometry import connected_labels


def collapse(
    x: np.ndarray,
    near_dup_cos: float = 0.99,
    knn_dist: np.ndarray | None = None,
    knn_idx: np.ndarray | None = None,
    sample_pairs: int = 10,
) -> dict:
    """``knn_idx`` / ``knn_dist`` may carry several neighbors per row; every neighbor within
    the near-duplicate cutoff links the two rows, so a jittered boilerplate cluster forms one
    group instead of fragmenting into nearest-neighbor pairs."""
    n = x.shape[0]
    if knn_dist is None or knn_idx is None:
        knn_idx, knn_dist = exact_knn(x, 1)
    cos = 1.0 - np.asarray(knn_dist, dtype=np.float64)
    nn_cos = cos[:, 0]
    nn = knn_idx[:, 0]
    exact = float((nn_cos >= 1.0 - 1e-6).mean())
    flagged = nn_cos >= near_dup_cos
    near = float(flagged.mean())

    xr = np.asarray(x, dtype=np.float32).round(6)
    _, first, inverse = np.unique(xr, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    exact_sizes = np.bincount(inverse)
    exact_groups = [int(s) for s in np.sort(exact_sizes[exact_sizes > 1])[::-1][:5]]

    # identical rows tie for nearest neighbor, so link every exact-duplicate group explicitly
    # as well as every (row, neighbor) pair within the cutoff
    r, c = np.nonzero(cos >= near_dup_cos)
    src = np.concatenate([r, np.arange(n)])
    dst = np.concatenate([knn_idx[r, c], first[inverse]])
    labels = connected_labels(n, src, dst)
    sizes = np.bincount(labels, minlength=n)
    near_groups = [int(s) for s in np.sort(sizes[sizes > 1])[::-1][:5]]

    # sample pairs: one per group, largest groups first, the closest pair of each group
    rows = np.nonzero(flagged)[0]
    pairs = []
    if rows.size:
        order = rows[np.lexsort((-nn_cos[rows], -sizes[labels[rows]]))]
        _, keep = np.unique(labels[order], return_index=True)
        for i in order[np.sort(keep)][:sample_pairs]:
            pairs.append([int(min(i, nn[i])), int(max(i, nn[i])), float(min(nn_cos[i], 1.0))])

    return {
        "n": int(n),
        "exact_duplicate_ratio": 1.0 - first.size / n,
        "near_duplicate_ratio": near,
        "nn_cosine_median": float(np.median(nn_cos)),
        "nn_cosine_p95": float(np.quantile(nn_cos, 0.95)),
        "near_dup_threshold": near_dup_cos,
        "exact_nn_match_ratio": exact,
        "near_duplicate_groups": int((sizes > 1).sum()),
        "largest_near_duplicate_groups": near_groups,
        "largest_exact_duplicate_groups": exact_groups,
        "sample_pairs": pairs,
    }
