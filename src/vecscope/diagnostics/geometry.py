"""Geometry diagnostics derived from the shared exact k-NN pass.

- rogue_dimensions: a handful of dimensions hold most of the variance, so cosine is decided
  by them alone (Timkey & van Schijndel, 2021).
- norms: vector lengths vary or vanish, which breaks inner-product retrieval.
- metric_mismatch: cosine top-k and inner-product top-k disagree, so the index metric matters.
- distance_concentration: all distances look alike, the nearest neighbor is barely nearer than
  a random item (relative contrast, Aggarwal et al., 2001).
- outliers: items whose nearest neighbor is unusually far, typically junk or foreign content.
- connectivity: small islands in the k-NN graph that graph-based ANN attaches badly or never reaches.
"""

from __future__ import annotations

import numpy as np


def rogue_dimensions(x: np.ndarray, top: int = 3) -> dict:
    var = np.asarray(x, dtype=np.float64).var(axis=0)
    total = var.sum()
    expected = 1.0 / x.shape[1]
    if total <= 0:  # constant matrix: every dimension is equally (un)informative
        return {"top1_dim_variance_share": 0.0, "top1_dim_variance_ratio": 0.0,
                "top3_dim_variance_share": 0.0, "expected_share": expected,
                "rogue_dims": [0, 1, 2][: x.shape[1]]}
    share = var / total
    order = np.argsort(share)[::-1]
    return {
        "top1_dim_variance_share": float(share[order[0]]),
        "top1_dim_variance_ratio": float(share[order[0]] / expected),
        "top3_dim_variance_share": float(share[order[:top]].sum()),
        "expected_share": expected,
        "rogue_dims": [int(i) for i in order[:top]],
    }


def norms(x: np.ndarray) -> dict:
    n = np.linalg.norm(np.asarray(x, dtype=np.float64), axis=1)
    mean = n.mean()
    return {
        "norm_cv": float(n.std() / mean) if mean > 0 else 0.0,
        "zero_norm_ratio": float((n == 0).mean()),
        "norm_min": float(n.min()),
        "norm_median": float(np.median(n)),
        "norm_max": float(n.max()),
    }


def metric_mismatch(cos_knn: np.ndarray, dot_knn: np.ndarray) -> dict:
    k = min(cos_knn.shape[1], dot_knn.shape[1])
    a = np.sort(cos_knn[:, :k], axis=1)
    b = np.sort(dot_knn[:, :k], axis=1)
    overlap = np.array([np.intersect1d(r, s, assume_unique=True).size for r, s in zip(a, b)]) / k
    return {
        "k": int(k),
        "topk_overlap": float(overlap.mean()),
        "queries_fully_agreeing_ratio": float((overlap == 1.0).mean()),
        "min_overlap": float(overlap.min()),
    }


def distance_concentration(nn_dist: np.ndarray, random_pair_cos: float) -> dict:
    """Relative contrast on cosine distance: mean random distance over median NN distance."""
    nn = np.asarray(nn_dist, dtype=np.float64)
    distinct = nn[nn > 1e-4]  # exact duplicates (float32 noise ~1e-7) would pin the median at 0
    median_nn = float(np.median(distinct)) if distinct.size else 0.0
    mean_random = float(max(1.0 - random_pair_cos, 1e-9))
    return {
        "relative_contrast": mean_random / max(median_nn, 1e-6),
        "median_nn_distance": median_nn,
        "mean_random_distance": mean_random,
    }


def outliers(nn_dist: np.ndarray, factor: float = 3.0, sample_ids: int = 10) -> dict:
    nn = np.asarray(nn_dist, dtype=np.float64)
    distinct = nn[nn > 1e-4] if (nn > 1e-4).any() else nn
    median = float(np.median(distinct))
    mad = float(np.median(np.abs(distinct - median)))
    cutoff = median + factor * 1.4826 * mad if mad > 0 else median * (1 + factor)
    mask = nn > cutoff
    worst = np.argsort(nn)[::-1][:sample_ids]
    return {
        "outlier_ratio": float(mask.mean()),
        "cutoff_nn_distance": float(cutoff),
        "median_nn_distance": median,
        "sample_ids": [int(i) for i in worst if mask[i]],
    }


def connected_labels(n: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Component label (the smallest member id) for every node of an undirected graph.

    Min-label propagation alone needs as many rounds as the graph diameter, so each round is
    followed by pointer jumping (label = label[label]) until nothing changes. This converges
    for any graph; there is deliberately no iteration cap.
    """
    a = np.concatenate([src, dst])
    b = np.concatenate([dst, src])
    label = np.arange(n)
    while True:
        new = label.copy()
        np.minimum.at(new, a, label[b])
        new = np.minimum(new, label)
        while True:  # pointer jumping: follow labels to their root
            jumped = new[new]
            if np.array_equal(jumped, new):
                break
            new = jumped
        if np.array_equal(new, label):
            return label
        label = new


def connectivity(knn: np.ndarray, min_island_frac: float = 0.005) -> dict:
    """Connected components of the symmetrized k-NN graph.

    Large components are ordinary cluster structure. The retrieval risk is small islands:
    components smaller than max(k + 1, min_island_frac * n), which graph-based ANN tends to
    attach to the wrong region or never reach.
    """
    n, k = knn.shape
    label = connected_labels(n, np.repeat(np.arange(n), k), knn.ravel())
    labels, counts = np.unique(label, return_counts=True)
    min_size = max(k + 1, int(np.ceil(min_island_frac * n)))
    small = counts < min_size
    island_items = int(counts[small].sum())
    island_labels = labels[small]
    sample = np.nonzero(np.isin(label, island_labels))[0][:10]
    return {
        "island_ratio": island_items / n,
        "n_components": int(counts.size),
        "n_islands": int(small.sum()),
        "min_island_size": int(min_size),
        "largest_component_ratio": float(counts.max() / n),
        "sample_ids": [int(i) for i in sample],
    }
