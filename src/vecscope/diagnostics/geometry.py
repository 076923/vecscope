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

import functools

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
    from vecscope._knn import row_overlap

    k = min(cos_knn.shape[1], dot_knn.shape[1])
    overlap = row_overlap(cos_knn, dot_knn)
    return {
        "k": int(k),
        "topk_overlap": float(overlap.mean()),
        "queries_fully_agreeing_ratio": float((overlap == 1.0).mean()),
        "min_overlap": float(overlap.min()),
    }


BASELINE_ROWS = 4000  # rows used for the noise baseline and the matching data subsample


def noise_relative_contrast(n: int, dim: int, seed: int = 0, cap: int = BASELINE_ROWS) -> float:
    """Relative contrast of seeded isotropic Gaussian noise with the same dimension.

    Pure noise has no neighbor structure, yet its relative contrast is not 1: it falls with
    the dimension (about 1.43 at 128-d, 1.14 at 768-d, 1.10 at 1536-d) and rises slowly with
    the number of rows. Dividing by this baseline makes the finding comparable across models
    of different width. At most ``cap`` rows are used, and ``distance_concentration`` measures
    the data on a subsample of the same size so both sides see the same n.
    """
    return _noise_relative_contrast(int(min(max(n, 2), cap)), int(dim), int(seed))


@functools.lru_cache(maxsize=16)
def _noise_relative_contrast(m: int, dim: int, seed: int) -> float:
    from vecscope._knn import exact_knn

    g = np.random.default_rng(seed).standard_normal((m, int(dim))).astype(np.float32)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    _, d = exact_knn(g, 1)
    median_nn = float(np.median(d[:, 0]))
    mean_random = float(1.0 - (g.mean(0) @ g.mean(0)))  # 1 - E[cos] over random pairs
    return mean_random / max(median_nn, 1e-6)


def duplicate_cutoff(mean_random_distance: float) -> float:
    """Cosine distance below which two rows count as the same vector: 1e-4 of the typical
    distance, floored at float32 resolution near cos = 1."""
    return max(1e-6, 1e-4 * float(mean_random_distance))


def _median_distinct(nn: np.ndarray, cutoff: float) -> float:
    # exact duplicates (float32 noise ~1e-7) would pin the median at 0; the cutoff scales with
    # the typical distance so a heavily offset space (random distance ~0.02) keeps its neighbors
    distinct = nn[nn > cutoff]
    return float(np.median(distinct)) if distinct.size else 0.0


def distance_concentration(nn_dist: np.ndarray, random_pair_cos: float, dim: int | None = None,
                           seed: int = 0, x: np.ndarray | None = None,
                           cap: int = BASELINE_ROWS) -> dict:
    """Relative contrast on cosine distance (mean random distance over median NN distance),
    and the same number divided by what isotropic noise of this dimension would give.

    The nearest-neighbor distance shrinks as rows are added, so the noise baseline (computed
    on at most ``cap`` rows) is compared with the data measured on a seeded subsample of the
    same size when ``x`` is given and has more rows than that. ``relative_contrast`` itself
    always describes the full input.
    """
    nn = np.asarray(nn_dist, dtype=np.float64)
    mean_random = float(max(1.0 - random_pair_cos, 1e-9))
    cutoff = duplicate_cutoff(mean_random)
    median_nn = _median_distinct(nn, cutoff)
    rc = mean_random / max(median_nn, 1e-6)
    out = {
        "relative_contrast": rc,
        "median_nn_distance": median_nn,
        "mean_random_distance": mean_random,
    }
    if dim is not None:
        n = int(nn.shape[0])
        median_at_base = median_nn
        if x is not None and n > cap:
            from vecscope._knn import exact_knn, normalize

            rows = np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))
            _, d = exact_knn(normalize(np.asarray(x)[rows]), 1)
            median_at_base = _median_distinct(d[:, 0].astype(np.float64), cutoff)
        base = noise_relative_contrast(n, dim, seed, cap)
        out["baseline_rows"] = int(min(n, cap))
        out["median_nn_distance_at_baseline_rows"] = median_at_base
        out["noise_relative_contrast"] = base
        out["contrast_over_noise"] = (mean_random / max(median_at_base, 1e-6)) / max(base, 1e-6)
    return out


def outliers(nn_dist: np.ndarray, factor: float = 3.0, sample_ids: int = 10,
             dup_cutoff: float = 1e-4) -> dict:
    nn = np.asarray(nn_dist, dtype=np.float64)
    distinct = nn[nn > dup_cutoff] if (nn > dup_cutoff).any() else nn
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
