"""Per-cluster report: where in the space do the problems live?

A fixed-seed spherical k-means splits the corpus into a few regions; for each one the report
gives size, tightness, near-duplicate share, how often its items act as hubs, and sample ids
the agent can look up. No verdicts: this is a map, not a test.
"""

from __future__ import annotations

import numpy as np

from vecscope._knn import exact_copy_groups, normalize


def spherical_kmeans(q: np.ndarray, n_clusters: int, seed: int = 0, iters: int = 15) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = q[rng.choice(q.shape[0], n_clusters, replace=False)].copy()
    labels = np.zeros(q.shape[0], dtype=np.int64)
    for _ in range(iters):
        sims = q @ centers.T
        new = sims.argmax(axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for c in range(n_clusters):
            members = q[labels == c]
            if members.size:
                centers[c] = members.mean(axis=0)
        centers = normalize(centers)
    return labels


def cluster_report(
    x: np.ndarray,
    n_clusters: int,
    knn_idx: np.ndarray,
    knn_dist: np.ndarray,
    near_dup_cos: float,
    seed: int = 0,
    sample_ids: int = 5,
) -> list[dict]:
    q = normalize(x)
    n = q.shape[0]
    n_clusters = max(1, min(n_clusters, n // 10))
    labels = spherical_kmeans(q, n_clusters, seed)
    nk = np.bincount(knn_idx.ravel(), minlength=n).astype(np.float64)
    _, inverse, counts = exact_copy_groups(x)
    nn_cos = 1.0 - np.asarray(knn_dist[:, 0], dtype=np.float64)  # float64, as in collapse
    near_dup = (nn_cos >= near_dup_cos) | (counts[inverse] > 1)
    report = []
    for c in range(n_clusters):
        m = np.nonzero(labels == c)[0]
        if m.size == 0:
            continue
        center = normalize(q[m].mean(axis=0, keepdims=True))[0]
        report.append({
            "cluster": int(c),
            "size": int(m.size),
            "share": float(m.size / n),
            "mean_cosine_to_center": float((q[m] @ center).mean()),
            "near_duplicate_ratio": float(near_dup[m].mean()),
            "hub_share": float(nk[m].sum() / nk.sum()) if nk.sum() > 0 else 0.0,
            "sample_ids": [int(i) for i in m[np.argsort(-(q[m] @ center), kind="stable")[:sample_ids]]],
        })
    report.sort(key=lambda r: -r["size"])
    return report
