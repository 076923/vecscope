"""ANN recall: does the approximate index actually return the true neighbors?

Compare neighbors returned by an ANN index (HNSW, IVF, ...) with exact k-NN.
Evidence: recall@k, per-query worst recall, share of queries below a floor.
"""

from __future__ import annotations

import numpy as np


def tie_aware_hits(exact_dist: np.ndarray, ann_dist: np.ndarray) -> int:
    """How many of the k exact neighbor slots the returned distances fill.

    ``exact_dist`` holds the k exact distances in ascending order; ``ann_dist`` the distances
    of the (deduplicated) returned ids. A returned id may fill any slot whose distance is
    within the tie tolerance of its own, so exact neighbors at one distance (a duplicate
    group) are interchangeable, while a missed closer neighbor leaves its slot empty. Each
    returned id matches a contiguous run of slots, so assigning ids in ascending order to the
    lowest open slot in their run is a maximum matching."""
    e = np.asarray(exact_dist, dtype=np.float64)
    tol = tie_tolerance(e)
    taken = np.zeros(e.size, dtype=bool)
    hits = 0
    for d in np.sort(np.asarray(ann_dist, dtype=np.float64)):
        open_slots = np.nonzero((np.abs(e - d) <= tol) & ~taken)[0]
        if open_slots.size:
            taken[open_slots[0]] = True
            hits += 1
    return hits


def tie_tolerance(kth: np.ndarray) -> np.ndarray:
    """Slack below which a returned id counts as tied with the k-th exact neighbor: float32
    rounding of a cosine distance (about 1e-6; measured at most 5.9e-7 at 768 dims) plus
    2e-6 of the distance itself, so corpora whose neighbor distances are tiny are not scored
    with a slack larger than the distances. Relative terms of 1e-4 and 1e-5 credited an 11th
    neighbor 1.3e-6 to 7e-5 farther than the 10th and inflated recall by up to +0.0016 and
    +0.0003 against a strict scorer (docs/verification.md); at 2e-6 the slack is 1.7e-6 at
    distance 0.37 (about three times the rounding error), so gaps of 2e-6 and more are misses,
    and exact ties (duplicates) still match."""
    return 1e-6 + 2e-6 * np.asarray(kth, dtype=np.float64)


def ann_recall(exact_knn_idx: np.ndarray, ann_idx: np.ndarray, floor: float = 0.8,
               exact_dist: np.ndarray | None = None, vectors: np.ndarray | None = None) -> dict:
    """recall@k of ``ann_idx`` against exact neighbors. Negative ids (faiss fillers, or the
    item itself masked out by the caller) are ignored; each row contributes its first k
    valid ids, so an index that returns self first should be queried with k + 1.

    With ``exact_dist`` (the exact cosine distances, same shape as ``exact_knn_idx``) and
    ``vectors`` (the rows the ids index), hits are counted by distance: exact neighbors at the
    same distance are interchangeable, so a duplicate group larger than k is not scored as a
    miss whichever members the index picked, while a missed closer neighbor still counts as
    a miss. Both sides are recomputed from ``vectors`` with the same arithmetic, so float32
    rounding between the k-NN pass and this check cannot turn an exact id into a miss.
    Without them, ids are compared."""
    if ann_idx.shape[0] != exact_knn_idx.shape[0]:
        raise ValueError("ann_idx and exact_knn_idx must have the same number of rows")
    k = min(exact_knn_idx.shape[1], ann_idx.shape[1])
    if k == 0:
        raise ValueError("ann_idx and exact_knn_idx need at least one neighbor column")
    exact = exact_knn_idx[:, :k]
    by_distance = exact_dist is not None and vectors is not None
    if by_distance:
        from vecscope._knn import normalize

        q = normalize(vectors)
        if ann_idx.size and ann_idx.max() >= q.shape[0]:
            raise ValueError(f"ann_idx contains id {int(ann_idx.max())} but vectors has "
                             f"{q.shape[0]} rows")
        if np.asarray(exact_dist).shape[1] < k:
            raise ValueError("exact_dist must have at least as many columns as exact_knn_idx")
    hits = np.empty(exact.shape[0])
    short = 0
    for i, (e, a) in enumerate(zip(exact, ann_idx)):
        valid = np.unique(a[a >= 0][:k])
        short += valid.size < k
        if by_distance:
            qi = q[i].astype(np.float64)  # float64: the result must not depend on row order
            ed = np.sort(1.0 - q[e].astype(np.float64) @ qi)
            hits[i] = tie_aware_hits(ed, 1.0 - q[valid].astype(np.float64) @ qi)
        else:
            hits[i] = np.intersect1d(e, valid).size
    per_query = hits / k
    return {
        "k": int(k),
        "recall_at_k": float(per_query.mean()),
        "recall_p05": float(np.quantile(per_query, 0.05)),
        "min_recall": float(per_query.min()),
        "queries_below_floor_ratio": float((per_query < floor).mean()),
        "floor": floor,
        "rows_with_fewer_than_k_ids_ratio": float(short / exact.shape[0]),
    }
