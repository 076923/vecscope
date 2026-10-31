"""Query-side evidence: do the queries live in the same space as the corpus?

- query_gap: median query->corpus nearest distance over the median corpus->corpus nearest
  distance. Asymmetric encoding keeps good pipelines near 1-2.5; queries from a different
  model land far from every document and push it to 5-10.
- query_shift (needs a reference): share of queries farther from the corpus than the median
  plus 3 MAD of a reference query set (the same pipeline when it was known to be healthy).
- ood_queries (deprecated): queries whose nearest corpus item is farther
  than almost every corpus item is from its own nearest neighbor. It mixes the normal
  query/passage distance into the verdict; query_gap replaces it.
- query_hubness: skewness of how often each corpus item appears in query top-k lists. High
  values mean the same few items answer everything.
- coverage (details): share of corpus items that any query reaches in its top-k.
"""

from __future__ import annotations

import numpy as np

from vecscope._knn import normalize


def query_knn(corpus: np.ndarray, queries: np.ndarray, k: int, block: int | None = None):
    from vecscope._knn import auto_block

    c = normalize(corpus)
    q = normalize(queries)
    k = min(k, c.shape[0])
    if block is None:
        block = auto_block(c.shape[0])
    idx = np.empty((q.shape[0], k), dtype=np.int64)
    dist = np.empty((q.shape[0], k), dtype=np.float32)
    for s in range(0, q.shape[0], block):
        e = min(s + block, q.shape[0])
        d = 1.0 - q[s:e] @ c.T
        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        pd = np.take_along_axis(d, part, axis=1)
        order = np.argsort(pd, axis=1, kind="stable")
        idx[s:e] = np.take_along_axis(part, order, axis=1)
        dist[s:e] = np.take_along_axis(pd, order, axis=1)
    return idx, dist


def query_audit(
    corpus: np.ndarray,
    queries: np.ndarray,
    k: int,
    corpus_nn_dist: np.ndarray,
    sample_ids: int = 10,
    dup_cutoff: float = 1e-4,
    reference: tuple[float, float] | None = None,
) -> dict:
    """``reference`` is (median, MAD) of the nearest corpus distance of a reference query set,
    as reported in ``median_query_nn_distance`` and ``query_nn_mad``; it adds the
    ``shifted_*`` keys."""
    idx, dist = query_knn(corpus, queries, k)
    n_corpus = corpus.shape[0]
    corpus_nn = np.asarray(corpus_nn_dist, dtype=np.float64)
    # drop duplicate rows (NN distance ~0) so they do not drag the OOD cutoff down; when no
    # row has a neighbor that is not a copy of it there is no reference distance at all
    distinct = corpus_nn > dup_cutoff
    corpus_nn = corpus_nn[distinct]
    cutoff = float(np.quantile(corpus_nn, 0.99)) if corpus_nn.size else 0.0
    nearest = dist[:, 0].astype(np.float64)
    ood = nearest > cutoff if corpus_nn.size else np.zeros(nearest.shape[0], dtype=bool)
    worst = np.argsort(-nearest, kind="stable")[:sample_ids]
    median_q = float(np.median(nearest))
    median_c = float(np.median(corpus_nn)) if corpus_nn.size else 0.0

    nk = np.bincount(idx.ravel(), minlength=n_corpus).astype(np.float64)
    std = nk.std()
    skew = float(((nk - nk.mean()) ** 3).mean() / std**3) if std > 0 else 0.0
    out = {
        "n_queries": int(queries.shape[0]),
        "k": int(idx.shape[1]),
        "ood_query_ratio": float(ood.mean()),
        "ood_cutoff_distance": cutoff,
        "median_query_nn_distance": median_q,
        "query_nn_mad": float(np.median(np.abs(nearest - median_q))),
        "median_corpus_nn_distance": median_c,
        "query_gap_ratio": median_q / median_c if median_c > 0 else 0.0,
        "corpus_has_distinct_neighbors": bool(corpus_nn.size),
        "ood_sample_ids": [int(i) for i in worst if ood[i]],
        "query_hubness_skewness": skew,
        "max_query_k_occurrence": int(nk.max()),
        "coverage": float((nk > 0).mean()),
    }
    if reference is not None:
        ref_median, ref_mad = reference
        cutoff_ref = float(ref_median + 3.0 * ref_mad)
        shifted = nearest > cutoff_ref
        out.update({
            "shifted_query_ratio": float(shifted.mean()),
            "shift_cutoff_distance": cutoff_ref,
            "reference_median_nn_distance": float(ref_median),
            "reference_nn_mad": float(ref_mad),
            "shifted_sample_ids": [int(i) for i in worst if shifted[i]],
        })
    return out
