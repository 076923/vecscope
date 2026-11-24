"""What search returns, kept small enough to diff: the exact top-k of a fixed sample of queries
and corpus rows, so a later run can say how much of it changed without labels.

Without relevance labels no statistic of the space tells a harmful change from a harmless
one. On 594 injected faults over public datasets the share of changed top-k results
separated the two no better than chance (AUC 0.58): a harmless truncation moves as many
results as a harmful one. What it does tell is that the results changed, including faults
no finding sees (a lost id mapping moves no statistic of the space). So the diff reports
the change and fails a gate only when a tolerance for it is set, for a pipeline that is
meant to return the same results (a re-index, a refactor, a library upgrade), and a change
that is meant to move results goes to labeled queries or a person.

The sample is chosen by a multiplicative hash of the row number, keeping the rows with the
smallest hashes. Rows appended later never displace a kept row from the old range, they
only take part of the sample, so a baseline and a grown corpus still share most of it.
"""

from __future__ import annotations

import numpy as np

from vecscope._knn import normalize

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def pick_rows(n: int, m: int) -> np.ndarray:
    """``m`` row numbers out of ``n``, the same ones for every ``n`` up to the hash order."""
    m = min(int(m), int(n))
    if m <= 0:
        return np.empty(0, dtype=np.int64)
    h = np.arange(n, dtype=np.uint64) * _GOLDEN  # wraps modulo 2**64
    return np.sort(np.argsort(h, kind="stable")[:m]).astype(np.int64)


def _topk(src: np.ndarray, corpus: np.ndarray, k: int, self_rows: np.ndarray | None) -> np.ndarray:
    """Exact cosine top-k corpus rows for each row of ``src``; ties go to the lower row id."""
    q, c = normalize(src), normalize(corpus)
    k = min(k, c.shape[0] - (1 if self_rows is not None else 0))
    out = np.empty((q.shape[0], k), dtype=np.int64)
    for s in range(0, q.shape[0], 256):
        sim = q[s : s + 256] @ c.T
        if self_rows is not None:
            sim[np.arange(sim.shape[0]), self_rows[s : s + 256]] = -np.inf
        part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(sim, part, axis=1)
        order = np.lexsort((part, -ps), axis=1)
        out[s : s + 256] = np.take_along_axis(part, order, axis=1)
    return out


def result_sample(x: np.ndarray, queries: np.ndarray | None, m: int, k: int) -> dict:
    """Top-k corpus rows of ``m`` sampled queries (when given) and ``m`` sampled corpus rows
    (their neighbors, self excluded), all over the full corpus."""
    out: dict = {"k": int(k), "m": int(m), "n": int(x.shape[0])}
    rows = pick_rows(x.shape[0], m)
    out["rows"] = rows.tolist()
    out["row_neighbors"] = _topk(x[rows], x, k, rows).tolist()
    if queries is not None:
        qrows = pick_rows(queries.shape[0], m)
        out["n_queries"] = int(queries.shape[0])
        out["queries"] = qrows.tolist()
        out["query_results"] = _topk(queries[qrows], x, k, None).tolist()
    return out


def _agreement(ids_a, res_a, ids_b, res_b) -> dict | None:
    a = dict(zip(ids_a, res_a))
    b = dict(zip(ids_b, res_b))
    shared = sorted(a.keys() & b.keys())
    if not shared:
        return None
    k = min(len(a[shared[0]]), len(b[shared[0]]))
    over = np.array([len(set(a[i][:k]) & set(b[i][:k])) / k for i in shared])
    return {
        "compared": len(shared),
        "share_of_sample": len(shared) / max(len(a), len(b)),
        "mean_overlap": float(over.mean()),
        "changed_share": float(1.0 - over.mean()),
        "unchanged_items": float((over == 1.0).mean()),
    }


def result_agreement(new: dict, old: dict) -> dict:
    """``{"queries": ..., "rows": ...}``: per side, how much of the sampled top-k the two runs
    share; ``changed_share`` is the share of top-k slots that hold another row. A side is
    missing when either run lacks it or the samples share no item."""
    out = {}
    if not all("rows" in s for s in (new or {}, old or {})) or new.get("k") != old.get("k"):
        return out
    r = _agreement(old["rows"], old["row_neighbors"], new["rows"], new["row_neighbors"])
    if r is not None:
        out["rows"] = r
    if "queries" in new and "queries" in old and new.get("n_queries") == old.get("n_queries"):
        q = _agreement(old["queries"], old["query_results"], new["queries"], new["query_results"])
        if q is not None:
            out["queries"] = q
    return out
