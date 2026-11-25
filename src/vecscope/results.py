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

With a reference model (the same rows and queries embedded by another, independently good
model), each run also stores how much of every sampled top-k the reference returns too
(``reference_overlap``). The reference stands in for labels: a change that moves results
away from what the reference retrieves on the same rows is more likely harmful than one that
moves them as much but keeps the agreement. On the first held-out set's injected faults
(duplicates left out, see below), the paired drop in that overlap over 200 sampled queries
separated harmful from harmless changes at AUC 0.875, where the changed share did at 0.56.
It is a proxy: it cannot see rows that are wrong for both models (duplicated rows show up
in the reference too; ``collapse`` finds them), and a change that makes the pipeline better
than the reference also lowers the overlap. Read a drop as "likely harmful, check with
labels", not as a measured loss.

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


def _overlap(a: np.ndarray, b: np.ndarray) -> list[float]:
    return [len(set(i.tolist()) & set(j.tolist())) / max(len(i), 1) for i, j in zip(a, b)]


def result_sample(
    x: np.ndarray,
    queries: np.ndarray | None,
    m: int,
    k: int,
    reference: np.ndarray | None = None,
    reference_queries: np.ndarray | None = None,
) -> dict:
    """Top-k corpus rows of ``m`` sampled queries (when given) and ``m`` sampled corpus rows
    (their neighbors, self excluded), all over the full corpus. With ``reference`` (the same
    rows embedded by another model) and ``reference_queries``, ``reference_overlap`` holds,
    per sampled item, the share of its top-k the reference returns too."""
    out: dict = {"k": int(k), "m": int(m), "n": int(x.shape[0])}
    rows = pick_rows(x.shape[0], m)
    out["rows"] = rows.tolist()
    nb = _topk(x[rows], x, k, rows)
    out["row_neighbors"] = nb.tolist()
    ref: dict = {}
    if reference is not None:
        ref["rows"] = _overlap(nb, _topk(reference[rows], reference, k, rows))
    if queries is not None:
        qrows = pick_rows(queries.shape[0], m)
        out["n_queries"] = int(queries.shape[0])
        out["queries"] = qrows.tolist()
        res = _topk(queries[qrows], x, k, None)
        out["query_results"] = res.tolist()
        if reference is not None and reference_queries is not None:
            ref["queries"] = _overlap(res, _topk(reference_queries[qrows], reference, k, None))
    if ref:
        out["reference_overlap"] = ref
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


# a drop in the overlap with the reference counts only beyond this many standard errors
# of the paired per-item drop, so sampling noise alone does not read as a change
REFERENCE_Z = 2.0
# the drop a change decision reads as "moved away" when no tolerance was given: on the first
# held-out set's faults it flagged 79% of changes that cost more than 0.02 nDCG@10 and 24% of
# harmless ones (200 sampled queries)
REFERENCE_TOLERANCE = 0.02


def _reference_drop(ids_a, ov_a, ids_b, ov_b) -> dict | None:
    """Paired drop of the reference overlap from run a (baseline) to run b (new)."""
    a, b = dict(zip(ids_a, ov_a)), dict(zip(ids_b, ov_b))
    shared = sorted(a.keys() & b.keys())
    if not shared:
        return None
    d = np.array([a[i] - b[i] for i in shared], dtype=np.float64)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    mean = float(d.mean())
    if se > 0:
        z = mean / se
    else:
        z = 0.0 if mean == 0 else (1e6 if mean > 0 else -1e6)
    return {
        "compared": len(shared),
        "baseline_overlap": float(np.mean([a[i] for i in shared])),
        "overlap": float(np.mean([b[i] for i in shared])),
        "drop": mean,
        "standard_error": se,
        "z": float(z),
    }


def result_agreement(new: dict, old: dict) -> dict:
    """``{"queries": ..., "rows": ...}``: per side, how much of the sampled top-k the two runs
    share; ``changed_share`` is the share of top-k slots that hold another row. A side is
    missing when either run lacks it or the samples share no item."""
    out = {}
    if not all("rows" in s for s in (new or {}, old or {})) or new.get("k") != old.get("k"):
        return out
    ra, rb = old.get("reference_overlap") or {}, new.get("reference_overlap") or {}
    r = _agreement(old["rows"], old["row_neighbors"], new["rows"], new["row_neighbors"])
    if r is not None:
        out["rows"] = r
        if "rows" in ra and "rows" in rb:
            ref = _reference_drop(old["rows"], ra["rows"], new["rows"], rb["rows"])
            if ref is not None:
                r["reference"] = ref
    if "queries" in new and "queries" in old and new.get("n_queries") == old.get("n_queries"):
        q = _agreement(old["queries"], old["query_results"], new["queries"], new["query_results"])
        if q is not None:
            out["queries"] = q
            if "queries" in ra and "queries" in rb:
                ref = _reference_drop(old["queries"], ra["queries"], new["queries"], rb["queries"])
                if ref is not None:
                    q["reference"] = ref
    return out


def reference_regressions(agreement: dict, tolerance: float) -> list[str]:
    """Sides whose overlap with the reference dropped by more than ``tolerance`` and by more
    than ``REFERENCE_Z`` standard errors. The queries side decides when it was measured (it
    is the one that separated harmful changes); the rows side only without queries."""
    sides = [s for s in ("queries", "rows") if "reference" in agreement.get(s, {})]
    if "queries" in sides:
        sides = ["queries"]
    return [
        s
        for s in sides
        if agreement[s]["reference"]["drop"] > tolerance
        and agreement[s]["reference"]["z"] > REFERENCE_Z
    ]
