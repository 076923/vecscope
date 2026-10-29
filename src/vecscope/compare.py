"""Compare two embeddings of the same rows: do two models agree on who is near whom?

Typical uses: two image or text models embedded the same items, the same model before and
after a transform (centering, truncation, re-training), or a candidate model against the one
in production. Like the cluster report this carries no verdicts; it tells the agent how much
the two spaces agree and where they disagree, so it can decide whether a finding (a collapse
of 10%, say) comes from the items or from one model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from vecscope._knn import (
    check_magnitude,
    exact_copy_groups,
    exact_knn,
    normalize,
    resolve_sample,
    row_overlap,
)


@dataclass
class CompareResult:
    tool: str
    version: str
    n: int
    dim_a: int
    dim_b: int
    k: int
    near_dup_cos: float
    sampled: bool
    n_total: int
    nn1_agreement: float
    topk_overlap: float
    topk_overlap_p10: float
    rows_with_no_overlap: float
    near_duplicate_ratio_a: float
    near_duplicate_ratio_b: float
    near_duplicate_both: float
    near_duplicate_only_a: float
    near_duplicate_only_b: float
    same_nearest_among_both_flagged: float | None
    sample_disagreements: list[dict] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def compare(
    a: np.ndarray,
    b: np.ndarray,
    k: int = 10,
    near_dup_cos: float = 0.99,
    sample: int | None = 100_000,
    seed: int = 0,
    sample_disagreements: int = 10,
) -> CompareResult:
    """Compare two embedding matrices whose rows describe the same items in the same order.
    ``sample`` works as in ``audit``: None or 0 compares every row."""
    from vecscope import __version__

    a = np.asarray(a)
    b = np.asarray(b)
    for arr, what in ((a, "embeddings A"), (b, "embeddings B")):
        if arr.dtype.kind not in "iufb":
            raise ValueError(f"{what} must be real numbers, got dtype {arr.dtype}")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] < 1 or b.shape[1] < 1:
        raise ValueError("both inputs must be 2-d (n x dim) with at least one column")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"row counts differ ({a.shape[0]} vs {b.shape[0]}); rows must be the "
                         "same items in the same order")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if not (0.0 < float(near_dup_cos) <= 1.0):
        raise ValueError(f"near_dup_cos must be in (0, 1], got {near_dup_cos!r}")
    near_dup_cos = float(near_dup_cos)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("embeddings contain NaN or Inf")
    check_magnitude(a, "embeddings A")
    check_magnitude(b, "embeddings B")
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape[0] < 3:
        raise ValueError("need at least 3 rows")
    sample = resolve_sample(sample, k)
    n_total = int(a.shape[0])
    sampled = sample is not None and n_total > sample
    if sampled:
        keep = np.sort(np.random.default_rng(seed).choice(n_total, sample, replace=False))
        a, b = a[keep], b[keep]
    n = a.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be smaller than the number of rows ({n})")
    ia, da = exact_knn(normalize(a), k)
    ib, db = exact_knn(normalize(b), k)

    # the same partner, or two partners that are exact copies of each other in both spaces
    # (an index may pick any member of a copy group as the nearest)
    first_a, inv_a, cnt_a = exact_copy_groups(a)
    first_b, inv_b, cnt_b = exact_copy_groups(b)
    nn1 = (ia[:, 0] == ib[:, 0]) | (
        (first_a[inv_a[ia[:, 0]]] == first_a[inv_a[ib[:, 0]]])
        & (first_b[inv_b[ia[:, 0]]] == first_b[inv_b[ib[:, 0]]])
    )
    overlap = row_overlap(ia, ib)
    # as collapse: an exact copy counts whatever its cosine says (zero vectors)
    fa = ((1.0 - da[:, 0].astype(np.float64)) >= near_dup_cos) | (cnt_a[inv_a] > 1)
    fb = ((1.0 - db[:, 0].astype(np.float64)) >= near_dup_cos) | (cnt_b[inv_b] > 1)
    both = fa & fb
    same_nearest = float(nn1[both].mean()) if both.any() else None

    worst = np.argsort(overlap, kind="stable")[:sample_disagreements]
    ids = keep if sampled else np.arange(n)  # report ids of the input rows, not of the subsample
    dis = [{"row": int(ids[i]), "overlap": float(overlap[i]), "nearest_a": int(ids[ia[i, 0]]),
            "nearest_b": int(ids[ib[i, 0]])} for i in worst]

    res = CompareResult(
        "vecscope", __version__, int(n), int(a.shape[1]), int(b.shape[1]), int(k),
        float(near_dup_cos), bool(sampled), n_total,
        nn1_agreement=float(nn1.mean()),
        topk_overlap=float(overlap.mean()),
        topk_overlap_p10=float(np.quantile(overlap, 0.10)),
        rows_with_no_overlap=float((overlap == 0).mean()),
        near_duplicate_ratio_a=float(fa.mean()),
        near_duplicate_ratio_b=float(fb.mean()),
        near_duplicate_both=float(both.mean()),
        near_duplicate_only_a=float((fa & ~fb).mean()),
        near_duplicate_only_b=float((~fa & fb).mean()),
        same_nearest_among_both_flagged=same_nearest,
        sample_disagreements=dis,
    )
    res.explanation = (
        f"the two spaces share the nearest neighbor for {res.nn1_agreement:.1%} of rows and "
        f"{res.topk_overlap:.1%} of their top-{k} lists on average ({res.rows_with_no_overlap:.1%} "
        f"of rows share nothing). Near-duplicates: {res.near_duplicate_ratio_a:.1%} in A, "
        f"{res.near_duplicate_ratio_b:.1%} in B, {res.near_duplicate_both:.1%} in both"
        + (f", and {same_nearest:.1%} of those agree on the partner" if same_nearest is not None
           else "")
        + ". Duplicates flagged by both spaces with the same partner are a property of the "
          "items; duplicates flagged by one space only point at that model."
    )
    return res


def render_compare(res: CompareResult) -> str:
    head = (f"vecscope {res.version}  compare  n={res.n} dim={res.dim_a}/{res.dim_b} k={res.k}"
            + (f"  (sampled from {res.n_total} rows)" if res.sampled else ""))
    same = (f"{res.same_nearest_among_both_flagged:.4f}"
            if res.same_nearest_among_both_flagged is not None else "n/a")
    lines = [head,
             (f"  nn1_agreement={res.nn1_agreement:.4f}  topk_overlap={res.topk_overlap:.4f}  "
              f"topk_overlap_p10={res.topk_overlap_p10:.4f}  rows_with_no_overlap="
              f"{res.rows_with_no_overlap:.4f}"),
             (f"  near_dup A={res.near_duplicate_ratio_a:.4f}  B={res.near_duplicate_ratio_b:.4f}  "
              f"both={res.near_duplicate_both:.4f}  only_A={res.near_duplicate_only_a:.4f}  "
              f"only_B={res.near_duplicate_only_b:.4f}  same_nearest_among_both={same}"),
             f"{'':4}{res.explanation}",
             "  largest disagreements (row: nearest in A / nearest in B, top-k overlap):"]
    for d in res.sample_disagreements:
        lines.append(f"{'':4}{d['row']}: {d['nearest_a']} / {d['nearest_b']}  {d['overlap']:.2f}")
    return "\n".join(lines)
