"""Evidence model: deterministic diagnostics with verdicts an agent can act on."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from vecscope._knn import exact_knn
from vecscope.diagnostics import (
    anisotropy,
    ann_recall,
    cluster_report,
    collapse,
    connectivity,
    distance_concentration,
    hubness,
    metric_mismatch,
    norms,
    outliers,
    query_audit,
    rogue_dimensions,
)

# Verdict thresholds. Conservative, documented, overridable.
THRESHOLDS = {
    "hubness.skewness": (2.0, 4.0),  # real 1k-1.5k-dim text models sit at 1.1-1.6
    "hubness.antihub_ratio": (0.05, 0.20),
    "anisotropy.mean_random_cosine": (0.3, 0.6),  # ada-002 ~0.69, 3-large ~0.09
    "anisotropy.effective_rank_ratio": (0.10, 0.05),  # lower is worse; real models 0.09-0.18
    "collapse.near_duplicate_ratio": (0.02, 0.10),
    "ann_recall.recall_at_k": (0.95, 0.85),  # lower is worse
    "rogue_dimensions.top1_dim_variance_share": (0.05, 0.20),
    "norms.norm_cv": (0.10, 0.30),
    "metric_mismatch.topk_overlap": (0.90, 0.70),  # lower is worse
    "distance_concentration.relative_contrast": (1.2, 1.05),  # lower is worse; real 1.9-3.0, 256-d noise 1.28
    "outliers.outlier_ratio": (0.01, 0.05),
    "connectivity.island_ratio": (0.01, 0.10),
    "queries.ood_query_ratio": (0.05, 0.20),
    "queries.query_hubness_skewness": (2.0, 4.0),
}


@dataclass
class Finding:
    name: str
    metric: str
    value: float
    verdict: str  # ok | warn | fail
    explanation: str
    details: dict = field(default_factory=dict)
    warn_at: float | None = None
    fail_at: float | None = None
    lower_is_worse: bool = False
    healthy_range: str = ""


@dataclass
class Evidence:
    tool: str
    version: str
    n: int
    dim: int
    k: int
    findings: list[Finding]
    n_total: int = 0
    sampled: bool = False
    n_queries: int = 0
    clusters: list[dict] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        order = {"ok": 0, "warn": 1, "fail": 2}
        return max((f.verdict for f in self.findings), key=order.get, default="ok")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict
        return d

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _verdict(key: str, value: float) -> str:
    warn, fail = THRESHOLDS[key]
    if warn > fail:  # lower is worse
        return "fail" if value < fail else "warn" if value < warn else "ok"
    return "fail" if value >= fail else "warn" if value >= warn else "ok"


_FINDING_KEYS = {
    "hubness": "hubness.skewness",
    "antihubs": "hubness.antihub_ratio",
    "anisotropy": "anisotropy.mean_random_cosine",
    "effective_rank": "anisotropy.effective_rank_ratio",
    "collapse": "collapse.near_duplicate_ratio",
    "ann_recall": "ann_recall.recall_at_k",
    "rogue_dimensions": "rogue_dimensions.top1_dim_variance_share",
    "norms": "norms.norm_cv",
    "metric_mismatch": "metric_mismatch.topk_overlap",
    "distance_concentration": "distance_concentration.relative_contrast",
    "outliers": "outliers.outlier_ratio",
    "connectivity": "connectivity.island_ratio",
    "ood_queries": "queries.ood_query_ratio",
    "query_hubness": "queries.query_hubness_skewness",
}


def _attach_thresholds(findings: list[Finding]) -> list[Finding]:
    for f in findings:
        warn, fail = THRESHOLDS[_FINDING_KEYS[f.name]]
        f.warn_at, f.fail_at, f.lower_is_worse = warn, fail, warn > fail
        if f.lower_is_worse:
            f.healthy_range = f"ok >= {warn:g}, warn < {warn:g}, fail < {fail:g}"
        else:
            f.healthy_range = f"ok < {warn:g}, warn >= {warn:g}, fail >= {fail:g}"
    return findings


def audit(
    x: np.ndarray,
    k: int = 10,
    ann_neighbors: np.ndarray | None = None,
    near_dup_cos: float = 0.99,
    sample: int | None = 100_000,
    seed: int = 0,
    queries: np.ndarray | None = None,
    clusters: int = 0,
) -> Evidence:
    """Audit an embedding matrix and return deterministic Evidence.

    Exact k-NN is O(n^2); when ``sample`` is set and ``n > sample``, a seeded random
    subset of rows is audited instead (``ann_neighbors`` is then ignored because its
    row ids no longer line up). Pass ``sample=None`` to audit every row.

    ``queries`` adds query-side findings (out-of-distribution queries, query hubness).
    ``clusters`` > 0 attaches a per-cluster report with sample ids (no verdicts).
    """
    from vecscope import __version__

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError("expected a 2-D array with at least 3 rows")
    if not np.isfinite(x).all():
        raise ValueError("embeddings contain NaN or Inf")
    n_total = int(x.shape[0])
    sampled = sample is not None and n_total > sample
    if sampled:
        rows = np.sort(np.random.default_rng(seed).choice(n_total, sample, replace=False))
        x = x[rows]
        ann_neighbors = None
    if k >= x.shape[0]:
        raise ValueError(f"k={k} must be smaller than the number of rows ({x.shape[0]})")
    knn_idx, knn_dist = exact_knn(x, k)
    findings: list[Finding] = []

    h = hubness(x, k, knn=knn_idx)
    findings.append(
        Finding(
            "hubness", "skewness", h["skewness"], _verdict("hubness.skewness", h["skewness"]),
            f"k-occurrence skewness {h['skewness']:.2f}; top 1% of items take "
            f"{h['hub_share_top1pct']:.1%} of all neighbor slots (max N_k={h['max_k_occurrence']}, "
            f"expected {k}).",
            h,
        )
    )
    findings.append(
        Finding(
            "antihubs", "antihub_ratio", h["antihub_ratio"],
            _verdict("hubness.antihub_ratio", h["antihub_ratio"]),
            f"{h['antihub_ratio']:.1%} of items are never anyone's neighbor; they are "
            "unreachable through k-NN retrieval.",
            {},
        )
    )

    a = anisotropy(x)
    findings.append(
        Finding(
            "anisotropy", "mean_random_cosine", a["mean_random_cosine"],
            _verdict("anisotropy.mean_random_cosine", a["mean_random_cosine"]),
            f"random pairs have mean cosine {a['mean_random_cosine']:.3f} (isotropic ~0); "
            f"mean unit vector has norm {a['mean_vector_norm']:.2f}; centering would bring the "
            f"cosine to {a['mean_random_cosine_after_centering']:.3f}.",
            a,
        )
    )
    findings.append(
        Finding(
            "effective_rank", "effective_rank_ratio", a["effective_rank_ratio"],
            _verdict("anisotropy.effective_rank_ratio", a["effective_rank_ratio"]),
            f"effective rank {a['effective_rank']:.1f} of {a['dim']} dims "
            f"({a['effective_rank_ratio']:.1%}); low values mean most dimensions carry nothing.",
            {},
        )
    )

    c = collapse(x, near_dup_cos, knn_dist=knn_dist)
    findings.append(
        Finding(
            "collapse", "near_duplicate_ratio", c["near_duplicate_ratio"],
            _verdict("collapse.near_duplicate_ratio", c["near_duplicate_ratio"]),
            f"{c['near_duplicate_ratio']:.1%} of items have a neighbor with cosine >= {near_dup_cos} "
            f"(exact duplicates {c['exact_duplicate_ratio']:.1%}).",
            c,
        )
    )

    g = rogue_dimensions(x)
    findings.append(
        Finding(
            "rogue_dimensions", "top1_dim_variance_share", g["top1_dim_variance_share"],
            _verdict("rogue_dimensions.top1_dim_variance_share", g["top1_dim_variance_share"]),
            f"dimension {g['rogue_dims'][0]} alone holds {g['top1_dim_variance_share']:.1%} of "
            f"variance (uniform would be {g['expected_share']:.2%}); top 3 dims hold "
            f"{g['top3_dim_variance_share']:.1%}. Rogue dimensions dominate cosine.",
            g,
        )
    )

    nm = norms(x)
    findings.append(
        Finding(
            "norms", "norm_cv", nm["norm_cv"], _verdict("norms.norm_cv", nm["norm_cv"]),
            f"vector norms vary with CV {nm['norm_cv']:.3f} (median {nm['norm_median']:.3f}, "
            f"range {nm['norm_min']:.3f}-{nm['norm_max']:.3f}); {nm['zero_norm_ratio']:.2%} are "
            "zero vectors. Unequal norms bias inner-product search toward long vectors.",
            nm,
        )
    )

    if nm["norm_cv"] < 1e-3:
        mm = {"k": int(k), "topk_overlap": 1.0, "queries_fully_agreeing_ratio": 1.0,
              "min_overlap": 1.0, "skipped": "norms are uniform, cosine == inner product"}
    else:
        dot_idx, _ = exact_knn(x, k, metric="dot")
        mm = metric_mismatch(knn_idx, dot_idx)
    findings.append(
        Finding(
            "metric_mismatch", "topk_overlap", mm["topk_overlap"],
            _verdict("metric_mismatch.topk_overlap", mm["topk_overlap"]),
            f"cosine and inner-product top-{k} share {mm['topk_overlap']:.1%} of neighbors; "
            f"{mm['queries_fully_agreeing_ratio']:.1%} of items get identical lists. "
            "Low overlap means the index metric changes the results.",
            mm,
        )
    )

    dc = distance_concentration(knn_dist[:, 0], a["mean_random_cosine"])
    findings.append(
        Finding(
            "distance_concentration", "relative_contrast", dc["relative_contrast"],
            _verdict("distance_concentration.relative_contrast", dc["relative_contrast"]),
            f"a random item is only {dc['relative_contrast']:.2f}x farther than the nearest "
            f"neighbor (median NN distance {dc['median_nn_distance']:.3f}, random "
            f"{dc['mean_random_distance']:.3f}). Near 1.0 means distances carry no signal.",
            dc,
        )
    )

    o = outliers(knn_dist[:, 0])
    findings.append(
        Finding(
            "outliers", "outlier_ratio", o["outlier_ratio"],
            _verdict("outliers.outlier_ratio", o["outlier_ratio"]),
            f"{o['outlier_ratio']:.2%} of items have no neighbor within cosine distance "
            f"{o['cutoff_nn_distance']:.3f} (median {o['median_nn_distance']:.3f}); see "
            "details.sample_ids. Typically junk, empty or foreign-language content.",
            o,
        )
    )

    cn = connectivity(knn_idx)
    findings.append(
        Finding(
            "connectivity", "island_ratio", cn["island_ratio"],
            _verdict("connectivity.island_ratio", cn["island_ratio"]),
            f"{cn['island_ratio']:.2%} of items sit in {cn['n_islands']} small island(s) of the "
            f"k-NN graph (fewer than {cn['min_island_size']} items; {cn['n_components']} "
            "components in total). Graph-based ANN (HNSW) attaches islands to the wrong region "
            "or never reaches them.",
            cn,
        )
    )

    n_queries = 0
    if queries is not None:
        qv = np.asarray(queries, dtype=np.float32)
        if qv.ndim != 2 or qv.shape[1] != x.shape[1]:
            raise ValueError(f"queries must be (m, {x.shape[1]}) to match the corpus")
        if not np.isfinite(qv).all():
            raise ValueError("queries contain NaN or Inf")
        n_queries = int(qv.shape[0])
        qa = query_audit(x, qv, k, knn_dist[:, 0])
        findings.append(
            Finding(
                "ood_queries", "ood_query_ratio", qa["ood_query_ratio"],
                _verdict("queries.ood_query_ratio", qa["ood_query_ratio"]),
                f"{qa['ood_query_ratio']:.1%} of {qa['n_queries']} queries are farther from "
                f"their nearest corpus item (median {qa['median_query_nn_distance']:.3f}) than "
                f"99% of corpus items are from theirs (cutoff {qa['ood_cutoff_distance']:.3f}). "
                "Out-of-distribution queries: wrong model, missing query prefix, or uncovered "
                "topics; see details.ood_sample_ids.",
                qa,
            )
        )
        findings.append(
            Finding(
                "query_hubness", "query_hubness_skewness", qa["query_hubness_skewness"],
                _verdict("queries.query_hubness_skewness", qa["query_hubness_skewness"]),
                f"k-occurrence skewness over query results {qa['query_hubness_skewness']:.2f} "
                f"(max {qa['max_query_k_occurrence']} appearances); queries reach "
                f"{qa['coverage']:.1%} of the corpus in their top-{qa['k']}.",
                {},
            )
        )

    if ann_neighbors is not None:
        r = ann_recall(knn_idx, np.asarray(ann_neighbors))
        findings.append(
            Finding(
                "ann_recall", "recall_at_k", r["recall_at_k"],
                _verdict("ann_recall.recall_at_k", r["recall_at_k"]),
                f"ANN recall@{r['k']} = {r['recall_at_k']:.3f}; "
                f"{r['queries_below_floor_ratio']:.1%} of queries fall below {r['floor']:.0%}.",
                r,
            )
        )

    report = cluster_report(x, clusters, knn_idx, knn_dist, near_dup_cos, seed) if clusters else []
    return Evidence(
        "vecscope", __version__, int(x.shape[0]), int(x.shape[1]), int(k),
        _attach_thresholds(findings),
        n_total=n_total, sampled=sampled, n_queries=n_queries, clusters=report,
    )
