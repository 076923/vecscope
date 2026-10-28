"""Evidence model: deterministic diagnostics with verdicts an agent can act on."""

from __future__ import annotations

import json
from collections.abc import Mapping
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
    duplicate_cutoff,
    hubness,
    metric_mismatch,
    norms,
    outliers,
    query_audit,
    rogue_dimensions,
)

# Default verdict thresholds: (warn, fail). Calibrated on eight public embedding models
# (see README "Real embeddings"); they are a starting point for diagnosis, not a universal
# quality bar. Override per call with ``audit(..., thresholds={key: (warn, fail)})``.
THRESHOLDS = {
    "hubness.skewness": (2.0, 4.0),  # real 1k-1.5k-dim text models sit at 1.1-1.6
    "hubness.antihub_excess": (0.05, 0.20),  # share never retrieved, above the exp(-k) baseline
    "anisotropy.mean_random_cosine": (0.5, 0.85),  # top BEIR models sit at 0.58-0.81 with no nDCG cost
    "anisotropy.effective_rank_ratio": (0.10, 0.05),  # lower is worse; real models 0.09-0.18
    "collapse.near_duplicate_ratio": (0.02, 0.10),
    "ann_recall.recall_at_k": (0.95, 0.85),  # lower is worse
    "rogue_dimensions.top1_dim_variance_ratio": (10.0, 50.0),  # x the uniform 1/dim share; real 1.5-3
    "norms.norm_cv": (0.10, 0.30),
    "metric_mismatch.topk_overlap": (0.90, 0.70),  # lower is worse
    "distance_concentration.contrast_over_noise": (1.4, 1.1),  # lower is worse; pure noise 1.0, real 1.43-2.9
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
    likely_causes: list[str] = field(default_factory=list)
    suggested_action: str = ""
    verify_by: str = ""
    value_range: list[float] | None = None  # [min, max] over sample repeats, when sampled
    verdict_stable: bool | None = None  # verdict identical on every sample repeat


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
    thresholds: dict = field(default_factory=dict)
    sample_repeats: int = 1
    unstable_findings: list[str] = field(default_factory=list)

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


_RENAMED_KEYS = {
    "distance_concentration.relative_contrast": "distance_concentration.contrast_over_noise",
}


def resolve_thresholds(overrides: dict | None) -> dict:
    """Merge user overrides into the defaults, validating keys and (warn, fail) pairs."""
    th = dict(THRESHOLDS)
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ValueError("thresholds must be a mapping of key -> (warn, fail)")
    for key, pair in (overrides or {}).items():
        if key in _RENAMED_KEYS:
            raise ValueError(
                f"threshold key {key!r} was renamed to {_RENAMED_KEYS[key]!r} and its metric "
                f"changed (relative contrast divided by the noise baseline, 1.0 = noise); "
                f"re-derive the override, defaults are {THRESHOLDS[_RENAMED_KEYS[key]]}"
            )
        if key not in THRESHOLDS:
            raise ValueError(f"unknown threshold key {key!r}; known: {sorted(THRESHOLDS)}")
        try:
            warn, fail = (float(pair[0]), float(pair[1]))
        except (TypeError, ValueError, IndexError, KeyError) as e:
            raise ValueError(f"threshold {key!r} must be a (warn, fail) pair") from e
        if not (np.isfinite(warn) and np.isfinite(fail)):
            raise ValueError(f"threshold {key!r} must be finite numbers")
        default_warn, default_fail = THRESHOLDS[key]
        if warn != fail and (default_warn > default_fail) != (warn > fail):
            raise ValueError(
                f"threshold {key!r}: keep the direction of the default "
                f"({'lower' if default_warn > default_fail else 'higher'} is worse)"
            )
        th[key] = (warn, fail)
    return th


def _verdict(key: str, value: float, th: dict) -> str:
    warn, fail = th[key]
    if warn > fail:  # lower is worse
        return "fail" if value < fail else "warn" if value < warn else "ok"
    return "fail" if value >= fail else "warn" if value >= warn else "ok"


# What an agent should consider when a finding is not ok. Each entry: likely causes, the
# smallest change to try first, and how to confirm it helped. The audit is evidence; the
# agent still has to test the change on real queries before adopting it.
ACTIONS = {
    "hubness": (
        ["high-dimensional concentration (dim >> effective rank)",
         "strong shared direction (see anisotropy)",
         "a few generic items close to the corpus mean"],
        "re-rank top-k with mutual k-NN or local scaling; if anisotropy also fails, center first",
        ("re-run the audit on the transformed vectors and compare hubness.skewness and "
        "details.hub_share_top1pct; check that recall on held-out queries did not drop"),
    ),
    "antihubs": (
        ["same causes as hubness", "items embedded from empty or near-empty text"],
        "inspect a sample of never-retrieved items; drop empty inputs, then apply the hubness fix",
        ("antihub_ratio should fall below warn_at after the fix; spot-check that the dropped "
        "items were genuinely empty"),
    ),
    "anisotropy": (
        ["model emits a large shared component (common in older text embedding models)",
         "un-normalized vectors with a constant offset"],
        ("subtract the corpus mean vector and re-normalize, applying the same transform to "
        "queries; details.mean_random_cosine_after_centering shows the expected result"),
        ("re-audit the centered vectors; then compare retrieval quality on real queries "
        "before and after, because centering changes the ranking"),
    ),
    "effective_rank": (
        ["model dimensions mostly unused", "corpus covers a very narrow topic",
         "a rogue dimension dominates the variance"],
        ("if rogue_dimensions also fails, standardize dimensions; otherwise consider PCA or "
        "Matryoshka truncation to the effective rank to cut index size"),
        ("re-audit after the transform; recall@k of the truncated index against the full "
        "exact k-NN should stay above 0.95"),
    ),
    "collapse": (
        ["duplicate or boilerplate documents", "chunking that repeats headers/footers",
         "embedding failures returning a constant vector"],
        ("read details.largest_near_duplicate_groups: a few huge groups point at an embedding "
         "failure or boilerplate (re-embed or fix chunking), many small groups at duplicate "
         "items (deduplicate or collapse them at serving time); details.sample_pairs lists row "
         "pairs to inspect"),
        ("near_duplicate_ratio should drop below warn_at; top-k on sample queries should no "
        "longer repeat the same item"),
    ),
    "ann_recall": (
        ["ef_search / nprobe too low", "index built with too few links (HNSW M)",
         "hard corpus: see distance_concentration and connectivity"],
        ("raise ef_search (HNSW) or nprobe (IVF) first; rebuild with larger M / "
        "ef_construction only if that is not enough; run `vecscope sweep` to pick values"),
        "recall_at_k against exact k-NN reaches the target; queries_per_second stays acceptable",
    ),
    "rogue_dimensions": (
        ["model outputs a few high-variance coordinates (known in BERT-family encoders)"],
        ("z-score each dimension or drop details.rogue_dims before indexing; apply the same "
        "transform to queries"),
        ("top1_dim_variance_share falls below warn_at; retrieval quality on real queries "
        "does not regress"),
    ),
    "norms": (
        ["vectors not L2-normalized", "zero vectors from empty inputs or failed embedding calls"],
        "drop zero vectors; L2-normalize before indexing if cosine ranking is intended",
        "norm_cv near 0 and zero_norm_ratio 0 on re-audit; metric_mismatch returns to 1.0",
    ),
    "metric_mismatch": (
        ["vector norms vary, so cosine and inner-product rankings diverge"],
        ("decide which metric the index uses; normalize vectors if cosine is intended, or "
        "keep norms deliberately if length should carry relevance"),
        "topk_overlap above warn_at, or a documented decision to keep inner product",
    ),
    "distance_concentration": (
        ["curse of dimensionality on near-isotropic vectors", ("model with little structure "
         "for this domain")],
        ("try a different model or dimensionality reduction; add a re-ranker over a larger "
        "candidate set"),
        "contrast_over_noise rises above warn_at; expect ann_recall to improve as well",
    ),
    "outliers": (
        ["junk, empty or foreign-language documents", ("chunks of code or tables embedded as "
         "prose"), "a few vectors from a different model"],
        "inspect details.sample_ids; remove or re-embed them",
        "outlier_ratio below warn_at; the removed items were confirmed irrelevant",
    ),
    "connectivity": (
        ["tiny self-contained groups (near-duplicates, boilerplate) isolated from the rest",
         "corpus assembled from disconnected sources"],
        "inspect details.sample_ids; deduplicate or merge; for HNSW raise M / ef_construction",
        ("island_ratio below warn_at; ANN recall measured on items from former islands is "
        "comparable to the rest"),
    ),
    "ood_queries": (
        ["queries embedded with a different model than the corpus",
         "asymmetric model used without its query prefix / instruction",
         "questions about topics the corpus does not cover"],
        ("confirm the same model and prefix on both sides; inspect details.ood_sample_ids "
        "for uncovered topics"),
        ("ood_query_ratio below warn_at after fixing the query pipeline; otherwise extend "
        "the corpus"),
    ),
    "query_hubness": (
        ["a few generic items near the query mean", "queries that are too short or generic"],
        "re-rank or apply hubness reduction; check query construction",
        "query_hubness_skewness below warn_at and details.coverage rising",
    ),
}


_FINDING_KEYS = {
    "hubness": "hubness.skewness",
    "antihubs": "hubness.antihub_excess",
    "anisotropy": "anisotropy.mean_random_cosine",
    "effective_rank": "anisotropy.effective_rank_ratio",
    "collapse": "collapse.near_duplicate_ratio",
    "ann_recall": "ann_recall.recall_at_k",
    "rogue_dimensions": "rogue_dimensions.top1_dim_variance_ratio",
    "norms": "norms.norm_cv",
    "metric_mismatch": "metric_mismatch.topk_overlap",
    "distance_concentration": "distance_concentration.contrast_over_noise",
    "outliers": "outliers.outlier_ratio",
    "connectivity": "connectivity.island_ratio",
    "ood_queries": "queries.ood_query_ratio",
    "query_hubness": "queries.query_hubness_skewness",
}


def _attach_thresholds(findings: list[Finding], th: dict) -> list[Finding]:
    for f in findings:
        warn, fail = th[_FINDING_KEYS[f.name]]
        f.warn_at, f.fail_at, f.lower_is_worse = warn, fail, warn > fail
        causes, action, verify = ACTIONS[f.name]
        f.likely_causes, f.suggested_action, f.verify_by = list(causes), action, verify
        if f.lower_is_worse:
            f.healthy_range = f"ok >= {warn:g}, warn < {warn:g}, fail < {fail:g}"
        else:
            f.healthy_range = f"ok < {warn:g}, warn >= {warn:g}, fail >= {fail:g}"
    return findings


def _core_findings(
    x: np.ndarray, k: int, near_dup_cos: float, queries: np.ndarray | None, th: dict
) -> tuple[list[Finding], np.ndarray, np.ndarray]:
    """Compute every finding that depends only on the (sampled) matrix and the queries."""
    knn_idx, knn_dist = exact_knn(x, k)
    findings: list[Finding] = []

    h = hubness(x, k, knn=knn_idx)
    findings.append(
        Finding(
            "hubness", "skewness", h["skewness"], _verdict("hubness.skewness", h["skewness"], th),
            f"k-occurrence skewness {h['skewness']:.2f}; top 1% of items take "
            f"{h['hub_share_top1pct']:.1%} of all neighbor slots (max N_k={h['max_k_occurrence']}, "
            f"expected {k}).",
            h,
        )
    )
    findings.append(
        Finding(
            "antihubs", "antihub_excess", h["antihub_excess"],
            _verdict("hubness.antihub_excess", h["antihub_excess"], th),
            f"{h['antihub_ratio']:.1%} of items are never anyone's neighbor (random data at "
            f"k={k} would give {h['antihub_baseline']:.1%}); the excess is unreachable "
            "through k-NN retrieval.",
            {"antihub_ratio": h["antihub_ratio"], "antihub_baseline": h["antihub_baseline"]},
        )
    )

    a = anisotropy(x)
    findings.append(
        Finding(
            "anisotropy", "mean_random_cosine", a["mean_random_cosine"],
            _verdict("anisotropy.mean_random_cosine", a["mean_random_cosine"], th),
            f"random pairs have mean cosine {a['mean_random_cosine']:.3f} (isotropic ~0); "
            f"mean unit vector has norm {a['mean_vector_norm']:.2f}; centering would bring the "
            f"cosine to {a['mean_random_cosine_after_centering']:.3f}.",
            a,
        )
    )
    findings.append(
        Finding(
            "effective_rank", "effective_rank_ratio", a["effective_rank_ratio"],
            _verdict("anisotropy.effective_rank_ratio", a["effective_rank_ratio"], th),
            f"effective rank {a['effective_rank']:.1f} of {a['dim']} dims "
            f"({a['effective_rank_ratio']:.1%}); low values mean most dimensions carry nothing.",
            {},
        )
    )

    c = collapse(x, near_dup_cos, knn_dist=knn_dist, knn_idx=knn_idx)
    groups = c["largest_near_duplicate_groups"]
    if groups:
        shape = (f"they form {c['near_duplicate_groups']} groups, the largest of "
                 f"{groups[0]} rows; largest exact-duplicate group "
                 f"{(c['largest_exact_duplicate_groups'] or [1])[0]} rows. Many small groups "
                 "mean duplicate items, a few huge groups mean an embedding failure or "
                 "boilerplate; see details.sample_pairs.")
    else:
        shape = "no row has a near-duplicate."
    findings.append(
        Finding(
            "collapse", "near_duplicate_ratio", c["near_duplicate_ratio"],
            _verdict("collapse.near_duplicate_ratio", c["near_duplicate_ratio"], th),
            f"{c['near_duplicate_ratio']:.1%} of items have a neighbor with cosine >= {near_dup_cos} "
            f"(exact duplicates {c['exact_duplicate_ratio']:.1%}); {shape}",
            c,
        )
    )

    g = rogue_dimensions(x)
    findings.append(
        Finding(
            "rogue_dimensions", "top1_dim_variance_ratio", g["top1_dim_variance_ratio"],
            _verdict("rogue_dimensions.top1_dim_variance_ratio", g["top1_dim_variance_ratio"], th),
            f"dimension {g['rogue_dims'][0]} alone holds {g['top1_dim_variance_share']:.1%} of "
            f"variance, {g['top1_dim_variance_ratio']:.1f}x the uniform share of "
            f"{g['expected_share']:.2%}; top 3 dims hold {g['top3_dim_variance_share']:.1%}. "
            "Rogue dimensions dominate cosine.",
            g,
        )
    )

    nm = norms(x)
    findings.append(
        Finding(
            "norms", "norm_cv", nm["norm_cv"], _verdict("norms.norm_cv", nm["norm_cv"], th),
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
            _verdict("metric_mismatch.topk_overlap", mm["topk_overlap"], th),
            f"cosine and inner-product top-{k} share {mm['topk_overlap']:.1%} of neighbors; "
            f"{mm['queries_fully_agreeing_ratio']:.1%} of items get identical lists. "
            "Low overlap means the index metric changes the results.",
            mm,
        )
    )

    dc = distance_concentration(knn_dist[:, 0], a["mean_random_cosine"], dim=x.shape[1], x=x)
    findings.append(
        Finding(
            "distance_concentration", "contrast_over_noise", dc["contrast_over_noise"],
            _verdict("distance_concentration.contrast_over_noise", dc["contrast_over_noise"], th),
            f"a random item is {dc['relative_contrast']:.2f}x farther than the nearest neighbor "
            f"(median NN distance {dc['median_nn_distance']:.3f}, random "
            f"{dc['mean_random_distance']:.3f}); isotropic noise at {x.shape[1]}-d and "
            f"{dc['baseline_rows']} rows would give {dc['noise_relative_contrast']:.2f}x, so "
            f"the contrast is {dc['contrast_over_noise']:.2f}x that of noise. Near 1.0 means "
            "distances carry no signal.",
            dc,
        )
    )

    o = outliers(knn_dist[:, 0], dup_cutoff=duplicate_cutoff(dc["mean_random_distance"]))
    findings.append(
        Finding(
            "outliers", "outlier_ratio", o["outlier_ratio"],
            _verdict("outliers.outlier_ratio", o["outlier_ratio"], th),
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
            _verdict("connectivity.island_ratio", cn["island_ratio"], th),
            f"{cn['island_ratio']:.2%} of items sit in {cn['n_islands']} small island(s) of the "
            f"k-NN graph (fewer than {cn['min_island_size']} items; {cn['n_components']} "
            "components in total). Graph-based ANN (HNSW) attaches islands to the wrong region "
            "or never reaches them.",
            cn,
        )
    )

    if queries is not None:
        qv = np.asarray(queries, dtype=np.float32)
        if qv.ndim != 2 or qv.shape[1] != x.shape[1]:
            raise ValueError(f"queries must be (m, {x.shape[1]}) to match the corpus")
        if not np.isfinite(qv).all():
            raise ValueError("queries contain NaN or Inf")
        qa = query_audit(x, qv, k, knn_dist[:, 0],
                         dup_cutoff=duplicate_cutoff(dc["mean_random_distance"]))
        findings.append(
            Finding(
                "ood_queries", "ood_query_ratio", qa["ood_query_ratio"],
                _verdict("queries.ood_query_ratio", qa["ood_query_ratio"], th),
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
                _verdict("queries.query_hubness_skewness", qa["query_hubness_skewness"], th),
                f"k-occurrence skewness over query results {qa['query_hubness_skewness']:.2f} "
                f"(max {qa['max_query_k_occurrence']} appearances); queries reach "
                f"{qa['coverage']:.1%} of the corpus in their top-{qa['k']}.",
                {},
            )
        )

    return findings, knn_idx, knn_dist


def audit(
    x: np.ndarray,
    k: int = 10,
    ann_neighbors: np.ndarray | None = None,
    near_dup_cos: float = 0.99,
    sample: int | None = 100_000,
    seed: int = 0,
    queries: np.ndarray | None = None,
    clusters: int = 0,
    thresholds: dict | None = None,
    sample_repeats: int = 1,
) -> Evidence:
    """Audit an embedding matrix and return deterministic Evidence.

    Exact k-NN is O(n^2); when ``sample`` is set and ``n > sample``, a seeded random
    subset of rows is audited instead. ``ann_neighbors`` requires every row (its ids index
    the rows), so combining it with sampling raises. Pass ``sample=None`` (or 0) to audit
    every row.

    ``queries`` adds query-side findings (out-of-distribution queries, query hubness).
    ``clusters`` > 0 attaches a per-cluster report with sample ids (no verdicts).
    ``thresholds`` overrides default (warn, fail) pairs by key, e.g.
    ``{"anisotropy.mean_random_cosine": (0.4, 0.7)}``.
    ``sample_repeats`` > 1 re-audits that many independent samples (seeds ``seed+1`` ...)
    when sampling applies and attaches ``value_range`` and ``verdict_stable`` to each
    finding, so a verdict that flips between samples is not mistaken for a property of the
    corpus. Costs ``sample_repeats`` times the audit time; ignored when nothing is sampled.
    """
    from vecscope import __version__

    th = resolve_thresholds(thresholds)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    k = int(k)
    if sample is not None:
        if isinstance(sample, bool) or not isinstance(sample, (int, np.integer)) or sample < 0:
            raise ValueError(f"sample must be None or a non-negative integer, got {sample!r}")
        sample = None if sample == 0 else int(sample)
        if sample is not None and sample <= k:
            raise ValueError(f"sample={sample} must exceed k={k} (0 audits every row)")
    if isinstance(clusters, bool) or not isinstance(clusters, (int, np.integer)) or clusters < 0:
        raise ValueError(f"clusters must be a non-negative integer, got {clusters!r}")
    if not (0.0 < float(near_dup_cos) <= 1.0):
        raise ValueError(f"near_dup_cos must be in (0, 1], got {near_dup_cos!r}")
    if (isinstance(sample_repeats, bool) or not isinstance(sample_repeats, (int, np.integer))
            or sample_repeats < 1):
        raise ValueError(f"sample_repeats must be a positive integer, got {sample_repeats!r}")
    sample_repeats = int(sample_repeats)

    x = np.asarray(x)
    if x.dtype == object or np.iscomplexobj(x):
        raise ValueError(f"embeddings must be real numbers, got dtype {x.dtype}")
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError(f"expected a 2-D array with at least 3 rows, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("embeddings contain NaN or Inf")
    n_total = int(x.shape[0])
    sampled = sample is not None and n_total > sample
    if sampled and ann_neighbors is not None:
        raise ValueError(
            f"ann_neighbors needs every row audited, but n={n_total} exceeds sample={sample}; "
            "pass sample=None (CLI: --sample 0) so row ids line up"
        )
    x_full = x
    if sampled:
        rows = np.sort(np.random.default_rng(seed).choice(n_total, sample, replace=False))
        x = x[rows]
    if not sampled:
        sample_repeats = 1
    if k >= x.shape[0]:
        raise ValueError(f"k={k} must be smaller than the number of rows ({x.shape[0]})")
    if ann_neighbors is not None:
        ann_neighbors = np.asarray(ann_neighbors)
        if ann_neighbors.ndim != 2 or ann_neighbors.shape[0] != x.shape[0]:
            raise ValueError(
                f"ann_neighbors must be shaped (n, k) with n={x.shape[0]} rows, got "
                f"{ann_neighbors.shape}"
            )
        if not np.issubdtype(ann_neighbors.dtype, np.integer):
            raise ValueError("ann_neighbors must hold integer row ids")
        if ann_neighbors.max() >= x.shape[0]:
            raise ValueError(
                f"ann_neighbors contains id {int(ann_neighbors.max())} but there are only "
                f"{x.shape[0]} rows; ids must index the same rows as the embedding file"
            )
    findings, knn_idx, knn_dist = _core_findings(x, k, near_dup_cos, queries, th)
    n_queries = 0 if queries is None else int(np.asarray(queries).shape[0])
    if ann_neighbors is not None:
        own = ann_neighbors == np.arange(x.shape[0])[:, None]
        if own.any():  # the index returned the item itself; it is not a neighbor
            ann_neighbors = np.where(own, -1, ann_neighbors)
        r = ann_recall(knn_idx, ann_neighbors)
        findings.append(
            Finding(
                "ann_recall", "recall_at_k", r["recall_at_k"],
                _verdict("ann_recall.recall_at_k", r["recall_at_k"], th),
                f"ANN recall@{r['k']} = {r['recall_at_k']:.3f}; "
                f"{r['queries_below_floor_ratio']:.1%} of queries fall below {r['floor']:.0%}.",
                r,
            )
        )

    unstable: list[str] = []
    if sample_repeats > 1:
        values: dict[str, list[float]] = {f.name: [f.value] for f in findings}
        verdicts: dict[str, set[str]] = {f.name: {f.verdict} for f in findings}
        for r in range(1, sample_repeats):
            rows_r = np.sort(np.random.default_rng(seed + r).choice(n_total, sample, replace=False))
            rep, _, _ = _core_findings(x_full[rows_r], k, near_dup_cos, queries, th)
            for f in rep:
                values[f.name].append(f.value)
                verdicts[f.name].add(f.verdict)
        for f in findings:
            f.value_range = [float(min(values[f.name])), float(max(values[f.name]))]
            f.verdict_stable = len(verdicts[f.name]) == 1
            if not f.verdict_stable:
                unstable.append(f.name)

    report = cluster_report(x, clusters, knn_idx, knn_dist, near_dup_cos, seed) if clusters else []
    return Evidence(
        "vecscope", __version__, int(x.shape[0]), int(x.shape[1]), int(k),
        _attach_thresholds(findings, th),
        n_total=n_total, sampled=sampled, n_queries=n_queries, clusters=report,
        thresholds={k: list(v) for k, v in th.items()},
        sample_repeats=sample_repeats, unstable_findings=unstable,
    )
