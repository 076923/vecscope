"""Evidence model: deterministic diagnostics with verdicts an agent can act on."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np

from vecscope._knn import check_magnitude, exact_knn, resolve_sample
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
from vecscope.incremental import fingerprint

# Default verdict thresholds: (warn, fail). Calibrated on seven public embedding models
# (see README "Real embeddings") and re-checked against 19 models x 3 labeled datasets
# (docs/verification.md); they are a starting point for diagnosis, not a universal quality
# bar. Override per call with ``audit(..., thresholds={key: (warn, fail)})``.
THRESHOLDS = {
    "hubness.skewness": (2.0, 4.0),  # real 1k-1.5k-dim text models sit at 1.1-1.6
    "hubness.antihub_excess": (0.05, 0.20),  # share never retrieved, above the exp(-k) baseline
    # good retrievers sit at 0.58-0.82 with no nDCG cost, so warn starts above them; 0.95 is
    # where scores are so compressed that absolute thresholds and float32 resolution suffer
    "anisotropy.mean_random_cosine": (0.85, 0.95),
    "anisotropy.effective_rank_ratio": (0.10, 0.05),  # lower is worse; real models 0.09-0.18
    "collapse.near_duplicate_ratio": (0.02, 0.10),
    "ann_recall.recall_at_k": (0.95, 0.85),  # lower is worse
    "rogue_dimensions.top1_dim_variance_ratio": (10.0, 50.0),  # x the uniform 1/dim share; real 1.5-3
    "norms.norm_cv": (0.10, 0.30),
    "metric_mismatch.topk_overlap": (0.90, 0.70),  # lower is worse
    "distance_concentration.contrast_over_noise": (1.4, 1.1),  # lower is worse; pure noise 1.0, real 1.43-2.9
    "outliers.outlier_ratio": (0.01, 0.05),
    "connectivity.island_ratio": (0.01, 0.10),
    # short queries against long passages are farther than passages are from each other by
    # design: good models score 0.0-0.65 on BEIR, a wrong model or language 0.7-1.0
    # (deprecated with the ood_queries finding)
    "queries.ood_query_ratio": (0.5, 0.8),
    # pipelines reproducing published nDCG sit at 0.94-2.52 (NFCorpus highest), queries from a
    # different model at a median of 5-8 (docs/verification.md, query findings)
    "queries.query_gap_ratio": (2.6, 3.0),
    # share beyond the reference median + 3 MAD: healthy query sets 0.00-0.09
    "queries.shifted_query_ratio": (0.20, 0.30),
    # good models: 1.5-2.9 on English BEIR, up to 9 on Korean; wrong-language models 11-30
    "queries.query_hubness_skewness": (4.0, 10.0),
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
    value_range: list[float] | None = None  # [min, max] over sample repeats: the spread
    # between samples, not an interval that contains the full-corpus value
    verdict_stable: bool | None = None  # verdict identical on every sample repeat
    inconclusive: bool = False  # sampled, and this verdict cannot stand for the full corpus


SCHEMA_VERSION = 1  # bumped when a key in the JSON is renamed, removed or changes meaning

WORSE = {"ok": 0, "warn": 1, "fail": 2}


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
    near_dup_cos: float = 0.99
    inconclusive_findings: list[str] = field(default_factory=list)
    # ordered steps for the non-ok findings (vecscope.plan): shared cause, first change
    plan: list[dict] = field(default_factory=list)
    # quality on labeled queries (vecscope.retrieval); empty without qrels
    retrieval: dict = field(default_factory=dict)
    # where the vectors came from when read from a vector store (vecscope.adapters)
    source: dict = field(default_factory=dict)
    # k-NN cache use for an incremental audit (vecscope.incremental)
    knn_cache: dict = field(default_factory=dict)
    # SHA-256 of every input row as float32, so a later run can tell whether a baseline was
    # made on the same corpus (or on an unchanged prefix of it)
    corpus_fingerprint: str = ""
    schema_version: int = SCHEMA_VERSION

    @property
    def verdict(self) -> str:
        return max((f.verdict for f in self.findings), key=WORSE.get, default="ok")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict
        return d

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


_RENAMED_KEYS = {
    "distance_concentration.relative_contrast": "distance_concentration.contrast_over_noise",
}


def _finding_direction(f: Mapping) -> bool | None:
    """Whether lower values of this finding are worse: from the finding itself, else from the
    default threshold key of its name, else unknown (None)."""
    if isinstance(f.get("lower_is_worse"), bool):
        return f["lower_is_worse"]
    key = _FINDING_KEYS.get(f.get("name"))
    return lower_is_worse(key) if key else None


def resolve_tolerances(tolerances: Mapping | None) -> dict:
    """Validate ``{finding name: allowed worsening}``: each value is a finite number >= 0 in
    the units of that finding's metric."""
    if tolerances is None:
        return {}
    if not isinstance(tolerances, Mapping):
        raise TypeError("tolerances must be a mapping of finding name -> allowed worsening")
    out = {}
    for name, tol in tolerances.items():
        if name not in _FINDING_KEYS:
            raise ValueError(f"unknown finding {name!r} in tolerances; known: {sorted(_FINDING_KEYS)}")
        if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not np.isfinite(tol) or tol < 0:
            raise ValueError(f"tolerance for {name!r} must be a finite number >= 0, got {tol!r}")
        out[name] = float(tol)
    return out


def diff_evidence(new: dict, old: dict, tolerances: Mapping | None = None) -> dict:
    """Compare two evidence dicts (``Evidence.to_dict()`` or a parsed ``--json`` file).

    Returns per-finding value and verdict changes, the findings whose verdict got worse
    (``regressions``), those that improved, and findings present on one side only. Each
    change row says whether the value moved in the worse direction (``worsened``) and by how
    much (``worsening``, positive = worse, in the metric's units), so a drift that stays
    inside one verdict band is visible too.

    ``tolerances`` maps a finding name to the worsening it may accumulate against the
    baseline, e.g. ``{"collapse": 0.01, "ann_recall": 0.02}``; findings beyond it are listed
    in ``value_regressions`` whether or not their verdict changed. Findings without a
    tolerance are never value regressions. Meant for CI: audit the new index, diff against
    the committed baseline, fail on ``regressions`` or ``value_regressions``; ``comparable``
    says whether n, dim, k, sampling, thresholds and the near-duplicate cutoff match, and a
    CI policy should refuse to read the diff when it is false.
    """
    tol = resolve_tolerances(tolerances)
    if not isinstance(new, Mapping) or not isinstance(old, Mapping):
        raise TypeError("evidence must be a mapping (Evidence.to_dict() or parsed --json output)")
    for name, ev in (("new", new), ("baseline", old)):
        if "findings" not in ev or not isinstance(ev["findings"], list):
            raise ValueError(f"{name} evidence has no findings list")
        if ev.get("schema_version", 1) != SCHEMA_VERSION:
            raise ValueError(
                f"{name} evidence has schema_version {ev.get('schema_version')}, this vecscope "
                f"writes {SCHEMA_VERSION}; re-create the baseline with this version"
            )
    for name, ev in (("new", new), ("baseline", old)):
        for f in ev["findings"]:
            if (not isinstance(f, Mapping) or not isinstance(f.get("name"), str)
                    or not isinstance(f.get("metric"), str)
                    or not isinstance(f.get("verdict"), str) or f["verdict"] not in WORSE
                    or isinstance(f.get("value"), bool)
                    or not isinstance(f.get("value"), (int, float))):
                raise ValueError(f"{name} evidence has a malformed finding: {f!r:.80}")
    a = {f["name"]: f for f in new["findings"]}
    b = {f["name"]: f for f in old["findings"]}
    changes, regressions, improvements, value_regressions = [], [], [], []
    for name in a.keys() & b.keys():
        fa, fb = a[name], b[name]
        delta = fa["value"] - fb["value"]
        low = _finding_direction(fa)
        worsening = None if low is None else (-delta if low else delta)
        row = {"name": name, "metric": fa["metric"], "value": fa["value"], "baseline_value": fb["value"],
               "delta": delta, "lower_is_worse": low,
               "worsened": None if worsening is None else bool(worsening > 0),
               "worsening": worsening, "tolerance": tol.get(name),
               "verdict": fa["verdict"], "baseline_verdict": fb["verdict"]}
        changes.append(row)
        if WORSE[fa["verdict"]] > WORSE[fb["verdict"]]:
            regressions.append(name)
        elif WORSE[fa["verdict"]] < WORSE[fb["verdict"]]:
            improvements.append(name)
        if name in tol and worsening is not None and worsening > tol[name]:
            value_regressions.append(name)
    order = [f["name"] for f in new["findings"]]
    changes.sort(key=lambda r: order.index(r["name"]))
    retrieval_changes = []
    ra, rb = new.get("retrieval") or {}, old.get("retrieval") or {}
    for part in ("exact", "ranked"):
        pa, pb = ra.get(part) or {}, rb.get(part) or {}
        for metric in pa.keys() & pb.keys():
            retrieval_changes.append({"ranking": part, "metric": metric, "value": pa[metric],
                                      "baseline_value": pb[metric],
                                      "delta": pa[metric] - pb[metric]})
    retrieval_changes.sort(key=lambda r: (r["ranking"], r["metric"]))
    return {
        "verdict": new.get("verdict"),
        "baseline_verdict": old.get("verdict"),
        "regressions": regressions,
        "value_regressions": sorted(value_regressions, key=order.index),
        "tolerances": tol,
        "improvements": improvements,
        "only_in_new": sorted(a.keys() - b.keys()),
        "only_in_baseline": sorted(b.keys() - a.keys()),
        "comparable": new.get("n") == old.get("n") and new.get("dim") == old.get("dim")
        and new.get("k") == old.get("k") and new.get("sampled") == old.get("sampled")
        and new.get("thresholds") == old.get("thresholds")
        and new.get("near_dup_cos") == old.get("near_dup_cos"),
        "changes": changes,
        "retrieval_changes": retrieval_changes,
        "retrieval_comparable": bool(ra and rb)
        and ra.get("qrels_fingerprint") == rb.get("qrels_fingerprint") and ra.get("k") == rb.get("k"),
    }


def _apply_sampling(findings: list[Finding], fraction: float, th: dict) -> list[str]:
    """Make a sampled audit say what each value can stand for, and keep a verdict that the
    sample cannot support from reading as a clean bill. Measured on synthetic corpora at
    200k x 384 with 100k/50k/20k samples (docs/verification.md) and in
    tests/test_regression_injection.py:

    - hubness and antihubs fall as the sample shrinks (k-occurrence concentrates with
      density), so a sampled value is a lower bound: a full-corpus ``warn`` read ``ok`` on a
      25% sample. Their verdict is kept but marked ``inconclusive`` unless it is ``fail``.
    - a duplicate pair is seen only when both rows were drawn, so the sampled collapse share
      is about ``fraction`` times the full one (0.96 to 1.03 of it in the measurements). The
      verdict is taken from the estimated full share ``value / fraction`` (in details), so a
      full-corpus ``fail`` does not read ``warn``.
    - islands are those of the sample's k-NN graph: a corpus with 37% of its rows in islands
      scored 0.08% on a 25% sample. connectivity is ``inconclusive`` unless it is ``fail``.
    - outliers are over-counted on small samples (3.9x at 10%), so a sampled ``warn``/``fail``
      is ``inconclusive``; ``ok`` holds.
    - anisotropy, effective_rank, rogue_dimensions, norms, metric_mismatch and
      distance_concentration stayed within 2.3% of the full value; nothing changes for them.

    Returns the names marked inconclusive."""
    pct = f"{fraction:.0%}"
    out = []
    for f in findings:
        if f.name in ("hubness", "antihubs"):
            f.explanation += (f" Measured on a {pct} sample: hubness grows with density, so this "
                              "is a lower bound for the full corpus (a full-corpus warn can read "
                              "ok here); compare only runs with the same sample size.")
            if f.verdict != "fail":
                f.inconclusive = True
        elif f.name == "collapse":
            est = min(1.0, f.value / fraction)
            f.details["estimated_full_share"] = est
            f.details["sample_fraction"] = fraction
            f.verdict = _verdict("collapse.near_duplicate_ratio", est, th)
            f.explanation += (f" Measured on a {pct} sample: a duplicate pair is seen only when "
                              f"both rows were sampled, so the full-corpus share is estimated at "
                              f"{est:.1%} (this value divided by {fraction:.2f}) and the verdict "
                              "uses that estimate; audit every row (sample=0) to count duplicates.")
        elif f.name == "outliers":
            f.explanation += (f" Measured on a {pct} sample: small samples over-count outliers, "
                              "and details.sample_ids lists only those inside the sample.")
            if f.verdict != "ok":
                f.inconclusive = True
        elif f.name == "connectivity":
            f.explanation += (f" Measured on a {pct} sample: islands are those of the sample's "
                              "k-NN graph and most full-corpus islands are not visible in it; "
                              "audit every row (sample=0) to judge connectivity.")
            if f.verdict != "fail":
                f.inconclusive = True
        if f.inconclusive:
            f.explanation += (" This verdict is inconclusive for the full corpus and is not "
                              "evidence that the finding is healthy.")
            out.append(f.name)
    return out


def _to_original_rows(findings: list[Finding], clusters: list[dict], rows: np.ndarray) -> None:
    """Row ids in details index the audited subsample; rewrite them as ids of the input."""
    for f in findings:
        if "sample_ids" in f.details:
            f.details["sample_ids"] = [int(rows[i]) for i in f.details["sample_ids"]]
        if "sample_pairs" in f.details:
            f.details["sample_pairs"] = [[int(rows[a]), int(rows[b]), cos]
                                         for a, b, cos in f.details["sample_pairs"]]
    for c in clusters:
        c["sample_ids"] = [int(rows[i]) for i in c["sample_ids"]]


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
        if isinstance(pair, (str, bytes, Mapping)) or not hasattr(pair, "__len__") or len(pair) != 2:
            raise ValueError(f"threshold {key!r} must be a (warn, fail) pair")
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


def lower_is_worse(key: str) -> bool:
    """Direction of a threshold key, fixed by the defaults (an override may set warn == fail)."""
    warn, fail = THRESHOLDS[key]
    return warn > fail


def _verdict(key: str, value: float, th: dict) -> str:
    warn, fail = th[key]
    if lower_is_worse(key):
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
        ("re-rank top-k with local scaling against the queries (CSLS: 2 cos(q, d) minus d's mean "
         "cosine to its 10 nearest queries; scaling by corpus neighbors raised hubness in the "
         "verification); if anisotropy also fails, center first"),
        ("re-run the audit on the transformed vectors and compare hubness.skewness and "
        "details.hub_share_top1pct; check that recall on held-out queries did not drop"),
    ),
    "antihubs": (
        ["same causes as hubness", "items embedded from empty or near-empty text"],
        "inspect a sample of never-retrieved items; drop empty inputs, then apply the hubness fix",
        ("antihub_excess (antihub_ratio minus the random baseline) should fall below warn_at "
         "after the fix; spot-check that the dropped "
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
        ("top1_dim_variance_ratio falls below warn_at; retrieval quality on real queries "
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
        ("deprecated: read query_gap (and query_shift with a baseline) instead; confirm the "
        "same model and prefix on both sides; inspect details.ood_sample_ids"),
        ("ood_query_ratio below warn_at after fixing the query pipeline; otherwise extend "
        "the corpus"),
    ),
    "query_gap": (
        ["queries embedded with a different model (or model version) than the corpus",
         "a weak model for this domain: query and passage distances both grow",
         "very short queries against very long passages"],
        ("embed a handful of queries with the corpus model and compare query_gap_ratio; if "
        "it drops, the query pipeline uses another model. A missing query prefix does not "
        "move this number reliably, check the prefix in code"),
        ("query_gap_ratio below warn_at after fixing the query pipeline, and nDCG or recall "
        "on labeled queries back to the reference"),
    ),
    "query_shift": (
        ["the query pipeline changed since the reference (model, version, preprocessing)",
         "new query traffic on topics the corpus does not cover",
         "the corpus changed under the same queries"],
        ("inspect details.shifted_sample_ids; if most queries moved, diff the query "
        "pipeline against the one that produced the reference; if a few did, check their "
        "topics against the corpus"),
        ("shifted_query_ratio below warn_at against the same reference; labeled retrieval "
        "quality unchanged"),
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
    "query_gap": "queries.query_gap_ratio",
    "query_shift": "queries.shifted_query_ratio",
    "query_hubness": "queries.query_hubness_skewness",
}


def _attach_thresholds(findings: list[Finding], th: dict) -> list[Finding]:
    for f in findings:
        warn, fail = th[_FINDING_KEYS[f.name]]
        f.warn_at, f.fail_at, f.lower_is_worse = warn, fail, lower_is_worse(_FINDING_KEYS[f.name])
        causes, action, verify = ACTIONS[f.name]
        f.likely_causes, f.suggested_action, f.verify_by = list(causes), action, verify
        if f.lower_is_worse:
            f.healthy_range = f"ok >= {warn:g}, warn < {warn:g}, fail < {fail:g}"
        else:
            f.healthy_range = f"ok < {warn:g}, warn >= {warn:g}, fail >= {fail:g}"
    return findings


def _checked_queries(queries, dim: int) -> np.ndarray:
    """Validate query vectors like the corpus, before any expensive pass runs."""
    qv = np.asarray(queries)
    if qv.dtype.kind not in "iufb":
        raise ValueError(f"queries must be real numbers, got dtype {qv.dtype}")
    if qv.ndim != 2 or qv.shape[1] != dim:
        raise ValueError(f"queries must be (m, {dim}) to match the corpus")
    if qv.shape[0] == 0:
        raise ValueError("queries has no rows")
    if not np.isfinite(qv).all():
        raise ValueError("queries contain NaN or Inf")
    check_magnitude(qv, "queries")
    return np.asarray(qv, dtype=np.float32)


def same_corpus(baseline: Mapping, x: np.ndarray) -> bool:
    """Whether ``baseline`` evidence was audited on ``x`` or on an unchanged prefix of it
    (rows appended since). Query distances from such a baseline are a valid reference for
    query_shift; after a transform (normalizing, centering, another model) they are not."""
    fp = baseline.get("corpus_fingerprint") if isinstance(baseline, Mapping) else None
    n_base = int(baseline.get("n_total") or baseline.get("n") or 0) if fp else 0
    if not fp or not 0 < n_base <= x.shape[0]:
        return False
    return fingerprint(x[:n_base]) == fp


def query_reference(baseline: Mapping) -> tuple[float, float]:
    """(median, MAD) of the query->corpus nearest distance from earlier evidence (an
    ``Evidence.to_dict()`` or ``--json`` output of an audit run with queries), for ``audit(..., query_baseline=...)``."""
    if not isinstance(baseline, Mapping) or not isinstance(baseline.get("findings"), list):
        raise TypeError("query_baseline must be evidence with a findings list "
                        "(Evidence.to_dict() or --json output)")
    for f in baseline["findings"]:
        if isinstance(f, Mapping) and f.get("name") == "query_gap":
            d = f.get("details") or {}
            try:
                med, mad = float(d["median_query_nn_distance"]), float(d["query_nn_mad"])
            except (KeyError, TypeError, ValueError):
                break
            if np.isfinite(med) and np.isfinite(mad) and med >= 0 and mad >= 0:
                return med, mad
            break
    raise ValueError("query_baseline has no query_gap finding with median_query_nn_distance "
                     "and query_nn_mad; re-create it with --queries")


def _core_findings(
    x: np.ndarray, k: int, near_dup_cos: float, queries: np.ndarray | None, th: dict,
    seed: int = 0, reference: tuple[float, float] | None = None,
    knn: Mapping | None = None,
) -> tuple[list[Finding], np.ndarray, np.ndarray]:
    """Compute every finding that depends only on the (sampled) matrix and the queries.
    ``knn`` supplies exact neighbor lists ({"cosine": (idx, dist), "dot": ...}) computed
    elsewhere, e.g. by an incremental cache."""
    knn_idx, knn_dist = knn["cosine"] if knn else exact_knn(x, k)
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

    a = anisotropy(x, seed=seed)
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
            f"effective rank {a['effective_rank']:.1f} of {a['max_attainable_rank']} attainable "
            + (f"({a['dim']} dims" if a["max_attainable_rank"] == a["dim"] else
               f"({a['dim']} dims, but only {a['max_attainable_rank'] + 1} rows were used, which "
               "bounds the rank")
            + f"; {a['effective_rank_ratio']:.1%}); low values mean most dimensions carry nothing.",
            {key: a[key] for key in ("dim", "effective_rank", "max_attainable_rank",
                                     "effective_rank_ratio", "top1_pc_variance_share")},
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
            f"or an exact copy (exact duplicates {c['exact_duplicate_ratio']:.1%}); {shape}",
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
        dot_idx, _ = knn["dot"] if knn and knn.get("dot") is not None else exact_knn(x, k, metric="dot")
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

    dc = distance_concentration(knn_dist[:, 0], a["mean_random_cosine"], dim=x.shape[1],
                                seed=seed, x=x)
    findings.append(
        Finding(
            "distance_concentration", "contrast_over_noise", dc["contrast_over_noise"],
            _verdict("distance_concentration.contrast_over_noise", dc["contrast_over_noise"], th),
            (f"on {dc['baseline_rows']} rows a random item is "
             f"{dc['relative_contrast_at_baseline_rows']:.2f}x farther than the nearest neighbor "
             f"(random distance {dc['mean_random_distance']:.3f}); isotropic noise at "
             f"{x.shape[1]}-d and the same rows would give {dc['noise_relative_contrast']:.2f}x, "
             f"so the contrast is {dc['contrast_over_noise']:.2f}x that of noise. On all rows the "
             f"contrast is {dc['relative_contrast']:.2f}x (median NN distance "
             f"{dc['median_nn_distance']:.3f}). Near 1.0 means distances carry no signal."
             + ("" if dc["distinct_neighbors"] else " No row has a neighbor that is not a copy "
                "of it, so the nearest-neighbor distance is 0 everywhere; see collapse.")),
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
        qa = query_audit(x, qv, k, knn_dist[:, 0],
                         dup_cutoff=duplicate_cutoff(dc["mean_random_distance"]),
                         reference=reference)
        findings.append(
            Finding(
                "ood_queries", "ood_query_ratio", qa["ood_query_ratio"],
                _verdict("queries.ood_query_ratio", qa["ood_query_ratio"], th),
                f"{qa['ood_query_ratio']:.1%} of {qa['n_queries']} queries are farther from "
                f"their nearest corpus item (median {qa['median_query_nn_distance']:.3f}) than "
                f"99% of corpus items are from theirs (cutoff {qa['ood_cutoff_distance']:.3f}). "
                "Out-of-distribution queries: wrong model, missing query prefix, or uncovered "
                "topics; see details.ood_sample_ids. Deprecated: this ratio also counts the "
                "normal query/passage distance of asymmetric models; read query_gap instead."
                + ("" if qa["corpus_has_distinct_neighbors"] else " No corpus row has a "
                   "neighbor that is not a copy of it, so there is no reference distance and "
                   "nothing is flagged; see collapse."),
                {key: v for key, v in qa.items() if not key.startswith(("shift", "reference_"))},
            )
        )
        findings.append(
            Finding(
                "query_gap", "query_gap_ratio", qa["query_gap_ratio"],
                _verdict("queries.query_gap_ratio", qa["query_gap_ratio"], th),
                f"the median query is {qa['median_query_nn_distance']:.3f} from its nearest "
                f"corpus item, {qa['query_gap_ratio']:.2f}x the median corpus item's distance to "
                f"its own nearest neighbor ({qa['median_corpus_nn_distance']:.3f}). Short "
                "queries against long passages keep healthy pipelines at about 1 to 2.5; queries "
                "embedded by a different model land at 5 to 10. A missing query prefix does not "
                "reliably move it."
                + ("" if qa["corpus_has_distinct_neighbors"] else " No corpus row has a "
                   "neighbor that is not a copy of it, so there is no reference distance and "
                   "the ratio is reported as 0; see collapse."),
                {key: qa[key] for key in ("n_queries", "median_query_nn_distance",
                                          "query_nn_mad", "median_corpus_nn_distance",
                                          "corpus_has_distinct_neighbors")},
            )
        )
        if reference is not None:
            findings.append(
                Finding(
                    "query_shift", "shifted_query_ratio", qa["shifted_query_ratio"],
                    _verdict("queries.shifted_query_ratio", qa["shifted_query_ratio"], th),
                    f"{qa['shifted_query_ratio']:.1%} of {qa['n_queries']} queries are farther "
                    f"from the corpus than {qa['shift_cutoff_distance']:.3f}, the reference "
                    f"query set's median ({qa['reference_median_nn_distance']:.3f}) plus 3 MAD "
                    f"({qa['reference_nn_mad']:.3f}). Healthy query sets stay below 10%; a "
                    "query pipeline that changed model moves almost every query. Only "
                    "meaningful when the reference was audited on the same corpus and sample "
                    "size; see details.shifted_sample_ids.",
                    {key: qa[key] for key in ("n_queries", "median_query_nn_distance",
                                              "shift_cutoff_distance",
                                              "reference_median_nn_distance",
                                              "reference_nn_mad", "shifted_sample_ids")},
                )
            )
        findings.append(
            Finding(
                "query_hubness", "query_hubness_skewness", qa["query_hubness_skewness"],
                _verdict("queries.query_hubness_skewness", qa["query_hubness_skewness"], th),
                f"k-occurrence skewness over query results {qa['query_hubness_skewness']:.2f} "
                f"(max {qa['max_query_k_occurrence']} appearances); queries reach "
                f"{qa['coverage']:.1%} of the corpus in their top-{qa['k']}. Skewness grows "
                "with corpus size: English models with published retrieval scores measured 1.5 "
                "to 3.4 on corpora up to 26k documents and 4.0 to 4.8 on 58k, good multilingual "
                "models on Korean up to 9.1, models that do not cover the language 11 to 30; "
                "compare against a baseline of the same corpus.",
                {key: qa[key] for key in ("k", "n_queries", "max_query_k_occurrence", "coverage")},
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
    query_baseline: Mapping | None = None,
    qrels: Mapping | None = None,
    ranked: np.ndarray | None = None,
    knn: Mapping | None = None,
) -> Evidence:
    """Audit an embedding matrix and return deterministic Evidence.

    Exact k-NN is O(n^2); when ``sample`` is set and ``n > sample``, a seeded random
    subset of rows is audited instead. ``ann_neighbors`` requires every row (its ids index
    the rows), so combining it with sampling raises. Pass ``sample=None`` (or 0) to audit
    every row.

    ``queries`` adds query-side findings (query gap, query hubness, and the deprecated
    out-of-distribution ratio). ``query_baseline`` is earlier evidence of the same corpus
    audited with healthy queries (``Evidence.to_dict()`` or ``--json`` output); it adds
    ``query_shift``, the share of these queries beyond that reference's median + 3 MAD.
    ``qrels`` ({query row: {corpus row: grade}}, see ``vecscope.retrieval``) adds the
    ``retrieval`` block: nDCG, recall, MRR and duplicate share of exact search over every
    corpus row (also when the findings are sampled), and of ``ranked`` (production top-k
    row ids per query) when given. ``knn`` supplies the exact cosine (and dot) neighbor
    lists of every row, e.g. from ``vecscope.incremental.cached_knn``; it needs an unsampled
    audit. ``plan`` orders the non-ok findings into steps (``vecscope.plan``).
    ``clusters`` > 0 attaches a per-cluster report with sample ids (no verdicts).
    ``thresholds`` overrides default (warn, fail) pairs by key, e.g.
    ``{"anisotropy.mean_random_cosine": (0.4, 0.7)}``.
    ``sample_repeats`` > 1 re-audits that many independent samples (seeds ``seed+1`` ...)
    when sampling applies and attaches ``value_range`` and ``verdict_stable`` to each
    finding, so a verdict that flips between samples is not mistaken for a property of the
    corpus. ``value_range`` is the spread between samples, not an interval that contains the
    full-corpus value: hubness is a lower bound on every sample, and a finding whose samples
    agree to three decimals can still sit a little away from the full value. Costs
    ``sample_repeats`` times the audit time; ignored when nothing is sampled.

    When sampling applies, hubness, antihubs and connectivity verdicts other than ``fail``
    and outlier verdicts other than ``ok`` are marked ``inconclusive`` (listed in
    ``inconclusive_findings``), because the sample systematically under- or over-states them;
    the collapse verdict is taken from the estimated full share ``value / fraction``.
    """
    from vecscope import __version__

    th = resolve_thresholds(thresholds)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    k = int(k)
    sample = resolve_sample(sample, k)
    if isinstance(clusters, bool) or not isinstance(clusters, (int, np.integer)) or clusters < 0:
        raise ValueError(f"clusters must be a non-negative integer, got {clusters!r}")
    if not (0.0 < float(near_dup_cos) <= 1.0):
        raise ValueError(f"near_dup_cos must be in (0, 1], got {near_dup_cos!r}")
    near_dup_cos = float(near_dup_cos)
    if (isinstance(sample_repeats, bool) or not isinstance(sample_repeats, (int, np.integer))
            or sample_repeats < 1):
        raise ValueError(f"sample_repeats must be a positive integer, got {sample_repeats!r}")
    sample_repeats = int(sample_repeats)

    x = np.asarray(x)
    if x.dtype.kind not in "iufb":
        raise ValueError(f"embeddings must be real numbers, got dtype {x.dtype}")
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 1:
        raise ValueError(f"expected a 2-D array with at least 3 rows and 1 column, got shape {x.shape}")
    if x.dtype.kind == "b":
        x = x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("embeddings contain NaN or Inf")
    check_magnitude(x)  # before the float32 cast, so float64 values beyond its range are named
    x = np.asarray(x, dtype=np.float32)
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
        if ann_neighbors.shape[1] == 0:
            raise ValueError("ann_neighbors has no neighbor columns; pass at least one id per row")
        if ann_neighbors.max() >= x.shape[0]:
            raise ValueError(
                f"ann_neighbors contains id {int(ann_neighbors.max())} but there are only "
                f"{x.shape[0]} rows; ids must index the same rows as the embedding file"
            )
    reference = None
    if query_baseline is not None:
        if queries is None:
            raise ValueError("query_baseline needs queries")
        reference = query_reference(query_baseline)
        if query_baseline.get("dim") not in (None, x.shape[1]):
            raise ValueError(f"query_baseline was audited at dim={query_baseline.get('dim')}, "
                             f"these vectors have dim={x.shape[1]}")
    if queries is not None:
        queries = _checked_queries(queries, x.shape[1])
    if (qrels is not None or ranked is not None) and queries is None:
        raise ValueError("qrels and ranked need queries")
    if ranked is not None and qrels is None:
        raise ValueError("ranked needs qrels to be scored")
    if knn is not None:
        if sampled:
            raise ValueError("precomputed knn needs every row audited; pass sample=None")
        ci = np.asarray(knn["cosine"][0])
        if ci.shape[0] != x.shape[0] or ci.shape[1] < min(k, x.shape[0] - 1):
            raise ValueError(f"knn lists are shaped {ci.shape}; need ({x.shape[0]}, >= {k})")
        knn = {m: (np.asarray(v[0])[:, :k], np.asarray(v[1])[:, :k])
               for m, v in knn.items() if v is not None}
    findings, knn_idx, knn_dist = _core_findings(x, k, near_dup_cos, queries, th, seed,
                                                 reference, knn)
    n_queries = 0 if queries is None else int(np.asarray(queries).shape[0])
    if ann_neighbors is not None:
        ann_neighbors = ann_neighbors.astype(np.int64)  # ids were checked to fit; -1 marks fillers
        own = ann_neighbors == np.arange(x.shape[0])[:, None]
        if own.any():  # the index returned the item itself; it is not a neighbor
            ann_neighbors = np.where(own, -1, ann_neighbors)
        r = ann_recall(knn_idx, ann_neighbors, exact_dist=knn_dist, vectors=x)
        short = r["rows_with_fewer_than_k_ids_ratio"]
        note = ("" if short == 0 else
                f" {short:.1%} of rows supplied fewer than {r['k']} valid ids (self ids are "
                f"dropped), which caps their recall below 1; query the index with k + 1.")
        findings.append(
            Finding(
                "ann_recall", "recall_at_k", r["recall_at_k"],
                _verdict("ann_recall.recall_at_k", r["recall_at_k"], th),
                f"ANN recall@{r['k']} = {r['recall_at_k']:.3f}; "
                f"{r['queries_below_floor_ratio']:.1%} of queries fall below {r['floor']:.0%}."
                + note,
                r,
            )
        )

    unstable: list[str] = []
    if sample_repeats > 1:
        values: dict[str, list[float]] = {f.name: [f.value] for f in findings}
        verdicts: dict[str, set[str]] = {f.name: {f.verdict} for f in findings}
        for r in range(1, sample_repeats):
            rows_r = np.sort(np.random.default_rng(seed + r).choice(n_total, sample, replace=False))
            rep, _, _ = _core_findings(x_full[rows_r], k, near_dup_cos, queries, th, seed,
                                       reference)
            for f in rep:
                values[f.name].append(f.value)
                verdicts[f.name].add(f.verdict)
        for f in findings:
            f.value_range = [float(min(values[f.name])), float(max(values[f.name]))]
            f.verdict_stable = len(verdicts[f.name]) == 1
            if not f.verdict_stable:
                unstable.append(f.name)

    retrieval = {}
    if qrels is not None:
        from vecscope.retrieval import evaluate_retrieval, parse_qrels

        retrieval = evaluate_retrieval(x_full, queries, parse_qrels(qrels), k=k, ranked=ranked,
                                       near_dup_cos=near_dup_cos)
    report = cluster_report(x, clusters, knn_idx, knn_dist, near_dup_cos, seed) if clusters else []
    inconclusive: list[str] = []
    if sampled:
        _to_original_rows(findings, report, rows)
        inconclusive = _apply_sampling(findings, sample / n_total, th)
    from vecscope.plan import build_plan

    findings = _attach_thresholds(findings, th)
    return Evidence(
        "vecscope", __version__, int(x.shape[0]), int(x.shape[1]), int(k), findings,
        n_total=n_total, sampled=sampled, n_queries=n_queries, clusters=report,
        thresholds={k: list(v) for k, v in th.items()},
        sample_repeats=sample_repeats, unstable_findings=unstable, near_dup_cos=near_dup_cos,
        inconclusive_findings=inconclusive,
        plan=build_plan([asdict(f) for f in findings], retrieval or None),
        retrieval=retrieval,
        corpus_fingerprint=fingerprint(x_full),
    )
