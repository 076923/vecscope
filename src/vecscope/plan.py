"""Order the non-ok findings into steps: which findings share a likely cause, what to change
first, and what to re-measure.

Several findings often fail for one reason (a shared direction raises anisotropy, rogue
dimensions and hubness together; unnormalized vectors raise norms and metric_mismatch). The
rules below are fixed and deterministic: each group lists the findings it explains, the
common cause to check, the smallest change that tests it, and the numbers that must move.
The order follows dependencies between fixes: broken inputs distort every statistic, a
dominant coordinate decides every distance (and with it the norms, outliers and hubs), the
metric decides which geometry matters, a vector transform changes the neighbor graph that
the corpus and index findings are computed on, and the query side is judged last against
the corrected corpus. A group is a hypothesis to test, not a diagnosis.
"""

from __future__ import annotations

# (group, members, common cause, first change, re-measure, why at this position)
GROUPS = [
    ("input_integrity", ["norms", "collapse", "outliers", "query_gap"],
     ("rows that are not real embeddings of distinct items: zero or constant vectors from "
     "failed embedding calls, duplicated documents or boilerplate chunks"),
     ("read norms.details.zero_norm_ratio, collapse.details.largest_near_duplicate_groups and "
     "outliers.details.sample_ids; drop zero and constant rows and deduplicate before "
     "changing anything else"),
     ("zero_norm_ratio 0, near_duplicate_ratio and outlier_ratio below warn_at; "
     "retrieval duplicate_result_share falls; query_gap, if listed here, read again"),
     "broken rows distort every other statistic, so later steps would be measured on noise"),
    ("dominant_dimension", ["rogue_dimensions", "norms", "metric_mismatch", "outliers",
                            "effective_rank", "hubness", "antihubs", "collapse"],
     ("a few coordinates carry most of the variance, so they decide every distance, the "
     "vector lengths and which items look like outliers or hubs"),
     ("z-score each dimension with the corpus mean and standard deviation (or drop "
     "rogue_dimensions.details.rogue_dims), then L2-normalize; apply the same transform to "
     "queries and rebuild the index. Normalizing alone does not remove a dominant coordinate"),
     ("top1_dim_variance_ratio below warn_at, and the norms, metric and hub findings it moved "
     "back with it; retrieval ndcg_at_k up"),
     ("one coordinate decides every distance, so the other geometry findings measure it "
     "rather than the model")),
    ("metric", ["norms", "metric_mismatch"],
     "vector norms vary, so cosine and inner product rank differently",
     "decide the index metric; L2-normalize if cosine is intended",
     "norm_cv near 0 and topk_overlap 1.0 on re-audit",
     "the metric decides which geometry the remaining findings should be read in"),
    ("shared_direction", ["anisotropy", "hubness", "antihubs", "effective_rank", "collapse"],
     ("a large component shared by all vectors (a mean offset) that compresses scores and "
     "creates hubs"),
     ("subtract the corpus mean and re-normalize (anisotropy.details shows the expected "
     "cosine after centering); apply the same transform to queries and rebuild the index"),
     ("anisotropy below warn_at, hubness.skewness down; retrieval ndcg_at_k not lower than "
     "before (centering healthy models cost up to 0.034 nDCG@10 in the verification)"),
     ("a transform rebuilds the neighbor graph that hubness, connectivity and ann_recall are "
     "computed on")),
    ("weak_signal", ["distance_concentration", "effective_rank"],
     ("the model separates this corpus poorly: nearest neighbors are barely closer than "
     "random items"),
     ("compare a different model or a domain-tuned one on a labeled sample; add a re-ranker "
     "over a larger candidate set"),
     "contrast_over_noise above warn_at and retrieval ndcg_at_k up on the same qrels",
     "a model change invalidates every later measurement, so decide it before tuning the index"),
    ("corpus_structure", ["hubness", "antihubs", "connectivity"],
     "a few generic items near the corpus center, or small groups disconnected from the rest",
     ("inspect connectivity.details.sample_ids and the hubs; deduplicate or merge islands, "
     "re-rank with local scaling against the queries (CSLS) for hubs"),
     ("hubness.skewness and island_ratio below warn_at; recall on items from former islands "
     "comparable to the rest"),
     "corpus fixes change what the index has to find"),
    ("index", ["ann_recall"],
     ("the approximate index misses true neighbors: ef_search / nprobe too low or the graph "
     "too sparse"),
     ("run `vecscope sweep` and raise ef_search (HNSW) or nprobe (IVF) to the recommended "
     "setting before rebuilding with larger M"),
     "recall_at_k at the target; retrieval ranked.overlap_with_exact and ranked.ndcg_at_k up",
     "the index is tuned on the corrected corpus, otherwise the sweep measures the old one"),
    ("query_pipeline", ["query_gap", "query_shift", "ood_queries", "query_hubness"],
     ("queries are not embedded like the corpus: another model or version, or a model that "
     "does not cover the query language"),
     ("embed a few queries with the corpus model and compare query_gap; check the model name, "
     "version and query prefix in the code that embeds queries"),
     "query_gap and query_shift below warn_at; retrieval ndcg_at_k back to the reference",
     "query findings are measured against the corpus, so they are read after it is fixed"),
]

# a finding is explained by the first group whose condition holds
_CONDITION = {
    # zero vectors make norms an integrity problem; otherwise it is a metric question
    ("input_integrity", "norms"): lambda f: f["norms"]["details"].get("zero_norm_ratio", 0) > 0,
    # a strong shared direction alone pushes every cosine past the near-duplicate cutoff;
    # without exact copies that collapse is the direction's, and deduplicating would not help
    ("input_integrity", "collapse"): lambda f: (
        f["collapse"]["details"].get("exact_duplicate_ratio", 0) > 0
        or not _bad(f, "anisotropy", "rogue_dimensions")),
    # near-copies shrink the corpus nearest-neighbor distance that query_gap divides by, so
    # the gap is re-read after deduplicating before the queries are blamed
    ("input_integrity", "query_gap"): lambda f: (
        _bad(f, "collapse") and _CONDITION[("input_integrity", "collapse")](f)),
    # a dominant coordinate makes far rows look like outliers
    ("input_integrity", "outliers"): lambda f: not _bad(f, "rogue_dimensions"),
    # everything a dominant coordinate moves belongs to it when it is measured
    **{("dominant_dimension", n): (lambda f: _bad(f, "rogue_dimensions"))
       for n in ("norms", "metric_mismatch", "outliers", "effective_rank", "hubness",
                 "antihubs", "collapse")},
    # hubs and a low rank follow a shared direction only when one is measured
    ("shared_direction", "hubness"): lambda f: _bad(f, "anisotropy"),
    ("shared_direction", "antihubs"): lambda f: _bad(f, "anisotropy"),
    ("shared_direction", "effective_rank"): lambda f: _bad(f, "anisotropy"),
    # query hubness with a corpus hubness problem is the corpus's; alone it is the queries'
    ("query_pipeline", "query_hubness"): lambda f: not _bad(f, "hubness"),
}


def _bad(f: dict, *names: str) -> bool:
    return any(n in f and f[n]["verdict"] != "ok" for n in names)


def build_plan(findings: list[dict], retrieval: dict | None = None) -> list[dict]:
    """Steps for the non-ok findings, in fix order. ``findings`` are finding dicts (name,
    value, verdict, details, inconclusive); ``retrieval`` is the retrieval block, named in
    each step's re-measure line when present."""
    by = {f["name"]: f for f in findings}
    pending = [f["name"] for f in findings if f["verdict"] != "ok"]
    taken: set[str] = set()
    steps = []
    for group, members, cause, change, verify, why in GROUPS:
        names = []
        for name in members:
            if name not in pending or name in taken:
                continue
            cond = _CONDITION.get((group, name))
            if cond is not None and not cond(by):
                continue
            names.append(name)
        if not names:
            continue
        taken.update(names)
        evidence = [{"name": n, "value": by[n]["value"], "verdict": by[n]["verdict"],
                     "inconclusive": bool(by[n].get("inconclusive"))} for n in names]
        conclusive = [e for e in evidence if not e["inconclusive"]]
        steps.append({
            "order": len(steps) + 1,
            "group": group,
            "findings": evidence,
            "common_cause": cause,
            "first_change": change,
            "re_measure": verify + ("" if retrieval else
                                    "; pass --qrels to measure retrieval quality as well"),
            "why_this_order": why,
            "only_inconclusive": not conclusive,
        })
    # query hubness next to a corpus hubness problem joins the step that explains hubness
    if "query_hubness" in pending and "query_hubness" not in taken:
        for s in steps:
            if any(e["name"] == "hubness" for e in s["findings"]):
                f = by["query_hubness"]
                s["findings"].append({"name": "query_hubness", "value": f["value"],
                                      "verdict": f["verdict"],
                                      "inconclusive": bool(f.get("inconclusive"))})
                s["only_inconclusive"] = all(e["inconclusive"] for e in s["findings"])
                taken.add("query_hubness")
                break
    # anything no rule placed gets a step of its own from its ACTIONS entry
    left = [n for n in pending if n not in taken]
    for n in left:
        steps.append({
            "order": len(steps) + 1, "group": n,
            "findings": [{"name": n, "value": by[n]["value"], "verdict": by[n]["verdict"],
                          "inconclusive": bool(by[n].get("inconclusive"))}],
            "common_cause": "; ".join(by[n].get("likely_causes", [])),
            "first_change": by[n].get("suggested_action", ""),
            "re_measure": by[n].get("verify_by", ""),
            "why_this_order": "no shared cause with the findings above",
            "only_inconclusive": bool(by[n].get("inconclusive")),
        })
    return steps
