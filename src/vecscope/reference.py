"""Judge the findings against a reference model that embedded the same rows.

A finding measures the space, and the space is shaped by the model and by the rows. Short
templated texts (product titles, FAQ questions) give a low effective rank, many near
neighbors and strong hubs in every good model, so on such a corpus a finding can fire for a
pipeline that retrieves well. A second, independent model embedding the same rows separates
the two causes: when the reference reaches the same verdict on the same rows, the rows
explain the finding; when only this model does, the model or the pipeline does.

The rule is fixed, with no learned part: a non-ok finding listed in ``DATA_SHAPED`` whose
reference verdict is at least as bad becomes ``ok``, and its details keep the reference value,
the reference verdict and the verdict it had without the reference. Findings that describe
the model's output format or the index (norms, rogue dimensions, metric mismatch, ANN recall,
query shift) are never explained this way: a reference with the same defect would have the
same bug, not the same rows.

A finding that stays at ``fail`` with the reference is this model's own, and for the hub
findings (``HUB_SHAPED``) that alone does not say retrieval suffers: some good models put a
few rows into many result lists and still rank the right row first. With queries, the
reference also measures how much of this pipeline's top-k the reference returns for the same
queries (``agreement``). Over 300 public pipelines of three held-out sets that agreement
separated good pipelines from bad ones at AUC 0.99: no bad pipeline at ``fail`` reached
0.31, and the good ones the hub findings kept at ``fail`` agreed at 0.40 to 0.77, except two
code-QA pipelines at 0.29. So a hub finding at ``fail`` on a pipeline that agrees with the
reference at ``AGREEMENT`` or more is lowered to ``warn``, never to ``ok``: the hubs are
real, and labeled retrieval decides what they cost.

The effective rank is a property of the whole space, and a narrow corpus (one scientific
field, one product category) lowers it for every model. When the reference sits at ``warn``
and this model at ``fail``, the rows explain most of it, and this model goes one step
further than a good model does on them; ``effective_rank`` is then lowered to ``warn``
(``PARTLY_EXPLAINED``), never to ``ok``. On the four held-out sets this lowered it on 10
good pipelines and 17 bad ones; 1 bad pipeline was left without a ``fail``. The row-level
findings (collapse, outliers, hubs) keep the strict rule: they are the ones that catch
duplicated, zeroed and crowding rows.

The reference must be good on its own, a public model with published scores on the
language and domain. A reference that is broken in the same way (two English-only models on
Korean text) explains away a real fault. Its own verdict is reported next to the result.
"""

from __future__ import annotations

from vecscope.evidence import WORSE

DATA_SHAPED = (
    "hubness",
    "antihubs",
    "anisotropy",
    "effective_rank",
    "collapse",
    "distance_concentration",
    "outliers",
    "connectivity",
    "ood_queries",
    "query_gap",
    "query_hubness",
)


# findings about the whole space that a narrow corpus lowers for every model: at fail where
# the reference is at warn, they are lowered one step, to warn
PARTLY_EXPLAINED = ("effective_rank",)

# findings about how often rows appear in other rows' or queries' results
HUB_SHAPED = ("hubness", "query_hubness", "antihubs")
# share of this pipeline's top-k the reference returns too, over sampled queries, from which a
# hub finding at fail is lowered to warn
AGREEMENT = 0.40


def apply_agreement(findings: list, agreement: float) -> list[str]:
    """Lower the hub findings still at ``fail`` to ``warn`` when this pipeline's results agree
    with the reference's at ``AGREEMENT`` or more; return their names. ``findings`` is
    changed in place."""
    lowered = []
    for f in findings:
        if f.name not in HUB_SHAPED or f.verdict != "fail" or agreement < AGREEMENT:
            continue
        ref = dict(f.details.get("reference") or {})
        ref.update({"agreement": agreement, "verdict_before_agreement": "fail"})
        f.details = {**f.details, "reference": ref}
        f.explanation += (
            f" TOLERATED BY RETRIEVAL AGREEMENT: the reference model returns {agreement:.0%} of "
            f"this pipeline's top results for the same queries (lowered at {AGREEMENT:.0%}), "
            "so these hubs do not keep it from finding what the reference finds; verdict "
            "without that: fail. The hubs are still there: labeled retrieval says what they cost."
        )
        f.verdict = "warn"
        lowered.append(f.name)
    return lowered


def apply_reference(findings: list, ref_findings: list) -> list[str]:
    """Mark the findings the reference reaches too (and an ``effective_rank`` at fail where the
    reference is at warn, lowered to warn) and return their names. ``findings`` and
    ``ref_findings`` are ``Finding`` objects; ``findings`` is changed in place."""
    ref = {f.name: f for f in ref_findings}
    explained = []
    for f in findings:
        r = ref.get(f.name)
        if r is None or f.name not in DATA_SHAPED:
            continue
        info = {"value": r.value, "verdict": r.verdict, "verdict_without_reference": f.verdict}
        f.details = {**f.details, "reference": info}
        if f.name in PARTLY_EXPLAINED and f.verdict == "fail" and r.verdict == "warn":
            f.explanation += (
                f" PARTLY EXPLAINED BY THE ROWS: the reference model reaches warn on the same "
                f"rows ({r.value:.4g}), so the rows produce most of it and this model goes one "
                "step further; verdict without the reference: fail."
            )
            f.verdict = "warn"
            explained.append(f.name)
            continue
        if f.verdict == "ok" or WORSE[r.verdict] < WORSE[f.verdict]:
            continue
        f.explanation += (
            f" EXPLAINED BY THE ROWS: the reference model reaches {r.verdict} on the same rows "
            f"({r.value:.4g}), so the rows, not this model, produce it; verdict without the "
            f"reference: {f.verdict}. The property is still there"
            + (
                ": the duplicated rows still crowd results, so deduplicate them."
                if f.name == "collapse"
                else "."
            )
        )
        f.verdict = "ok"
        explained.append(f.name)
    return explained
