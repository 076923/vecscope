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


def apply_reference(findings: list, ref_findings: list) -> list[str]:
    """Mark the findings the reference reaches too and return their names. ``findings``
    and ``ref_findings`` are ``Finding`` objects; ``findings`` is changed in place."""
    ref = {f.name: f for f in ref_findings}
    explained = []
    for f in findings:
        r = ref.get(f.name)
        if r is None or f.name not in DATA_SHAPED:
            continue
        info = {"value": r.value, "verdict": r.verdict, "verdict_without_reference": f.verdict}
        f.details = {**f.details, "reference": info}
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
