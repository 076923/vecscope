"""Turn the evidence JSON of `vecscope audit --json` into step outputs and a job summary.

Used by the composite action in action.yml. Standard library only, so it runs on the
Python that setup-python provided before or after vecscope is installed.

Usage: python report.py <evidence.json> <exit code of vecscope audit>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _value(v: object) -> str:
    return f"{v:.4f}" if isinstance(v, float) else _cell(v)


def _write(env: str, text: str) -> None:
    target = os.environ.get(env)
    if target:
        with open(target, "a", encoding="utf-8") as f:
            f.write(text)


def main(argv: list[str]) -> int:
    path, code = Path(argv[0]), argv[1]
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        evidence = None
    if not isinstance(evidence, dict):
        _write(
            "GITHUB_OUTPUT",
            f"verdict=error\nexit-code={code}\nregressions=\nretrieval-regressions=\n"
            f"evidence={path}\n",
        )
        _write(
            "GITHUB_STEP_SUMMARY",
            f"## vecscope audit: error\n\n`vecscope audit` exited {code} without evidence "
            f"JSON in `{path}`. The step log shows its message.\n",
        )
        return 0

    gate = evidence.get("gate") or {}
    names = (gate.get("verdict_regressions") or []) + (gate.get("value_regressions") or [])
    regressions = list(dict.fromkeys(names))
    retrieval = (
        list(gate.get("retrieval_regressions") or [])
        + [f"unchecked:{k}" for k in gate.get("retrieval_unchecked") or []]
        + [f"inconclusive:{k}" for k in gate.get("retrieval_inconclusive") or []]
    )
    verdict = evidence.get("verdict", "unknown")
    _write(
        "GITHUB_OUTPUT",
        f"verdict={verdict}\nexit-code={code}\nregressions={','.join(regressions)}\n"
        f"retrieval-regressions={','.join(retrieval)}\nevidence={path}\n",
    )

    lines = [
        f"## vecscope audit: {verdict}",
        "",
        (
            f"n = {evidence.get('n')}, dim = {evidence.get('dim')}, k = {evidence.get('k')}, "
            f"scope = {evidence.get('evidence_scope')}, exit code = {code}"
        ),
        "",
    ]
    reasons = gate.get("reasons") or []
    lines.append("**Gate:** " + ("; ".join(reasons) if reasons else "pass"))
    lines += ["", "| finding | value | verdict | healthy range |", "|---|---|---|---|"]
    for f in evidence.get("findings") or []:
        lines.append(
            f"| {_cell(f.get('name'))} | {_value(f.get('value'))} | {_cell(f.get('verdict'))} "
            f"| {_cell(f.get('healthy_range', ''))} |"
        )
    diff = evidence.get("baseline_diff")
    if isinstance(diff, dict):
        lines += ["", f"Baseline verdict {diff.get('baseline_verdict')} -> {diff.get('verdict')}"]
        if not diff.get("comparable", True):
            lines.append(
                "(n, dim, k, sampling, thresholds or near-dup cutoff differ from the baseline)"
            )
        if regressions:
            lines.append("Regressions: " + ", ".join(regressions))
        changes = diff.get("retrieval_changes") or []
        if changes:
            note = "" if diff.get("retrieval_comparable") else " (different qrels or k)"
            lines += [
                "",
                f"Retrieval vs baseline{note}:",
                "",
                "| ranking | metric | baseline | value | delta | tolerance |",
                "|---|---|---|---|---|---|",
            ]
            bad = set(diff.get("retrieval_regressions") or [])
            for c in changes:
                t = c.get("tolerance")
                key = f"{c.get('ranking')}.{c.get('metric')}"
                lines.append(
                    f"| {_cell(c.get('ranking'))} | {_cell(c.get('metric'))} "
                    f"| {_value(c.get('baseline_value'))} | {_value(c.get('value'))} "
                    f"| {_value(c.get('delta'))} "
                    f"| {'' if t is None else _value(t)}{' **exceeded**' if key in bad else ''} |"
                )
        if diff.get("retrieval_unchecked"):
            lines.append(
                "\nRetrieval tolerances not checked: " + ", ".join(diff["retrieval_unchecked"])
            )
        checks = diff.get("retrieval_confidence_checks") or {}
        for key in diff.get("retrieval_inconclusive") or []:
            c = checks.get(key) or {}
            lines.append(
                f"\nRetrieval inconclusive (label more queries, not broken): {_cell(key)} drop "
                f"{_value(c.get('mean_drop'))} in [{_value(c.get('low'))}, {_value(c.get('high'))}]"
                + (
                    f", about {c['queries_needed']} labeled queries"
                    if c.get("queries_needed")
                    else ""
                )
            )
    if isinstance(diff, dict) and diff.get("reference_regressions"):
        lines.append(
            "\nTop hits agreed with the reference model were lost (likely harmful, check with "
            "labels): " + ", ".join(_cell(s) for s in diff["reference_regressions"])
        )
    if isinstance(diff, dict) and diff.get("reference_review"):
        lines.append(
            "\nA few top hits agreed with the reference model were lost, below the tolerance "
            "(not a failure; a small loss is possible, check with labels): "
            + ", ".join(_cell(s) for s in diff["reference_review"])
        )
    lines += ["", f"Full evidence: `{path}`", ""]
    _write("GITHUB_STEP_SUMMARY", "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
