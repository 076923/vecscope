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
        _write("GITHUB_OUTPUT", f"verdict=error\nexit-code={code}\nregressions=\nevidence={path}\n")
        _write(
            "GITHUB_STEP_SUMMARY",
            f"## vecscope audit: error\n\n`vecscope audit` exited {code} without evidence "
            f"JSON in `{path}`. The step log shows its message.\n",
        )
        return 0

    gate = evidence.get("gate") or {}
    names = (gate.get("verdict_regressions") or []) + (gate.get("value_regressions") or [])
    regressions = list(dict.fromkeys(names))
    verdict = evidence.get("verdict", "unknown")
    _write(
        "GITHUB_OUTPUT",
        f"verdict={verdict}\nexit-code={code}\nregressions={','.join(regressions)}\nevidence={path}\n",
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
    lines += ["", f"Full evidence: `{path}`", ""]
    _write("GITHUB_STEP_SUMMARY", "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
