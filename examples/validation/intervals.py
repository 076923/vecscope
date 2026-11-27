"""95% intervals for the counts and AUCs of the preregistered held-out sets.

A criterion such as "12 of 23 listed" passes or fails on a count, and a small count is a weak
result even when it passes. This module puts an interval on each:

- ``wilson``: the Wilson score interval of k successes in n, for counts of pipelines.
- ``cluster_bootstrap``: a percentile interval for a statistic of the injected faults,
  resampling whole pipelines (dataset and model) with replacement, because the six faults of
  one pipeline share its corpus and queries and are not independent.

Run on the cached fault records of the fourth and fifth sets (the sets whose records hold
everything the criteria need):
    python examples/validation/intervals.py
Writes examples/validation/intervals_summary.json. The intervals are reported next to the
results; they change no verdict of a preregistered criterion.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SEED = 0
DRAWS = 2000
Z95 = 1.959963984540054


def wilson(k: int, n: int) -> list[float] | None:
    """Wilson score interval for k of n at 95%."""
    if n == 0:
        return None
    p = k / n
    den = 1 + Z95**2 / n
    mid = (p + Z95**2 / (2 * n)) / den
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / den
    return [round(max(0.0, mid - half), 4), round(min(1.0, mid + half), 4)]


def cluster_bootstrap(
    cases: list[dict], stat: Callable[[list[dict]], float | None], draws: int = DRAWS
) -> list[float] | None:
    """Percentile 95% interval of ``stat`` over pipelines resampled with replacement."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in cases:
        groups[(c["dataset"], c["model"])].append(c)
    keys = sorted(groups)
    if not keys:
        return None
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(draws):
        pick = rng.integers(0, len(keys), len(keys))
        v = stat([c for i in pick for c in groups[keys[i]]])
        if v is not None and not math.isnan(v):
            vals.append(v)
    if not vals:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def _auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    a = np.asarray(pos, dtype=float)[:, None]
    b = np.asarray(neg, dtype=float)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))


def _hit(c: dict) -> float | None:
    return (c["top_hit"].get("queries") or {}).get("z")


def fault_intervals(cases: list[dict], tolerance: float = 0.01) -> dict:
    """Intervals for the fault criteria of P8/P11 (and P12 when the records list reviews)."""
    nodup = [c for c in cases if c["fault"] != "duplicates" and _hit(c) is not None]

    def harm(cs):
        return [c for c in cs if c["ndcg_drop"] > tolerance]

    def quiet(cs):
        return [c for c in cs if c["ndcg_drop"] <= tolerance]

    def big(cs):
        return [c for c in cs if c["ndcg_drop"] > 0.02]

    def share(sel, cond):
        def f(cs):
            s = sel(cs)
            return sum(cond(c) for c in s) / len(s) if s else None

        return f

    flagged = share(big, lambda c: bool(c["reference_regressions"]))
    alarm = share(quiet, lambda c: bool(c["reference_regressions"]))
    out = {
        "pipelines": len({(c["dataset"], c["model"]) for c in nodup}),
        "auc_top_hit_z": {
            "value": _auc([_hit(c) for c in harm(nodup)], [_hit(c) for c in quiet(nodup)]),
            "interval": cluster_bootstrap(
                nodup, lambda cs: _auc([_hit(c) for c in harm(cs)], [_hit(c) for c in quiet(cs)])
            ),
        },
        "flagged_of_drop_over_002": {
            "value": flagged(nodup),
            "n": len(big(nodup)),
            "interval": cluster_bootstrap(nodup, flagged),
            "wilson": wilson(
                sum(bool(c["reference_regressions"]) for c in big(nodup)), len(big(nodup))
            ),
        },
        "flagged_of_harmless": {
            "value": alarm(nodup),
            "n": len(quiet(nodup)),
            "interval": cluster_bootstrap(nodup, alarm),
            "wilson": wilson(
                sum(bool(c["reference_regressions"]) for c in quiet(nodup)), len(quiet(nodup))
            ),
        },
    }
    if all("reference_review" in c for c in nodup):

        def missed(cs):
            return [c for c in big(cs) if not c["reference_regressions"]]

        def silent(cs):
            return [c for c in quiet(cs) if not c["reference_regressions"]]

        reviewed = share(missed, lambda c: bool(c["reference_review"]))

        def harmless_reviewed(cs):
            q = quiet(cs)
            return sum(bool(c["reference_review"]) for c in silent(cs)) / len(q) if q else None

        m, q = missed(nodup), quiet(nodup)
        out["reviewed_of_missed"] = {
            "value": reviewed(nodup),
            "n": len(m),
            "interval": cluster_bootstrap(nodup, reviewed),
            "wilson": wilson(sum(bool(c["reference_review"]) for c in m), len(m)),
        }
        out["harmless_reviewed"] = {
            "value": harmless_reviewed(nodup),
            "n": len(q),
            "interval": cluster_bootstrap(nodup, harmless_reviewed),
            "wilson": wilson(sum(bool(c["reference_review"]) for c in silent(nodup)), len(q)),
        }
    return out


def count_interval(k: int, n: int) -> dict:
    return {"k": k, "n": n, "share": k / n if n else None, "wilson": wilson(k, n)}


def _load(path: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("*.json"))]


def main(argv: list[str] | None = None) -> int:
    from run import out_dir

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", help="run.py cache directory")
    args = p.parse_args(argv)
    cache = out_dir(args.out)
    summary: dict = {"method": "Wilson 95% for counts; pipeline bootstrap 95%, 2000 draws"}
    for name, folder, summ in (
        ("fourth", "fourth_faults", "fourth_summary.json"),
        ("fifth", "fifth_faults", "fifth_summary.json"),
    ):
        cases = _load(cache / folder)
        s = json.loads((HERE / summ).read_text(encoding="utf-8"))
        rules = s.get("P9_agreement_rule") or s.get("P13_lowering_rules") or {}
        summary[name] = {
            "faults": fault_intervals(cases),
            "good_cleared": count_interval(
                rules.get("good_cleared", 0), rules.get("good_fail_with_reference", 0)
            ),
            "bad_kept": count_interval(
                rules.get("bad_kept", 0), rules.get("bad_fail_with_reference", 0)
            ),
            "faults_still_caught": count_interval(
                rules.get("faults_still_caught", 0),
                rules.get(
                    "faults_caught_before_rules", rules.get("faults_caught_before_agreement", 0)
                ),
            ),
        }
    (HERE / "intervals_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
