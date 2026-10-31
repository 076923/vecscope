"""Thresholds from your own pipeline states: which findings move with retrieval quality on
your labeled queries, and where to put (warn, fail) so that acceptable states pass.

Input is the evidence of several states of one pipeline (models, chunkers, transforms, index
settings), each audited with the same ``--queries`` and ``--qrels`` so it carries a
``retrieval`` block. A state is acceptable when its quality reaches ``min_quality``. For each
finding the report gives:

- the Spearman correlation between its value and the quality across the states, and whether
  the sign is the one the finding's direction implies (higher hubness, lower quality);
- the worst value among acceptable states and the best among the others, and whether the
  finding separates them;
- a proposed (warn, fail) pair only when it does both, placed a third and two thirds of the
  way from the worst acceptable value to the best unacceptable one.

A finding that does not track quality on your data gets no proposal: keep its default as a
reason to look, not as a gate. ``holdout`` states (other states, or the same states measured
on held-out queries) check the proposal on data it was not fitted on; a threshold fitted and
checked on the same states looks better than it is. Deterministic, NumPy only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from vecscope.evidence import _FINDING_KEYS, THRESHOLDS

MIN_RHO = 0.5


def _ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean of their positions)."""
    order = np.argsort(a, kind="stable")
    ranks = np.empty(a.size, dtype=np.float64)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Rank correlation, or None when either side is constant or there are fewer than 3."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ra, rb = _ranks(a), _ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _quality(ev: Mapping, metric: str, ranking: str, what: str) -> float:
    r = ev.get("retrieval") if isinstance(ev, Mapping) else None
    if not r or ranking not in r or not r[ranking]:
        raise ValueError(f"{what} has no retrieval.{ranking} block; audit it with --queries and "
                         "--qrels (and --ranked for the ranked block)")
    if metric not in r[ranking]:
        raise ValueError(f"{what}: unknown metric {metric!r}; one of {sorted(r[ranking])}")
    return float(r[ranking][metric])


def _values(ev: Mapping) -> dict[str, float]:
    return {f["name"]: float(f["value"]) for f in ev.get("findings", [])}


def calibrate(
    states: Sequence[Mapping],
    min_quality: float,
    metric: str = "ndcg_at_k",
    ranking: str = "exact",
    holdout: Sequence[Mapping] | None = None,
    names: Sequence[str] | None = None,
    min_rho: float = MIN_RHO,
) -> dict:
    """Propose thresholds from ``states`` (evidence dicts with a retrieval block).

    ``names`` labels the states in the report (file names); ``holdout`` states are judged with
    the proposal and ``min_quality``. Returns ``findings`` (one row per finding),
    ``thresholds`` (the proposed pairs by threshold key, ready for ``--thresholds``),
    ``states`` and, with a holdout, ``holdout`` with per-state predictions and counts."""
    if len(states) < 3:
        raise ValueError("calibration needs at least 3 states (5 or more to trust a correlation)")
    names = list(names) if names is not None else [f"state {i}" for i in range(len(states))]
    quality = [_quality(ev, metric, ranking, n) for ev, n in zip(states, names)]
    fps = {(ev["retrieval"].get("qrels_fingerprint"), ev["retrieval"].get("k")) for ev in states}
    ok = [q >= min_quality for q in quality]
    if all(ok) or not any(ok):
        raise ValueError(f"min_quality {min_quality} puts every state on one side ({metric} "
                         f"{min(quality):.3f} to {max(quality):.3f}); calibration needs "
                         "acceptable and unacceptable states")
    per_state = [_values(ev) for ev in states]
    common = [n for n in per_state[0] if all(n in v for v in per_state)]
    rows, thresholds = [], {}
    q = np.asarray(quality)
    for name in common:
        key = _FINDING_KEYS.get(name)
        if key is None:
            continue
        v = np.asarray([s[name] for s in per_state])
        warn, fail = THRESHOLDS[key]
        lower_worse = warn > fail
        rho = spearman(v, q)
        # higher values are worse unless the finding says lower is worse
        tracks = rho is not None and abs(rho) >= min_rho and (rho > 0 if lower_worse else rho < 0)
        acc, bad = v[np.asarray(ok)], v[~np.asarray(ok)]
        worst_ok = float(acc.min() if lower_worse else acc.max())
        best_bad = float(bad.max() if lower_worse else bad.min())
        separates = best_bad < worst_ok if lower_worse else best_bad > worst_ok
        row = {"name": name, "threshold_key": key, "spearman_with_quality": rho,
               "tracks_quality": tracks, "lower_is_worse": lower_worse,
               "worst_acceptable": worst_ok, "best_unacceptable": best_bad,
               "separates": bool(separates), "default": [warn, fail], "proposed": None}
        if tracks and separates:
            gap = best_bad - worst_ok
            pair = [worst_ok + gap / 3.0, worst_ok + 2.0 * gap / 3.0]
            row["proposed"] = pair
            thresholds[key] = pair
            row["advice"] = "gate on the proposed pair"
        elif rho is None:
            row["advice"] = "constant on these states: nothing to calibrate"
        elif abs(rho) < min_rho:
            row["advice"] = ("does not move with quality on these states: keep the default as "
                             "a reason to look, do not gate on it")
        elif not tracks:
            row["advice"] = ("moves with quality the other way: here a worse value goes with "
                             "better retrieval, so it is not a quality gate on this data")
        else:
            row["advice"] = ("moves with quality but overlaps between acceptable and "
                             "unacceptable states: no threshold separates them")
        rows.append(row)
    rows.sort(key=lambda r: (r["proposed"] is None, -abs(r["spearman_with_quality"] or 0)))
    out = {
        "metric": metric, "ranking": ranking, "min_quality": min_quality,
        "states": [{"name": n, "quality": qv, "acceptable": a}
                   for n, qv, a in zip(names, quality, ok)],
        "same_qrels": len(fps) == 1,
        "findings": rows,
        "thresholds": thresholds,
    }
    if holdout:
        out["holdout"] = _check(holdout, thresholds, min_quality, metric, ranking)
    return out


def _flagged(ev: Mapping, thresholds: Mapping[str, Sequence[float]]) -> list[str]:
    vals = _values(ev)
    hit = []
    for name, v in vals.items():
        pair = thresholds.get(_FINDING_KEYS.get(name, ""))
        if pair is not None and (v <= pair[0] if pair[0] > pair[1] else v >= pair[0]):
            hit.append(name)
    return sorted(hit)


def _check(holdout: Sequence[Mapping], thresholds, min_quality, metric, ranking) -> dict:
    rows, tp, fp, fn, tn = [], 0, 0, 0, 0
    for i, ev in enumerate(holdout):
        qv = _quality(ev, metric, ranking, f"holdout {i}")
        flagged = _flagged(ev, thresholds)
        bad, pred = qv < min_quality, bool(flagged)
        tp += bad and pred
        fp += pred and not bad
        fn += bad and not pred
        tn += not bad and not pred
        rows.append({"index": i, "quality": qv, "acceptable": not bad, "flagged_by": flagged})
    return {"states": rows, "caught_unacceptable": tp, "missed_unacceptable": fn,
            "false_alarms": fp, "passed_acceptable": tn}


def render_calibration(res: dict) -> str:
    lines = [(f"calibration on {len(res['states'])} states, acceptable when "
              f"{res['ranking']} {res['metric']} >= {res['min_quality']}")]
    for s in res["states"]:
        lines.append(f"  {s['name']:<32} {s['quality']:.3f}  {'ok' if s['acceptable'] else 'below'}")
    if not res["same_qrels"]:
        lines.append("  note: the states were not all scored on the same qrels and k")
    lines.append("")
    lines.append(f"  {'finding':<22} {'rho':>6}  {'worst ok':>9} {'best bad':>9}  proposal")
    for r in res["findings"]:
        rho = "n/a" if r["spearman_with_quality"] is None else f"{r['spearman_with_quality']:+.2f}"
        prop = (f"warn {r['proposed'][0]:.4g}, fail {r['proposed'][1]:.4g}" if r["proposed"]
                else r["advice"])
        lines.append(f"  {r['name']:<22} {rho:>6}  {r['worst_acceptable']:>9.4g} "
                     f"{r['best_unacceptable']:>9.4g}  {prop}")
    h = res.get("holdout")
    if h:
        lines.append("")
        lines.append(f"  holdout: caught {h['caught_unacceptable']} of "
                     f"{h['caught_unacceptable'] + h['missed_unacceptable']} unacceptable states, "
                     f"{h['false_alarms']} false alarms on "
                     f"{h['false_alarms'] + h['passed_acceptable']} acceptable ones")
    if not res["thresholds"]:
        lines.append("")
        lines.append("  no finding both tracks quality and separates the states: keep the "
                     "defaults as reasons to look and gate on retrieval quality itself")
    return "\n".join(lines)
