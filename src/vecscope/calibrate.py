"""Thresholds from your own pipeline states: which findings move with retrieval quality on
your labeled queries, and where to put (warn, fail) so that acceptable states pass.

Input is the evidence of several states of one pipeline (models, chunkers, transforms, index
settings), each audited with the same ``--queries`` and ``--qrels`` so it carries a
``retrieval`` block. A state is acceptable when its quality reaches ``min_quality``. For each
finding the report gives:

- the Spearman correlation between its value and the quality across the states, a 95%
  bootstrap interval for it, and whether the sign is the one the finding's direction implies
  (higher hubness, lower quality);
- a one-sided permutation p-value for that direction, Holm-adjusted over every finding
  tested, because with about fifteen findings one of them correlates by chance on few states;
- the worst value among acceptable states and the best among the others, and whether the
  finding separates them;
- a proposed (warn, fail) pair only when it tracks quality and separates the states, placed a
  third and two thirds of the way from the worst acceptable value to the best unacceptable one.

A proposal whose adjusted p-value reaches ``ALPHA`` is exploratory: it is reported but not
written to the gate thresholds. On five states no finding can be supported (the smallest
one-sided p is 1/120, times the number of findings tested); ten or more states are
recommended. A finding that does not track quality on your data gets no proposal: keep its
default as a reason to look, not as a gate.

``leave_one_out`` judges each state with thresholds fitted on the others, and ``holdout``
states (other states, or the same states measured on held-out queries) check the proposal on
data it was not fitted on; a threshold fitted and checked on the same states looks better than
it is. States scored on different qrels are refused, since their quality values do not
compare. Deterministic (seeded resampling), NumPy only.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np

from vecscope.evidence import _FINDING_KEYS, THRESHOLDS

MIN_RHO = 0.5
ALPHA = 0.05
RECOMMENDED_STATES = 10
_EXACT_PERMUTATIONS = 8  # up to 8! = 40,320 orderings are enumerated; above, a seeded sample
_N_PERMUTATIONS = 20_000
_N_BOOTSTRAP = 2_000


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


def _orderings(n: int) -> tuple[np.ndarray, bool]:
    """Every ordering of n states (exact test), or a seeded sample of them."""
    if n <= _EXACT_PERMUTATIONS:
        return np.asarray(list(itertools.permutations(range(n))), dtype=np.int64), True
    rng = np.random.default_rng(0)
    return np.argsort(rng.random((_N_PERMUTATIONS, n)), axis=1), False


def _permutation_p(v: np.ndarray, q: np.ndarray, sign: float, perms) -> float:
    """One-sided p-value of the rank correlation in the direction the finding implies: the
    share of orderings of the quality values that correlate at least as strongly."""
    order, exact = perms
    rv, rq = _ranks(v), _ranks(q)
    rv, rq = rv - rv.mean(), rq - rq.mean()
    denom = np.linalg.norm(rv) * np.linalg.norm(rq)
    stats = sign * (rq[order] @ rv) / denom
    observed = sign * float(rq @ rv) / denom
    hits = int(np.count_nonzero(stats >= observed - 1e-9))
    return hits / len(order) if exact else (hits + 1) / (len(order) + 1)


def _holm(p: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values: the family is every finding tested on these states."""
    names = sorted(p, key=lambda n: p[n])
    out, running = {}, 0.0
    for i, name in enumerate(names):
        running = max(running, min(1.0, (len(names) - i) * p[name]))
        out[name] = running
    return out


def _bootstrap_ci(v: np.ndarray, q: np.ndarray) -> list[float] | None:
    """95% percentile interval of the rank correlation over states resampled with
    replacement (seeded); None when most resamples are constant on one side."""
    rng = np.random.default_rng(0)
    rhos = []
    for idx in rng.integers(0, v.size, (_N_BOOTSTRAP, v.size)):
        r = spearman(v[idx], q[idx])
        if r is not None:
            rhos.append(r)
    if len(rhos) < _N_BOOTSTRAP // 2:
        return None
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return [float(lo), float(hi)]


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


def _fit(per_state: list[dict], quality: list[float], min_quality: float, min_rho: float,
         include_low: bool, intervals: bool) -> tuple[list, dict, dict]:
    """Rows, gate thresholds and exploratory thresholds from one set of states."""
    q = np.asarray(quality)
    ok = q >= min_quality
    common = [n for n in per_state[0] if all(n in v for v in per_state)]
    perms = _orderings(len(per_state))
    rows, pvals = [], {}
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
        acc, bad = v[ok], v[~ok]
        worst_ok = float(acc.min() if lower_worse else acc.max())
        best_bad = float(bad.max() if lower_worse else bad.min())
        separates = best_bad < worst_ok if lower_worse else best_bad > worst_ok
        row = {"name": name, "threshold_key": key, "spearman_with_quality": rho,
               "spearman_ci95": _bootstrap_ci(v, q) if intervals and rho is not None else None,
               "p_value": None, "p_holm": None,
               "tracks_quality": tracks, "lower_is_worse": lower_worse,
               "worst_acceptable": worst_ok, "best_unacceptable": best_bad,
               "separates": bool(separates), "default": [warn, fail], "proposed": None,
               "confidence": None}
        if rho is not None:
            row["p_value"] = pvals[name] = _permutation_p(v, q, 1.0 if lower_worse else -1.0, perms)
        rows.append((row, v))
    holm = _holm(pvals)
    thresholds, exploratory = {}, {}
    for row, _ in rows:
        name, key = row["name"], row["threshold_key"]
        row["p_holm"] = holm.get(name)
        rho = row["spearman_with_quality"]
        if row["tracks_quality"] and row["separates"]:
            worst_ok, gap = row["worst_acceptable"], row["best_unacceptable"] - row["worst_acceptable"]
            pair = [worst_ok + gap / 3.0, worst_ok + 2.0 * gap / 3.0]
            row["proposed"] = pair
            if row["p_holm"] < ALPHA:
                row["confidence"] = "supported"
                thresholds[key] = pair
                row["advice"] = "gate on the proposed pair"
            else:
                row["confidence"] = "low"
                exploratory[key] = pair
                if include_low:
                    thresholds[key] = pair
                row["advice"] = (f"tracks and separates these states, but with {len(per_state)} "
                                 f"states and {len(pvals)} findings tested the correlation can be "
                                 f"chance (Holm-adjusted p={row['p_holm']:.2g}): exploratory, "
                                 "add states before gating on it")
        elif rho is None:
            row["advice"] = "constant on these states: nothing to calibrate"
        elif abs(rho) < min_rho:
            row["advice"] = ("does not move with quality on these states: keep the default as "
                             "a reason to look, do not gate on it")
        elif not row["tracks_quality"]:
            row["advice"] = ("moves with quality the other way: here a worse value goes with "
                             "better retrieval, so it is not a quality gate on this data")
        else:
            row["advice"] = ("moves with quality but overlaps between acceptable and "
                             "unacceptable states: no threshold separates them")
    out = [r for r, _ in rows]
    out.sort(key=lambda r: ({"supported": 0, "low": 1}.get(r["confidence"], 2),
                            -abs(r["spearman_with_quality"] or 0)))
    return out, thresholds, exploratory


def calibrate(
    states: Sequence[Mapping],
    min_quality: float,
    metric: str = "ndcg_at_k",
    ranking: str = "exact",
    holdout: Sequence[Mapping] | None = None,
    names: Sequence[str] | None = None,
    min_rho: float = MIN_RHO,
    include_low_confidence: bool = False,
    allow_mixed_qrels: bool = False,
) -> dict:
    """Propose thresholds from ``states`` (evidence dicts with a retrieval block).

    ``names`` labels the states in the report (file names); ``holdout`` states are judged with
    the proposal and ``min_quality``. Returns ``findings`` (one row per finding),
    ``thresholds`` (the pairs to gate on by threshold key, ready for ``--thresholds``: the
    statistically supported ones, plus the exploratory ones with ``include_low_confidence``),
    ``exploratory_thresholds``, ``states``, ``leave_one_out`` and, with a holdout,
    ``holdout`` with per-state predictions and counts."""
    if len(states) < 3:
        raise ValueError("calibration needs at least 3 states (10 or more to support a threshold)")
    names = list(names) if names is not None else [f"state {i}" for i in range(len(states))]
    quality = [_quality(ev, metric, ranking, n) for ev, n in zip(states, names)]
    fps = {(ev["retrieval"].get("qrels_fingerprint"), ev["retrieval"].get("k")) for ev in states}
    if len(fps) > 1 and not allow_mixed_qrels:
        raise ValueError(f"the states were scored on {len(fps)} different qrels or k, so their "
                         "quality values are not comparable; audit every state with the same "
                         "--queries, --qrels and -k (or pass --allow-mixed-qrels)")
    ok = [q >= min_quality for q in quality]
    if all(ok) or not any(ok):
        raise ValueError(f"min_quality {min_quality} puts every state on one side ({metric} "
                         f"{min(quality):.3f} to {max(quality):.3f}); calibration needs "
                         "acceptable and unacceptable states")
    per_state = [_values(ev) for ev in states]
    rows, thresholds, exploratory = _fit(per_state, quality, min_quality, min_rho,
                                         include_low_confidence, intervals=True)
    out = {
        "metric": metric, "ranking": ranking, "min_quality": min_quality,
        "n_states": len(states), "recommended_states": RECOMMENDED_STATES, "alpha": ALPHA,
        "states": [{"name": n, "quality": qv, "acceptable": a}
                   for n, qv, a in zip(names, quality, ok)],
        "same_qrels": len(fps) == 1,
        "findings": rows,
        "thresholds": thresholds,
        "exploratory_thresholds": exploratory,
        "leave_one_out": _leave_one_out(per_state, quality, min_quality, min_rho,
                                        include_low_confidence),
    }
    if holdout:
        out["holdout"] = _check(holdout, thresholds, min_quality, metric, ranking)
    return out


def _leave_one_out(per_state, quality, min_quality, min_rho, include_low) -> dict | None:
    """Each state judged by thresholds fitted on the others: how the procedure does on a
    state it did not see. States whose removal leaves one side empty are skipped."""
    if len(per_state) < 4:
        return None
    caught = missed = false_alarms = passed = skipped = 0
    for i in range(len(per_state)):
        rest = [s for j, s in enumerate(per_state) if j != i]
        rq = [qv for j, qv in enumerate(quality) if j != i]
        if all(qv >= min_quality for qv in rq) or not any(qv >= min_quality for qv in rq):
            skipped += 1
            continue
        _, th, _ = _fit(rest, rq, min_quality, min_rho, include_low, intervals=False)
        bad, pred = quality[i] < min_quality, bool(_flagged({"findings": [
            {"name": n, "value": v} for n, v in per_state[i].items()]}, th))
        caught += bad and pred
        missed += bad and not pred
        false_alarms += pred and not bad
        passed += not bad and not pred
    return {"caught_unacceptable": caught, "missed_unacceptable": missed,
            "false_alarms": false_alarms, "passed_acceptable": passed, "skipped": skipped}


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
    lines = [(f"calibration on {res['n_states']} states, acceptable when "
              f"{res['ranking']} {res['metric']} >= {res['min_quality']}")]
    for s in res["states"]:
        lines.append(f"  {s['name']:<32} {s['quality']:.3f}  {'ok' if s['acceptable'] else 'below'}")
    if not res["same_qrels"]:
        lines.append("  note: the states were not all scored on the same qrels and k")
    if res["n_states"] < res["recommended_states"]:
        lines.append(f"  note: {res['n_states']} states are few; correlations this small a set "
                     f"shows can be chance, {res['recommended_states']} or more are recommended")
    lines.append("")
    lines.append(f"  {'finding':<22} {'rho':>6} {'95% interval':>14} {'p(holm)':>8}  "
                 f"{'worst ok':>9} {'best bad':>9}  proposal")
    for r in res["findings"]:
        rho = "n/a" if r["spearman_with_quality"] is None else f"{r['spearman_with_quality']:+.2f}"
        ci = ("" if r["spearman_ci95"] is None else
              f"[{r['spearman_ci95'][0]:+.2f},{r['spearman_ci95'][1]:+.2f}]")
        p = "" if r["p_holm"] is None else f"{r['p_holm']:.3g}"
        if r["confidence"] == "supported":
            prop = f"warn {r['proposed'][0]:.4g}, fail {r['proposed'][1]:.4g}"
        elif r["confidence"] == "low":
            prop = f"exploratory: warn {r['proposed'][0]:.4g}, fail {r['proposed'][1]:.4g}"
        else:
            prop = r["advice"]
        lines.append(f"  {r['name']:<22} {rho:>6} {ci:>14} {p:>8}  {r['worst_acceptable']:>9.4g} "
                     f"{r['best_unacceptable']:>9.4g}  {prop}")
    loo = res.get("leave_one_out")
    if loo:
        judged = loo["caught_unacceptable"] + loo["missed_unacceptable"]
        lines.append("")
        lines.append(f"  leave-one-out: caught {loo['caught_unacceptable']} of {judged} "
                     f"unacceptable states, {loo['false_alarms']} false alarms on "
                     f"{loo['false_alarms'] + loo['passed_acceptable']} acceptable ones"
                     + (f" ({loo['skipped']} skipped)" if loo["skipped"] else ""))
    h = res.get("holdout")
    if h:
        lines.append("")
        lines.append(f"  holdout: caught {h['caught_unacceptable']} of "
                     f"{h['caught_unacceptable'] + h['missed_unacceptable']} unacceptable states, "
                     f"{h['false_alarms']} false alarms on "
                     f"{h['false_alarms'] + h['passed_acceptable']} acceptable ones")
    if res["exploratory_thresholds"] and not set(res["exploratory_thresholds"]) <= set(res["thresholds"]):
        lines.append("")
        lines.append("  exploratory pairs are not in the gate thresholds; add states, or pass "
                     "--include-low-confidence to gate on them anyway")
    if not res["thresholds"]:
        lines.append("")
        lines.append("  no finding is supported as a gate on these states: keep the defaults as "
                     "reasons to look and gate on retrieval quality itself")
    return "\n".join(lines)
