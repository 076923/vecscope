"""Thresholds proposed from the user's own pipeline states (vecscope.calibrate)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vecscope import audit
from vecscope.calibrate import calibrate, spearman
from vecscope.cli import main

LEVELS = [0, 0.3, 0.6, 0.9, 1.2, 1.6, 2.0]


def _states(levels, seed):
    """One clustered corpus with labeled queries, documents degraded by growing noise."""
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((30, 32))
    clean = c[rng.integers(0, 30, 1500)] + rng.standard_normal((1500, 32)) * 0.5
    t = rng.choice(1500, 100, replace=False)
    q = (clean[t] + rng.standard_normal((100, 32)) * 0.2).astype(np.float32)
    qrels = {i: {int(x): 1} for i, x in enumerate(t)}
    return [audit((clean + rng.standard_normal(clean.shape) * lv).astype(np.float32), k=10,
                  queries=q, qrels=qrels).to_dict() for lv in levels]


@pytest.fixture(scope="module")
def train():
    return _states(LEVELS, 0)


def test_spearman_ranks_with_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 2, 4], [4, 3, 3, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None and spearman([1, 2], [1, 2]) is None


def test_proposes_only_for_findings_that_track_and_separate(train):
    res = calibrate(train, 0.6, names=[f"noise {lv}" for lv in LEVELS])
    rows = {r["name"]: r for r in res["findings"]}
    dc = rows["distance_concentration"]
    assert dc["tracks_quality"] and dc["separates"] and dc["lower_is_worse"]
    warn, fail = dc["proposed"]
    assert dc["best_unacceptable"] < fail < warn < dc["worst_acceptable"]
    assert res["thresholds"]["distance_concentration.contrast_over_noise"] == [warn, fail]
    # anisotropy falls as the noise grows: a worse value with better retrieval is no gate
    assert rows["anisotropy"]["proposed"] is None
    assert "the other way" in rows["anisotropy"]["advice"]
    assert "constant" in rows["collapse"]["advice"]
    assert set(res["thresholds"]) == {r["threshold_key"] for r in res["findings"] if r["proposed"]}
    assert res["same_qrels"] and [s["acceptable"] for s in res["states"]] == [
        True, True, True, True, False, False, False]


def test_proposal_is_used_by_audit_and_checked_on_holdout(train, tmp_path):
    res = calibrate(train, 0.6, holdout=_states([0.1, 1.8], 1))
    h = res["holdout"]
    assert h["caught_unacceptable"] == 1 and h["missed_unacceptable"] == 0
    assert h["states"][1]["flagged_by"] and not h["states"][1]["acceptable"]
    # the pairs drop straight into --thresholds
    x = np.random.default_rng(0).standard_normal((300, 8)).astype(np.float32)
    ev = audit(x, k=5, thresholds=res["thresholds"])
    used = {k: tuple(v) for k, v in ev.thresholds.items()}
    for key, pair in res["thresholds"].items():
        assert used[key] == tuple(pair)


def test_needs_both_sides_and_retrieval(train):
    with pytest.raises(ValueError, match="one side"):
        calibrate(train, 0.0)
    with pytest.raises(ValueError, match="at least 3"):
        calibrate(train[:2], 0.6)
    no_retrieval = [dict(s, retrieval={}) for s in train]
    with pytest.raises(ValueError, match="--qrels"):
        calibrate(no_retrieval, 0.6)


def test_cli_calibrate(train, tmp_path, capsys):
    paths = []
    for lv, ev in zip(LEVELS, train):
        p = tmp_path / f"noise{lv}.json"
        p.write_text(json.dumps(ev))
        paths.append(str(p))
    out = tmp_path / "th.json"
    assert main(["calibrate", *paths, "--min-quality", "0.6", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "distance_concentration" in text and "noise0.json" in text
    proposed = json.loads(out.read_text())
    assert "distance_concentration.contrast_over_noise" in proposed
    np.save(tmp_path / "x.npy", np.random.default_rng(0).standard_normal((300, 8)).astype(np.float32))
    assert main(["audit", str(tmp_path / "x.npy"), "--thresholds", str(out), "--fail-on", "none"]) == 0
    assert main(["calibrate", *paths[:2], "--min-quality", "0.6"]) == 2
