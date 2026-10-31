"""The fix-order plan over non-ok findings, and incremental exact k-NN."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vecscope import audit
from vecscope._knn import exact_knn
from vecscope.cli import main
from vecscope.incremental import (
    cached_knn,
    cached_knn_for_audit,
    extend_knn,
    fingerprint,
    refresh_knn,
)
from vecscope.plan import build_plan


def _clustered(n, d=32, centers=20, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((centers, d))
    return (c[rng.integers(0, centers, n)] + rng.standard_normal((n, d)) * 0.6).astype(np.float32)


def _unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _groups(ev):
    return [(s["group"], [f["name"] for f in s["findings"]]) for s in ev.plan]


# --------------------------------------------------------------------------------- plan

def test_healthy_corpus_has_an_empty_plan():
    assert audit(_unit(_clustered(1500)), k=10).plan == []


def test_shared_offset_groups_anisotropy_with_the_collapse_it_causes():
    x = _unit(_unit(_clustered(1500)) + 1.5)  # one direction shared by every vector
    ev = audit(x, k=10)
    assert [g for g, _ in _groups(ev)] == ["shared_direction"]
    first = ev.plan[0]
    names = [f["name"] for f in first["findings"]]
    assert first["order"] == 1 and "anisotropy" in names
    # every cosine passes the near-duplicate cutoff, but there is no copy to remove
    assert "collapse" in names or ev.findings[[f.name for f in ev.findings].index("collapse")].verdict == "ok"
    assert "subtract the corpus mean" in first["first_change"]


def test_zero_vectors_and_duplicates_come_before_everything():
    x = _clustered(1500)
    x[:30] = 0.0  # failed embedding calls
    x[30:330] = x[330]  # one boilerplate chunk
    x += 6.0 * (np.arange(1500) >= 330)[:, None]  # and a shared offset on the rest
    ev = audit(x, k=10)
    groups = [g for g, _ in _groups(ev)]
    assert groups[0] == "input_integrity"
    names = dict(_groups(ev))["input_integrity"]
    assert "norms" in names and "collapse" in names
    assert [s["order"] for s in ev.plan] == list(range(1, len(ev.plan) + 1))


def test_a_dominant_coordinate_owns_the_findings_it_moves():
    # one coordinate scaled 30x also spreads the norms and splits cosine from inner product;
    # normalizing would not remove it, so those findings join the z-score step
    x = _unit(_clustered(1500))
    x[:, 0] *= 30.0
    ev = audit(x, k=10)
    assert [g for g, _ in _groups(ev)] == ["dominant_dimension"]
    names = dict(_groups(ev))["dominant_dimension"]
    assert "rogue_dimensions" in names and "norms" in names and "metric_mismatch" in names
    assert "z-score" in ev.plan[0]["first_change"]


def test_unnormalized_vectors_are_a_metric_step():
    rng = np.random.default_rng(1)
    x = _clustered(1500) * rng.uniform(0.2, 3.0, (1500, 1)).astype(np.float32)
    groups = dict(_groups(audit(x, k=10)))
    assert "metric" in groups and "norms" in groups["metric"]


def test_query_side_steps_come_last_and_query_hubness_follows_corpus_hubs():
    findings = [
        {"name": "hubness", "value": 5.0, "verdict": "fail", "details": {}},
        {"name": "query_gap", "value": 4.0, "verdict": "fail", "details": {}},
        {"name": "query_hubness", "value": 12.0, "verdict": "fail", "details": {}},
        {"name": "anisotropy", "value": 0.2, "verdict": "ok", "details": {}},
    ]
    plan = build_plan(findings)
    assert [s["group"] for s in plan] == ["corpus_structure", "query_pipeline"]
    assert [f["name"] for f in plan[0]["findings"]] == ["hubness", "query_hubness"]
    assert [f["name"] for f in plan[1]["findings"]] == ["query_gap"]
    alone = build_plan([{"name": "query_hubness", "value": 12.0, "verdict": "fail", "details": {}}])
    assert alone[0]["group"] == "query_pipeline"
    assert "--qrels" in plan[0]["re_measure"]
    assert "--qrels" not in build_plan(findings, retrieval={"k": 10})[0]["re_measure"]


def test_query_gap_next_to_duplicates_is_read_again_after_deduplicating():
    # near-copies shrink the corpus nearest-neighbor distance the gap divides by
    dup = {"name": "collapse", "value": 0.75, "verdict": "fail",
           "details": {"exact_duplicate_ratio": 0.0}}
    gap = {"name": "query_gap", "value": 4.0, "verdict": "fail", "details": {}}
    plan = build_plan([dup, gap])
    assert [s["group"] for s in plan] == ["input_integrity"]
    assert [f["name"] for f in plan[0]["findings"]] == ["collapse", "query_gap"]
    # without a duplicate problem the gap is the queries'
    alone = build_plan([gap])
    assert [s["group"] for s in alone] == ["query_pipeline"]
    # collapse caused by a shared direction does not take the gap with it
    aniso = {"name": "anisotropy", "value": 0.97, "verdict": "fail", "details": {}}
    groups = {s["group"]: [f["name"] for f in s["findings"]] for s in build_plan([dup, gap, aniso])}
    assert groups["query_pipeline"] == ["query_gap"] and "collapse" in groups["shared_direction"]


def test_islands_made_of_near_copies_go_with_the_duplicates():
    dup = {"name": "collapse", "value": 0.03, "verdict": "warn",
           "details": {"exact_duplicate_ratio": 0.0, "largest_near_duplicate_groups": [120, 40]}}
    isl = {"name": "connectivity", "value": 0.02, "verdict": "warn",
           "details": {"min_island_size": 500}}
    plan = build_plan([dup, isl])
    assert [s["group"] for s in plan] == ["input_integrity"]
    placed = {f["name"]: f["placed_because"] for f in plan[0]["findings"]}
    assert "island size" in placed["connectivity"]
    # copies bigger than the island size are a component of their own, not an island
    big = {**dup, "details": {**dup["details"], "largest_near_duplicate_groups": [900]}}
    assert [s["group"] for s in build_plan([big, isl])] == ["input_integrity", "corpus_structure"]
    # without a duplicate problem islands are corpus structure
    assert [s["group"] for s in build_plan([isl])] == ["corpus_structure"]


def test_every_planned_finding_says_why_it_is_there():
    x = _unit(_clustered(1500))
    x[:, 0] *= 30.0
    for step in audit(x, k=10).plan:
        for f in step["findings"]:
            assert f["placed_because"], (step["group"], f["name"])
    reasons = {f["name"]: f["placed_because"] for s in audit(x, k=10).plan for f in s["findings"]}
    assert "rogue_dimensions is not ok" in reasons["norms"]
    assert reasons["rogue_dimensions"] == "this group's cause covers it"


def test_inconclusive_findings_are_marked_in_the_plan():
    findings = [{"name": "connectivity", "value": 0.02, "verdict": "warn", "details": {},
                 "inconclusive": True}]
    step = build_plan(findings)[0]
    assert step["only_inconclusive"] is True and step["findings"][0]["inconclusive"] is True


def test_plan_is_deterministic_and_in_json():
    x = _clustered(1200)
    x += 6.0
    a, b = audit(x, k=10).to_json(), audit(x, k=10).to_json()
    assert a == b and json.loads(a)["plan"]


# -------------------------------------------------------------------------- incremental

@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_extend_equals_full_exact_knn(metric):
    x = _clustered(1300, seed=3) * np.random.default_rng(4).uniform(0.5, 2, (1300, 1)).astype(np.float32)
    old_i, old_d = exact_knn(x[:1000], 10, metric=metric)
    ext_i, ext_d = extend_knn(x, old_i, old_d, 1000, 10, metric=metric)
    full_i, full_d = exact_knn(x, 10, metric=metric)
    np.testing.assert_allclose(ext_d, full_d, rtol=1e-6, atol=2e-6)  # dot values reach ~100
    assert (ext_i == full_i).mean() > 0.999  # identical except float32 near-ties


@pytest.mark.parametrize("added", [1, 4, 10, 37])
def test_extend_with_fewer_or_more_new_rows_than_k(added):
    x = _clustered(600 + added, seed=9)
    old_i, old_d = exact_knn(x[:600], 10)
    _, ext_d = extend_knn(x, old_i, old_d, 600, 10)
    _, full_d = exact_knn(x, 10)
    np.testing.assert_allclose(ext_d, full_d, atol=2e-6)


def test_cache_reuses_extends_and_rebuilds(tmp_path):
    x = _clustered(1300, seed=5)
    path = tmp_path / "knn.npz"
    _, info = cached_knn(x[:1000], 10, path)
    assert info["status"] == "created" and info["new_rows"] == 1000
    _, info = cached_knn(x[:1000], 10, path)
    assert info["status"] == "reused" and info["new_rows"] == 0
    out, info = cached_knn(x, 10, path)
    assert info["status"] == "extended" and info["cached_rows"] == 1000 and info["new_rows"] == 300
    _, full_d = exact_knn(x, 10)
    np.testing.assert_allclose(out["cosine"][1], full_d, atol=2e-6)
    changed = x.copy()
    changed[5] += 1.0  # one row re-embedded in place: only it and its neighbors are redone
    out, info = cached_knn(changed, 10, path)
    assert info["status"] == "updated" and info["changed_rows"] == 1
    assert 1 <= info["recomputed_rows"] < 100
    _, full_d = exact_knn(changed, 10)
    np.testing.assert_allclose(out["cosine"][1], full_d, atol=2e-6)
    many = changed.copy()
    many[::10] += 0.5  # a tenth changed, spread out: most lists hold one, so rebuilt
    _, info = cached_knn(many, 10, path)
    assert info["status"] == "created" and "130 changed rows would recompute" in info["reason"]
    _, info = cached_knn(many[:1200], 10, path)
    assert info["status"] == "created" and info["reason"] == "rows were removed"
    _, info = cached_knn(many[:1200, :16], 10, path)  # other dim
    assert info["status"] == "created" and info["reason"] == "other dim"
    z = np.load(path)
    assert str(z["fingerprint"]) == fingerprint(many[:1200, :16]) and int(z["n"]) == 1200


@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_refresh_after_changes_and_appends_equals_full_exact_knn(metric):
    rng = np.random.default_rng(11)
    x = _clustered(1300, seed=11) * rng.uniform(0.5, 2, (1300, 1)).astype(np.float32)
    old_i, old_d = exact_knn(x[:1000], 10, metric=metric)
    y = x.copy()
    moved = rng.choice(1000, 40, replace=False)
    y[moved] = _clustered(40, seed=12) * 1.5  # re-embedded in place
    y[moved[:5]] = y[moved[5:10]]  # some become copies of other changed rows
    got_i, got_d = refresh_knn(y, old_i, old_d, 1000, moved, 10, metric=metric)
    full_i, full_d = exact_knn(y, 10, metric=metric)
    np.testing.assert_allclose(got_d, full_d, rtol=1e-6, atol=2e-6)
    assert (got_i == full_i).mean() > 0.995  # identical except float32 near-ties
    with pytest.raises(ValueError, match="changed rows"):
        refresh_knn(y, old_i, old_d, 1000, [1000], 10, metric=metric)


def test_an_older_cache_is_rebuilt(tmp_path):
    x = _clustered(500, seed=13)
    np.savez(tmp_path / "old.npz", version=1, n=500, dim=32, k=10, fingerprint=fingerprint(x))
    _, info = cached_knn(x, 10, tmp_path / "old.npz")
    assert info["status"] == "created" and info["reason"] == "older cache format"
    assert int(np.load(tmp_path / "old.npz")["version"]) == 2


def test_cache_rejects_a_foreign_file(tmp_path):
    bad = tmp_path / "knn.npz"
    np.savez(bad, version=99)
    with pytest.raises(ValueError, match="version"):
        cached_knn(_clustered(100), 5, bad)


def test_audit_from_cache_matches_a_full_audit(tmp_path):
    rng = np.random.default_rng(6)
    x = _clustered(1400, seed=6) * rng.uniform(0.5, 2, (1400, 1)).astype(np.float32)
    cached_knn_for_audit(x[:1100], 10, tmp_path / "c.npz")
    knn, info = cached_knn_for_audit(x, 10, tmp_path / "c.npz")
    assert info["status"] == "extended" and info["metrics"] == ["cosine", "dot"]
    a = {f.name: f.value for f in audit(x, k=10, sample=0).findings}
    b = {f.name: f.value for f in audit(x, k=10, sample=0, knn=knn).findings}
    assert a.keys() == b.keys()
    for name, value in a.items():
        assert b[name] == pytest.approx(value, rel=1e-4, abs=1e-6), name


def test_audit_knn_needs_every_row_and_matching_shape():
    x = _clustered(600)
    knn = {"cosine": exact_knn(x, 10)}
    with pytest.raises(ValueError, match="every row"):
        audit(x, k=10, sample=300, knn=knn)
    with pytest.raises(ValueError, match="shaped"):
        audit(x[:500], k=10, sample=0, knn=knn)


def test_cli_knn_cache(tmp_path, capsys):
    x = _clustered(1200, seed=8)
    np.save(tmp_path / "a.npy", x[:900])
    np.save(tmp_path / "b.npy", x)
    cache = tmp_path / "knn.npz"
    main(["audit", str(tmp_path / "a.npy"), "--knn-cache", str(cache), "--json"])
    assert json.loads(capsys.readouterr().out)["knn_cache"]["status"] == "created"
    main(["audit", str(tmp_path / "b.npy"), "--knn-cache", str(cache)])
    assert "extended, 300 rows computed" in capsys.readouterr().out
    y = np.load(tmp_path / "b.npy")
    y[3] += 2.0
    np.save(tmp_path / "c.npy", y)
    main(["audit", str(tmp_path / "c.npy"), "--knn-cache", str(cache)])
    assert "updated, 1 changed," in capsys.readouterr().out
    assert main(["audit", str(tmp_path / "b.npy"), "--knn-cache", str(cache),
                 "--sample", "500"]) == 2
