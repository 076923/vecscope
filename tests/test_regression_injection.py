"""Injected-defect regressions: a known defect must move its own finding in the expected
direction, a sampled audit must agree with the full one on the distribution findings, and the
compare / CI gate must say what their numbers are about.

These complement tests/test_scenarios.py (static synthetic corpora) with the defects an
agent meets while changing a pipeline: a query embedding model that does not match the
corpus, an ANN index whose search parameter was lowered, a sampled compare that recomputed
its neighbors, and a CI gate that must separate the absolute thresholds from the baseline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vecscope import audit, compare
from vecscope._knn import exact_knn, normalize


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def structured(rng, n, d=64, centers=40, spread=0.6):
    c = rng.standard_normal((centers, d))
    return c[rng.integers(0, centers, n)] + rng.standard_normal((n, d)) * spread


def by_name(ev):
    return {f.name: f for f in ev.findings}


def test_query_model_mismatch_trips_query_findings_only():
    """Queries embedded by another model (here: the corpus space under a random rotation)
    score query_gap, query_shift (against the healthy run) and ood_queries = fail while the
    corpus-side findings do not move; a fresh healthy query set stays ok against the same
    reference."""
    rng = np.random.default_rng(7)
    centers = rng.standard_normal((40, 64))
    corpus = unit(centers[rng.integers(0, 40, 3000)] + rng.standard_normal((3000, 64)) * 0.6)
    same_model = unit(centers[rng.integers(0, 40, 300)] + rng.standard_normal((300, 64)) * 0.6)
    rot, _ = np.linalg.qr(rng.standard_normal((64, 64)))
    other_model = unit(same_model @ rot.astype(np.float32))
    good_ev = audit(corpus, k=10, queries=same_model)
    good = by_name(good_ev)
    bad = by_name(audit(corpus, k=10, queries=other_model, query_baseline=good_ev.to_dict()))
    fresh = unit(centers[rng.integers(0, 40, 300)] + rng.standard_normal((300, 64)) * 0.6)
    healthy = by_name(audit(corpus, k=10, queries=fresh, query_baseline=good_ev.to_dict()))
    assert good["ood_queries"].verdict == "ok" and good["query_gap"].verdict == "ok"
    assert "query_shift" not in good  # only with a reference
    assert bad["ood_queries"].verdict == "fail" and bad["ood_queries"].value > 0.9
    assert bad["query_gap"].verdict == "fail" and bad["query_gap"].value > 3.0
    assert bad["query_shift"].verdict == "fail" and bad["query_shift"].value > 0.9
    assert healthy["query_shift"].verdict == "ok" and healthy["query_shift"].value < 0.1
    corpus_side = [n for n in good if n not in {"ood_queries", "query_gap", "query_hubness"}]
    assert all(good[n].value == bad[n].value for n in corpus_side)  # queries never change the corpus


def test_lowering_ann_search_parameter_lowers_ann_recall():
    """An HNSW index searched with a smaller ef_search returns fewer true neighbors; the
    finding must fall with it and cross into warn/fail before the index is unusable."""
    faiss = pytest.importorskip("faiss")
    rng = np.random.default_rng(3)
    x = unit(structured(rng, 4000, d=48, centers=200, spread=0.9))
    index = faiss.IndexHNSWFlat(48, 8, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 40
    index.add(x)
    recalls = {}
    for ef in (1, 4, 16, 128):
        index.hnsw.efSearch = ef
        _, ids = index.search(x, 11)  # k + 1: the item itself comes back first
        ids = ids.astype(np.int64)
        recalls[ef] = by_name(audit(x, k=10, ann_neighbors=ids, sample=None))["ann_recall"]
    values = [recalls[ef].value for ef in (1, 4, 16, 128)]
    assert values == sorted(values), values  # recall is monotone in ef_search
    assert recalls[128].verdict == "ok" and recalls[128].value > 0.95
    assert recalls[1].verdict == "fail" and recalls[1].value < 0.85
    worse = {"ok": 0, "warn": 1, "fail": 2}
    assert worse[recalls[4].verdict] >= worse[recalls[16].verdict] >= worse[recalls[128].verdict]


def test_duplicate_injection_is_proportional_and_only_hits_collapse():
    rng = np.random.default_rng(5)
    base = unit(structured(rng, 2000))
    clean = by_name(audit(base, k=10))
    for share in (0.05, 0.15):
        extra = int(2000 * share)
        hit = by_name(audit(np.concatenate([base, base[:extra]]), k=10))
        expected = 2 * extra / (2000 + extra)  # every copied row and its original are flagged
        assert abs(hit["collapse"].value - expected) < 0.01
        assert hit["collapse"].verdict == ("warn" if share == 0.05 else "fail")
        for name in ("anisotropy", "effective_rank", "rogue_dimensions", "norms"):
            assert hit[name].verdict == clean[name].verdict


def test_sampled_audit_states_what_each_finding_can_say():
    """What a sample can stand for, finding by finding. The distribution findings give the
    same verdict and a close value on a 25% or 50% sample as on every row. hubness and
    antihubs fall with the sample size (k-occurrence concentrates with density), so a sampled
    value is a lower bound. A duplicate pair is seen only when both rows were sampled, so the
    sampled collapse share is about fraction x the full share. The explanations say so."""
    rng = np.random.default_rng(11)
    x = unit(structured(rng, 6000, d=96, centers=80))
    x[:120] = x[120:240]  # 2% duplicate rows, spread through the corpus
    full = by_name(audit(x, k=10, sample=None))
    assert "sample" not in full["hubness"].explanation
    stable = ("anisotropy", "effective_rank", "rogue_dimensions", "norms", "metric_mismatch",
              "distance_concentration")
    for size in (1500, 3000):
        fraction = size / 6000
        part = by_name(audit(x, k=10, sample=size))
        for name in stable:
            assert part[name].verdict == full[name].verdict, (size, name)
            scale = max(abs(full[name].value), 1e-3)
            assert abs(part[name].value - full[name].value) <= 0.05 * scale + 0.02, (size, name)
        for name in ("hubness", "antihubs"):
            assert part[name].value < full[name].value, (size, name)
            assert "lower bound" in part[name].explanation
        seen = part["collapse"].value / full["collapse"].value
        assert abs(seen - fraction) < 0.15, (size, seen)
        assert f"divided by {fraction:.2f}" in part["collapse"].explanation
        assert f"{fraction:.0%} sample" in part["outliers"].explanation
        assert "sample's k-NN graph" in part["connectivity"].explanation
    # sample repeats put the full-corpus value of a distribution finding inside the range
    rep = by_name(audit(x, k=10, sample=1500, sample_repeats=3))
    for name in ("anisotropy", "distance_concentration", "effective_rank"):
        lo, hi = rep[name].value_range
        slack = 0.05 * max(abs(full[name].value), 1e-3)
        assert lo - slack <= full[name].value <= hi + slack, (name, lo, hi, full[name].value)


def test_compare_sample_scope_differs_from_full_scope_and_says_so():
    """Neighbors recomputed inside a sample are not the full corpus's; the result must label
    the scope, and supplied full-corpus neighbors must reproduce the unsampled numbers."""
    rng = np.random.default_rng(0)
    c = rng.standard_normal((30, 16))
    a = (c[rng.integers(0, 30, 2000)] + rng.standard_normal((2000, 16)) * 0.5).astype(np.float32)
    rot, _ = np.linalg.qr(rng.standard_normal((16, 16)))
    b = (a @ rot.astype(np.float32) + rng.standard_normal(a.shape) * 0.2).astype(np.float32)
    full = compare(a, b, k=5, sample=0)
    assert (full.neighbor_scope, full.neighbor_source) == ("full", "recomputed")
    sampled = compare(a, b, k=5, sample=500)
    assert (sampled.neighbor_scope, sampled.neighbor_source) == ("sample", "recomputed")
    assert "sample's own neighbor graph" in sampled.explanation
    assert sampled.topk_overlap > full.topk_overlap + 0.1  # substitutes agree more often
    ia, _ = exact_knn(normalize(a), 6)
    ib, _ = exact_knn(normalize(b), 6)
    supplied_all = compare(a, b, k=5, sample=0, neighbors_a=ia, neighbors_b=ib)
    assert (supplied_all.neighbor_scope, supplied_all.neighbor_source) == ("full", "supplied")
    assert supplied_all.topk_overlap == full.topk_overlap
    assert supplied_all.nn1_agreement == full.nn1_agreement
    assert supplied_all.near_duplicate_ratio_a == full.near_duplicate_ratio_a
    supplied_sampled = compare(a, b, k=5, sample=500, neighbors_a=ia, neighbors_b=ib)
    assert supplied_sampled.sampled and supplied_sampled.neighbor_scope == "full"
    assert abs(supplied_sampled.topk_overlap - full.topk_overlap) < 0.05
    assert "supplied lists for a sample of 500 rows" in supplied_sampled.explanation
    rows = {d["row"] for d in supplied_sampled.sample_disagreements}
    assert all(0 <= d["nearest_a"] < 2000 for d in supplied_sampled.sample_disagreements)
    assert rows <= set(range(2000))
    # an index that returns the item itself first still works when it was queried with k + 1
    with_self = np.concatenate([np.arange(2000)[:, None], ia[:, :5]], axis=1)
    assert compare(a, b, k=5, sample=0, neighbors_a=with_self,
                   neighbors_b=ib).topk_overlap == full.topk_overlap
    with pytest.raises(ValueError, match="together"):
        compare(a, b, k=5, neighbors_a=ia)
    with pytest.raises(ValueError, match="fewer than k"):
        compare(a, b, k=5, neighbors_a=with_self[:, :5], neighbors_b=ib)
    with pytest.raises(ValueError, match="only 2000 rows"):
        compare(a, b, k=5, neighbors_a=ia + 1, neighbors_b=ib)
    with pytest.raises(ValueError, match="integer"):
        compare(a, b, k=5, neighbors_a=ia.astype(np.float32), neighbors_b=ib)
    with pytest.raises(ValueError, match="shaped"):
        compare(a, b, k=5, neighbors_a=ia[:-1], neighbors_b=ib)


def test_compare_cli_and_mcp_take_neighbor_files(tmp_path, capsys):
    from vecscope.cli import main

    rng = np.random.default_rng(1)
    a = rng.standard_normal((600, 8)).astype(np.float32)
    b = (a + rng.standard_normal(a.shape) * 0.3).astype(np.float32)
    ia, _ = exact_knn(normalize(a), 5)
    ib, _ = exact_knn(normalize(b), 5)
    for name, arr in (("a", a), ("b", b), ("na", ia), ("nb", ib)):
        np.save(tmp_path / f"{name}.npy", arr)
    args = ["compare", str(tmp_path / "a.npy"), str(tmp_path / "b.npy"), "-k", "5",
            "--sample", "200", "--json"]
    assert main(args) == 0
    plain = json.loads(capsys.readouterr().out)
    assert plain["neighbor_scope"] == "sample"
    assert main(args + ["--neighbors-a", str(tmp_path / "na.npy"),
                        "--neighbors-b", str(tmp_path / "nb.npy")]) == 0
    sup = json.loads(capsys.readouterr().out)
    assert sup["neighbor_scope"] == "full" and sup["neighbor_source"] == "supplied"
    assert main(args + ["--neighbors-a", str(tmp_path / "na.npy")]) == 2
    assert "go together" in capsys.readouterr().err
    assert main(args[:-1]) == 0
    assert "neighbors=recomputed/sample" in capsys.readouterr().out
    pytest.importorskip("mcp")
    from vecscope.mcp_server import compare_embeddings

    res = compare_embeddings(str(tmp_path / "a.npy"), str(tmp_path / "b.npy"), k=5, sample=200,
                             neighbors_a_path=str(tmp_path / "na.npy"),
                             neighbors_b_path=str(tmp_path / "nb.npy"))
    assert res["neighbor_source"] == "supplied" and res["topk_overlap"] == sup["topk_overlap"]
    with pytest.raises(Exception, match="go together"):
        compare_embeddings(str(tmp_path / "a.npy"), str(tmp_path / "b.npy"), k=5,
                           neighbors_a_path=str(tmp_path / "na.npy"))


def test_gate_separates_absolute_thresholds_from_baseline(tmp_path, capsys):
    """A corpus that fails the default thresholds but did not regress passes with
    --fail-on none --baseline; the gate names which rule decided."""
    from vecscope.cli import gate_decision, main

    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 32)).astype(np.float32) + 2.5  # anisotropy fails, collapse ok
    path = tmp_path / "x.npy"
    np.save(path, x)
    base = tmp_path / "base.json"
    base.write_text(audit(x, k=5).to_json(), encoding="utf-8")
    assert main(["audit", str(path), "-k", "5", "--json"]) == 1
    gate = json.loads(capsys.readouterr().out)["gate"]
    assert gate["absolute"] is True and gate["baseline"] is False and gate["exit_code"] == 1
    assert gate["reasons"] == ["verdict fail reaches --fail-on fail"]
    assert main(["audit", str(path), "-k", "5", "--fail-on", "none", "--json"]) == 0
    gate = json.loads(capsys.readouterr().out)["gate"]
    assert gate["absolute"] is False and gate["exit_code"] == 0 and gate["reasons"] == []
    # baseline only: same corpus, no regression, absolute rule off
    assert main(["audit", str(path), "-k", "5", "--fail-on", "none", "--baseline", str(base)]) == 0
    assert "gate: pass (baseline)" in capsys.readouterr().out
    # baseline only, but the corpus regressed: duplicates added
    np.save(tmp_path / "worse.npy", np.concatenate([x, x[:100]]))
    assert main(["audit", str(tmp_path / "worse.npy"), "-k", "5", "--fail-on", "none",
                 "--baseline", str(base), "--json"]) == 1
    gate = json.loads(capsys.readouterr().out)["gate"]
    assert gate["absolute"] is False and "collapse" in gate["verdict_regressions"]
    assert len(gate["reasons"]) == 1 and gate["reasons"][0].startswith("verdict worse than baseline:")
    assert "collapse" in gate["reasons"][0]
    assert main(["audit", str(tmp_path / "worse.npy"), "-k", "5", "--baseline", str(base)]) == 1
    out = capsys.readouterr().out
    assert "gate: FAIL; verdict fail reaches --fail-on fail; verdict worse than baseline:" in out
    d = {"regressions": [], "value_regressions": ["hubness"]}
    g = gate_decision("warn", "fail", d)
    assert g["exit_code"] == 1 and g["reasons"] == ["value worse than baseline beyond tolerance: hubness"]
    assert gate_decision("warn", "warn", None)["reasons"] == ["verdict warn reaches --fail-on warn"]
    assert gate_decision("fail", "none", None)["exit_code"] == 0
