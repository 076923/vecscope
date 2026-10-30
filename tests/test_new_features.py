"""Checks for what the local copy adds on top of upstream 0.5.0 and upstream has no tests for:
sweep latency percentiles and index_bytes, compare with supplied neighbors_a / neighbors_b,
and the audit gate (``gate_decision``, ``--fail-on none``)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vecscope import audit
from vecscope._knn import exact_knn, normalize, row_overlap
from vecscope.cli import gate_decision, main
from vecscope.compare import compare


def _clustered(n=2000, d=32, centers=20, spread=0.6, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((centers, d))
    return (c[rng.integers(0, centers, n)] + rng.standard_normal((n, d)) * spread).astype(np.float32)


# ----------------------------------------------------------------------- sweep: latency, size

faiss = pytest.importorskip("faiss")
from vecscope.sweep import render_sweep, sweep


@pytest.fixture(scope="module")
def small_sweep():
    return sweep(_clustered(3000), k=5, n_queries=150, latency_queries=40, device="cpu",
                 timing_repeats=1)


def test_latency_percentiles_are_ordered_and_positive(small_sweep):
    assert small_sweep.latency_queries == 40
    for p in small_sweep.points:
        assert p.latency_ms_p50 is not None and p.latency_ms_p50 > 0
        assert p.latency_ms_p50 <= p.latency_ms_p95 <= p.latency_ms_p99


def test_latency_queries_zero_skips_and_says_so():
    res = sweep(_clustered(1500), k=5, n_queries=100, latency_queries=0, device="cpu",
                timing_repeats=1)
    assert res.latency_queries == 0
    assert all(p.latency_ms_p50 is None and p.latency_ms_p99 is None for p in res.points)
    assert any("skipped" in c for c in res.caveats)
    assert res.recommendation["latency_ms_p95"] is None
    assert "p95=" not in render_sweep(res).splitlines()[-len(res.caveats) - 1]


def test_latency_queries_capped_at_query_count():
    res = sweep(_clustered(1500), k=5, n_queries=30, latency_queries=500, device="cpu",
                timing_repeats=1)
    assert res.latency_queries == 30


@pytest.mark.parametrize("bad", [-1, True, 1.5, "10"])
def test_latency_queries_validation(bad):
    with pytest.raises(ValueError, match="latency_queries"):
        sweep(_clustered(500), k=5, n_queries=20, latency_queries=bad, device="cpu")


def test_index_bytes_matches_independent_serialization(small_sweep):
    """index_bytes must be what faiss serializes for the same index, built independently."""
    x = normalize(_clustered(3000))
    n, d = x.shape
    by_index: dict[tuple, set] = {}
    for p in small_sweep.points:
        key = (p.index, p.params.get("M"), p.params.get("nlist"))
        by_index.setdefault(key, set()).add(p.index_bytes)
    # one size per built index: changing ef_search / nprobe does not rebuild
    assert all(len(v) == 1 for v in by_index.values())
    size = {key: next(iter(v)) for key, v in by_index.items()}
    for (kind, m, nlist), s in size.items():
        assert s >= n * d * 4  # at least the raw float32 vectors
        if kind == "hnsw":
            ref = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
            ref.hnsw.efConstruction = 100
            ref.add(x)
            # level assignment is seeded, so the serialized size must agree closely
            assert abs(s - faiss.serialize_index(ref).nbytes) / s < 0.02
        else:
            ref = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
            ref.train(x)
            ref.add(x)
            assert abs(s - faiss.serialize_index(ref).nbytes) / s < 0.01
    hnsw = [s for (kind, m, _), s in sorted(size.items(), key=lambda kv: kv[0][1] or 0)
            if kind == "hnsw"]
    assert hnsw == sorted(hnsw), "a larger M must not give a smaller graph"


def test_recommendation_carries_latency_and_size(small_sweep):
    r = small_sweep.recommendation
    match = [p for p in small_sweep.points if p.index == r["index"] and p.params == r["params"]]
    assert len(match) == 1
    assert r["latency_ms_p95"] == match[0].latency_ms_p95
    assert r["index_bytes"] == match[0].index_bytes
    text = render_sweep(small_sweep)
    assert "p95 ms" in text and "MiB" in text and "index=" in text


def test_to_dict_round_trips_through_json(small_sweep):
    d = json.loads(json.dumps(small_sweep.to_dict()))
    assert {"latency_ms_p50", "latency_ms_p95", "latency_ms_p99", "index_bytes"} <= set(
        d["points"][0])
    assert d["latency_queries"] == 40


def test_cli_sweep_latency_flag(tmp_path, capsys):
    p = tmp_path / "x.npy"
    np.save(p, _clustered(1200))
    assert main(["sweep", str(p), "-k", "5", "--n-queries", "50", "--latency-queries", "0",
                 "--device", "cpu", "--timing-repeats", "1", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["latency_queries"] == 0 and out["points"][0]["latency_ms_p50"] is None


def _gpu_ok():
    from vecscope.sweep import gpu_available
    return gpu_available(faiss)


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_ok(), reason="no faiss GPU")
def test_gpu_sweep_matches_cpu_recall_and_ivf_size():
    """The GPU exact baseline (faiss flat) and the CPU one (NumPy) must score HNSW the same;
    the IVF index copied back from the GPU serializes to the same size as a CPU IVF."""
    x = _clustered(4000, d=48)
    cpu = sweep(x, k=10, n_queries=200, device="cpu", timing_repeats=1, latency_queries=10)
    gpu = sweep(x, k=10, n_queries=200, device="gpu", timing_repeats=1, latency_queries=10)
    assert gpu.device == "gpu" and gpu.gpu_name
    for a, b in zip(cpu.points, gpu.points):
        if a.index == "hnsw":
            assert a.params == b.params
            assert abs(a.recall_at_k - b.recall_at_k) <= 0.01
    ivf_c = [p for p in cpu.points if p.index == "ivf"]
    ivf_g = [p for p in gpu.points if p.index == "ivf"]
    assert ivf_g and all(p.device == "gpu" for p in ivf_g)
    assert abs(ivf_c[0].index_bytes - ivf_g[0].index_bytes) / ivf_c[0].index_bytes < 0.05
    # exhaustive IVF (nprobe = nlist or the largest probed) must be near exact on both
    assert max(p.recall_at_k for p in ivf_g) >= 0.97


# ----------------------------------------------------------------- compare: neighbors_a / b

@pytest.fixture(scope="module")
def two_spaces():
    a = _clustered(1500, d=32, seed=1)
    rot = np.linalg.qr(np.random.default_rng(2).standard_normal((32, 32)))[0]
    b = (a @ rot + np.random.default_rng(3).standard_normal(a.shape) * 0.25).astype(np.float32)
    b[:10] = b[10:20]  # a few duplicates that only space B has
    return a, b


def test_supplied_exact_neighbors_equal_recomputed_on_full_rows(two_spaces):
    a, b = two_spaces
    k = 10
    ia, _ = exact_knn(normalize(a), k)
    ib, _ = exact_knn(normalize(b), k)
    rec = compare(a, b, k=k, sample=0)
    sup = compare(a, b, k=k, sample=0, neighbors_a=ia, neighbors_b=ib)
    assert rec.neighbor_source == "recomputed" and sup.neighbor_source == "supplied"
    assert rec.neighbor_scope == sup.neighbor_scope == "full"
    for f in ("nn1_agreement", "topk_overlap", "topk_overlap_p10", "rows_with_no_overlap",
              "near_duplicate_ratio_a", "near_duplicate_ratio_b", "near_duplicate_both"):
        assert getattr(rec, f) == pytest.approx(getattr(sup, f), abs=1e-6), f


def test_supplied_neighbors_with_self_first_are_handled(two_spaces):
    """An index queried with k + 1 returns self first; dropping it must give the same answer."""
    a, b = two_spaces
    k = 8
    ia, _ = exact_knn(normalize(a), k)
    ib, _ = exact_knn(normalize(b), k)
    n = a.shape[0]
    self_col = np.arange(n)[:, None]
    with_self_a = np.hstack([self_col, ia])
    with_self_b = np.hstack([self_col, ib])
    r1 = compare(a, b, k=k, sample=0, neighbors_a=ia, neighbors_b=ib)
    r2 = compare(a, b, k=k, sample=0, neighbors_a=with_self_a, neighbors_b=with_self_b)
    assert r1.topk_overlap == r2.topk_overlap and r1.nn1_agreement == r2.nn1_agreement


def test_sampled_compare_with_supplied_neighbors_is_full_corpus_answer(two_spaces):
    """With supplied full-corpus neighbors, a sampled compare must equal the full-corpus
    per-row agreement restricted to the sampled rows (computed here independently)."""
    a, b = two_spaces
    k, sample = 10, 400
    ia, _ = exact_knn(normalize(a), k)
    ib, _ = exact_knn(normalize(b), k)
    res = compare(a, b, k=k, sample=sample, seed=7, neighbors_a=ia, neighbors_b=ib)
    rows = np.sort(np.random.default_rng(7).choice(a.shape[0], sample, replace=False))
    assert res.sampled and res.neighbor_scope == "full" and res.n == sample
    expect = row_overlap(ia[rows], ib[rows]).mean()
    assert res.topk_overlap == pytest.approx(expect, abs=1e-12)
    # ids in disagreements index the input rows, and every reported row is a sampled row
    for d in res.sample_disagreements:
        assert d["row"] in set(rows.tolist())
        assert d["nearest_a"] == int(ia[d["row"], 0]) and d["nearest_b"] == int(ib[d["row"], 0])


def test_sampled_compare_without_neighbors_says_sample_scope(two_spaces):
    a, b = two_spaces
    res = compare(a, b, k=10, sample=400)
    assert res.neighbor_scope == "sample" and res.neighbor_source == "recomputed"
    assert "sample's own neighbor graph" in res.explanation
    # full-corpus answer differs from the sample's own graph: the doc's point
    ia, _ = exact_knn(normalize(a), 10)
    ib, _ = exact_knn(normalize(b), 10)
    full = compare(a, b, k=10, sample=400, neighbors_a=ia, neighbors_b=ib)
    assert full.topk_overlap != res.topk_overlap


@pytest.mark.parametrize("case,match", [
    ("one_side", "together"),
    ("shape", "shaped"),
    ("float", "integer"),
    ("range", "only"),
    ("too_few", "fewer than k"),
    ("empty", "no neighbor columns"),
])
def test_supplied_neighbors_validation(two_spaces, case, match):
    a, b = two_spaces
    n = a.shape[0]
    good = (np.arange(n)[:, None] + np.arange(1, 11)[None, :]) % n  # never the row itself
    kw = {"neighbors_a": good, "neighbors_b": good}
    if case == "one_side":
        kw = {"neighbors_a": good}
    elif case == "shape":
        kw["neighbors_b"] = good[:-1]
    elif case == "float":
        kw["neighbors_b"] = good.astype(float)
    elif case == "range":
        kw["neighbors_b"] = good + n
    elif case == "too_few":
        bad = good.copy()
        bad[:, 5:] = -1
        kw["neighbors_b"] = bad
    elif case == "empty":
        kw["neighbors_b"] = good[:, :0]
    with pytest.raises(ValueError, match=match):
        compare(a, b, k=10, sample=0, **kw)


def test_cli_compare_neighbors_must_come_in_pairs(tmp_path, capsys, two_spaces):
    a, b = two_spaces
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "b.npy", b)
    ia, _ = exact_knn(normalize(a), 10)
    np.save(tmp_path / "na.npy", ia)
    assert main(["compare", str(tmp_path / "a.npy"), str(tmp_path / "b.npy"), "--sample", "0",
                 "--neighbors-a", str(tmp_path / "na.npy")]) == 2
    assert "go together" in capsys.readouterr().err
    ib, _ = exact_knn(normalize(b), 10)
    np.save(tmp_path / "nb.npy", ib)
    assert main(["compare", str(tmp_path / "a.npy"), str(tmp_path / "b.npy"), "--sample", "500",
                 "--neighbors-a", str(tmp_path / "na.npy"), "--neighbors-b",
                 str(tmp_path / "nb.npy"), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["neighbor_scope"] == "full" and out["neighbor_source"] == "supplied"


# ---------------------------------------------------------------------------------- gate

DIFF_CLEAN = {"regressions": [], "value_regressions": []}
DIFF_VERDICT = {"regressions": ["hubness"], "value_regressions": []}
DIFF_VALUE = {"regressions": [], "value_regressions": ["collapse"]}


@pytest.mark.parametrize("verdict,fail_on,diff,code", [
    ("ok", "fail", None, 0), ("warn", "fail", None, 0), ("fail", "fail", None, 1),
    ("ok", "warn", None, 0), ("warn", "warn", None, 1), ("fail", "warn", None, 1),
    ("fail", "none", None, 0), ("warn", "none", None, 0),
    ("fail", "none", DIFF_CLEAN, 0), ("ok", "none", DIFF_VERDICT, 1),
    ("ok", "none", DIFF_VALUE, 1), ("fail", "fail", DIFF_CLEAN, 1),
    ("ok", "fail", DIFF_VALUE, 1),
])
def test_gate_decision_table(verdict, fail_on, diff, code):
    g = gate_decision(verdict, fail_on, diff)
    assert g["exit_code"] == code
    assert g["baseline"] is (diff is not None)
    assert bool(g["reasons"]) is bool(code)
    assert g["absolute"] is (fail_on != "none" and
                             {"ok": 0, "warn": 1, "fail": 2}[verdict] >=
                             {"ok": 0, "warn": 1, "fail": 2}[fail_on])


def _audit_json(tmp_path, x, name):
    p = tmp_path / f"{name}.npy"
    np.save(p, x)
    return p


def test_cli_fail_on_none_judges_only_the_change(tmp_path, capsys):
    """A corpus that already fails absolutely passes with --fail-on none against its own
    baseline, and fails once a finding regresses; the JSON carries the gate."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((800, 16)).astype(np.float32)
    base[:200] = base[0]  # collapse: fails on absolute thresholds
    pb = _audit_json(tmp_path, base, "base")
    assert main(["audit", str(pb), "--json"]) == 1
    baseline = tmp_path / "baseline.json"
    baseline.write_text(capsys.readouterr().out, encoding="utf-8")
    assert json.loads(baseline.read_text())["gate"]["absolute"] is True

    assert main(["audit", str(pb), "--baseline", str(baseline), "--fail-on", "none",
                 "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["gate"] == {**out["gate"], "exit_code": 0, "absolute": False, "baseline": True}

    worse = base.copy()
    worse[:400] = base[0]  # more collapse: value worse, verdict stays fail
    pw = _audit_json(tmp_path, worse, "worse")
    tol = tmp_path / "tol.json"
    tol.write_text(json.dumps({"collapse": 0.01}))
    assert main(["audit", str(pw), "--baseline", str(baseline), "--fail-on", "none",
                 "--tolerances", str(tol), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert "collapse" in out["gate"]["value_regressions"]
    assert out["gate"]["absolute"] is False

    assert main(["audit", str(pw), "--baseline", str(baseline), "--fail-on", "none",
                 "--tolerances", str(tol)]) == 1
    assert "gate: FAIL" in capsys.readouterr().out


def test_cli_text_gate_line_names_the_rule(tmp_path, capsys):
    x = _clustered(600, d=16)
    p = _audit_json(tmp_path, x, "x")
    main(["audit", str(p), "--fail-on", "none"])
    assert "gate: pass (absolute thresholds)" in capsys.readouterr().out
    ev = audit(x)
    b = tmp_path / "b.json"
    b.write_text(ev.to_json())
    main(["audit", str(p), "--fail-on", "none", "--baseline", str(b)])
    assert "gate: pass (baseline)" in capsys.readouterr().out
