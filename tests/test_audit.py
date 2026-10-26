import numpy as np

from vecscope import audit
from vecscope.diagnostics import ann_recall


def test_isotropic_gaussian_is_ok():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 64)).astype(np.float32)
    ev = audit(x, k=10)
    by = {f.name: f for f in ev.findings}
    assert by["anisotropy"].verdict == "ok"
    assert by["collapse"].verdict == "ok"
    assert ev.verdict in {"ok", "warn"}


def test_shared_offset_triggers_anisotropy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 64)).astype(np.float32) + 5.0
    ev = audit(x, k=10)
    by = {f.name: f for f in ev.findings}
    assert by["anisotropy"].verdict == "fail"


def test_duplicates_trigger_collapse():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((500, 32)).astype(np.float32)
    x = np.concatenate([base, base[:200]])
    ev = audit(x, k=5)
    by = {f.name: f for f in ev.findings}
    assert by["collapse"].verdict == "fail"
    assert by["collapse"].details["exact_duplicate_ratio"] > 0.2


def test_ann_recall_perfect_and_broken():
    exact = np.array([[1, 2, 3], [0, 2, 3]])
    assert ann_recall(exact, exact)["recall_at_k"] == 1.0
    assert ann_recall(exact, np.array([[9, 9, 9], [9, 9, 9]]))["recall_at_k"] == 0.0


def test_json_roundtrip():
    rng = np.random.default_rng(1)
    ev = audit(rng.standard_normal((300, 16)).astype(np.float32), k=5)
    s = ev.to_json()
    assert '"tool": "vecscope"' in s and '"verdict"' in s


def test_hub_cluster_raises_k_occurrence():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1500, 32)).astype(np.float32)
    center = rng.standard_normal((1, 32)).astype(np.float32)
    x[:100] = center + rng.standard_normal((100, 32)).astype(np.float32) * 0.05
    ev = audit(x, k=10)
    by = {f.name: f for f in ev.findings}
    assert by["hubness"].details["max_k_occurrence"] > 10


def test_sampling_caps_rows_and_refuses_ann():
    import pytest

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1000, 8)).astype(np.float32)
    ev = audit(x, k=5, sample=200)
    assert ev.n == 200 and ev.n_total == 1000 and ev.sampled
    with pytest.raises(ValueError, match="sample"):
        audit(x, k=5, sample=200, ann_neighbors=np.zeros((1000, 5), dtype=int))


def test_rejects_nan_and_large_k():
    import pytest

    x = np.ones((10, 4), dtype=np.float32)
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        audit(x, k=3)
    with pytest.raises(ValueError):
        audit(np.ones((10, 4), dtype=np.float32), k=10)


def test_cli_json_and_exit_code(tmp_path, capsys):
    from vecscope.cli import main

    rng = np.random.default_rng(0)
    path = tmp_path / "v.npy"
    np.save(path, rng.standard_normal((500, 16)).astype(np.float32) + 5.0)
    code = main(["audit", str(path), "--json"])
    out = capsys.readouterr().out
    assert code == 1 and '"verdict": "fail"' in out
    assert main(["audit", str(path), "-k", "5"]) == 1


def test_load_json_and_jsonl(tmp_path):
    import json

    from vecscope.io import load_vectors

    (tmp_path / "a.jsonl").write_text(
        "\n".join(json.dumps({"id": i, "embedding": [i, 0.5]}) for i in range(3)), encoding="utf-8"
    )
    (tmp_path / "b.json").write_text(json.dumps([[1, 2], [3, 4]]), encoding="utf-8")
    assert load_vectors(tmp_path / "a.jsonl").shape == (3, 2)
    assert load_vectors(tmp_path / "b.json").shape == (2, 2)


def test_findings_carry_thresholds():
    rng = np.random.default_rng(0)
    ev = audit(rng.standard_normal((300, 16)).astype(np.float32), k=5)
    for f in ev.findings:
        assert f.warn_at is not None and f.fail_at is not None
    by = {f.name: f for f in ev.findings}
    assert by["effective_rank"].lower_is_worse and not by["hubness"].lower_is_worse
    assert by["antihubs"].metric == "antihub_excess"


def test_centering_evidence_and_ranges():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1000, 32)).astype(np.float32) + 3.0
    ev = audit(x, k=5)
    by = {f.name: f for f in ev.findings}
    d = by["anisotropy"].details
    assert d["mean_random_cosine"] > 0.6 and d["mean_random_cosine_after_centering"] < 0.05
    assert d["mean_vector_norm"] > 0.8
    assert all(f.healthy_range.startswith("ok") for f in ev.findings)


def _by(ev):
    return {f.name: f for f in ev.findings}


def test_rogue_dimension_detected():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1000, 64)).astype(np.float32)
    x[:, 3] *= 20.0
    f = _by(audit(x, k=5))["rogue_dimensions"]
    assert f.verdict == "fail" and f.details["rogue_dims"][0] == 3


def test_norm_spread_triggers_metric_mismatch():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1000, 32)).astype(np.float32)
    x *= rng.uniform(0.2, 5.0, size=(1000, 1)).astype(np.float32)
    by = _by(audit(x, k=5))
    assert by["norms"].verdict == "fail"
    assert by["metric_mismatch"].value < 0.9


def test_unit_vectors_skip_metric_mismatch():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 32)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    by = _by(audit(x, k=5))
    assert by["norms"].verdict == "ok" and by["metric_mismatch"].value == 1.0


def test_outliers_and_sample_ids():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((10, 64)).astype(np.float32)
    clustered = centers[rng.integers(0, 10, 980)] + rng.standard_normal((980, 64)).astype(np.float32) * 0.1
    junk = rng.standard_normal((20, 64)).astype(np.float32)  # far from every cluster
    x = np.concatenate([junk, clustered])
    f = _by(audit(x, k=5))["outliers"]
    assert f.value >= 0.015 and set(f.details["sample_ids"]) <= set(range(20))


def test_small_islands_fail_but_big_clusters_pass():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((5, 32)).astype(np.float32) * 10
    big = centers[rng.integers(0, 5, 1000)] + rng.standard_normal((1000, 32)).astype(np.float32)
    assert _by(audit(big, k=3))["connectivity"].verdict == "ok"
    islands = rng.standard_normal((30, 32)).astype(np.float32) * 10
    islands = np.repeat(islands, 5, axis=0) + rng.standard_normal((150, 32)).astype(np.float32) * 0.3
    f = _by(audit(np.concatenate([big, islands]), k=3))["connectivity"]
    assert f.verdict == "fail" and f.details["n_islands"] >= 25


def test_constant_inputs_do_not_crash():
    for x in (np.ones((50, 8), np.float32), np.zeros((50, 8), np.float32)):
        ev = audit(x, k=5)
        assert all(np.isfinite(f.value) for f in ev.findings)


def test_duplicates_do_not_poison_outliers():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((500, 32)).astype(np.float32)
    by = _by(audit(np.concatenate([base, base[:300]]), k=5))
    assert by["outliers"].verdict == "ok"
    assert by["distance_concentration"].value < 100


def test_queries_in_and_out_of_distribution():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((8, 32)).astype(np.float32) * 5
    corpus = centers[rng.integers(0, 8, 2000)] + rng.standard_normal((2000, 32)).astype(np.float32)
    in_dist = centers[rng.integers(0, 8, 200)] + rng.standard_normal((200, 32)).astype(np.float32)
    ood = rng.standard_normal((200, 32)).astype(np.float32) * 5
    good = _by(audit(corpus, k=5, queries=in_dist))
    bad = _by(audit(corpus, k=5, queries=ood))
    assert good["ood_queries"].verdict == "ok"
    assert bad["ood_queries"].verdict == "fail" and len(bad["ood_queries"].details["ood_sample_ids"]) > 0


def test_queries_shape_is_validated():
    import pytest

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        audit(rng.standard_normal((100, 8)).astype(np.float32), k=3,
              queries=rng.standard_normal((10, 4)).astype(np.float32))


def test_cluster_report_sizes_and_ids():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((600, 16)).astype(np.float32)
    ev = audit(x, k=5, clusters=4)
    assert len(ev.clusters) == 4 and sum(c["size"] for c in ev.clusters) == 600
    assert all(0 <= i < 600 for c in ev.clusters for i in c["sample_ids"])
    assert '"clusters"' in ev.to_json()


def test_sweep_recommends_a_config():
    pytest = __import__("pytest")
    pytest.importorskip("faiss")
    from vecscope.sweep import sweep

    rng = np.random.default_rng(0)
    x = rng.standard_normal((3000, 32)).astype(np.float32)
    res = sweep(x, k=5, n_queries=200)
    assert res.points and res.recommendation["index"] in {"hnsw", "ivf"}
    assert all(0.0 <= p.recall_at_k <= 1.0 for p in res.points)


def test_sweep_device_selection():
    pytest = __import__("pytest")
    faiss = pytest.importorskip("faiss")
    from vecscope.sweep import _resolve_device, gpu_available, sweep

    assert _resolve_device(faiss, "cpu") == "cpu"
    assert _resolve_device(faiss, "auto") == ("gpu" if gpu_available(faiss) else "cpu")
    with pytest.raises(ValueError):
        _resolve_device(faiss, "tpu")
    if not gpu_available(faiss):
        with pytest.raises(RuntimeError):
            _resolve_device(faiss, "gpu")
    rng = np.random.default_rng(0)
    res = sweep(rng.standard_normal((1000, 16)).astype(np.float32), k=5, n_queries=50, device="cpu")
    assert res.device == "cpu" and all(p.device == "cpu" for p in res.points)


def test_mcp_server_exposes_tools():
    pytest = __import__("pytest")
    pytest.importorskip("mcp")
    import asyncio

    from vecscope.mcp_server import mcp as server

    tools = asyncio.run(server.list_tools())
    by = {t.name: t for t in tools}
    assert set(by) >= {"audit_embeddings", "sweep_index", "describe_findings"}
    ann = by["audit_embeddings"].annotations
    if ann is not None:  # older mcp may not carry annotations
        assert getattr(ann, "readOnlyHint", None) or getattr(ann, "read_only_hint", None)


def test_mcp_audit_summary_and_verbose(tmp_path):
    import json

    pytest = __import__("pytest")
    pytest.importorskip("mcp")
    from vecscope.mcp_server import audit_embeddings, describe_findings

    rng = np.random.default_rng(0)
    path = tmp_path / "v.npy"
    np.save(path, rng.standard_normal((400, 16)).astype(np.float32) + 5.0)
    full = audit_embeddings(str(path), k=5)
    small = audit_embeddings(str(path), k=5, verbose=False)
    assert list(full)[:2] == ["summary", "verdict"] and "anisotropy" in full["summary"]
    assert len(json.dumps(small)) < len(json.dumps(full))
    ok = [f for f in small["findings"] if f["verdict"] == "ok"]
    assert ok and all(f["details"] == {} and f["suggested_action"] == "" for f in ok)
    bad = [f for f in small["findings"] if f["verdict"] != "ok"]
    assert all(f["suggested_action"] for f in bad)
    desc = describe_findings()
    assert set(desc) == {f["name"] for f in full["findings"]} | {"ann_recall", "ood_queries", "query_hubness"}


def test_threshold_override_changes_verdict_and_is_reported():
    import pytest

    rng = np.random.default_rng(0)
    x = rng.standard_normal((800, 32)).astype(np.float32) + 0.5  # mild offset
    base = _by(audit(x, k=5))["anisotropy"]
    strict = _by(audit(x, k=5, thresholds={"anisotropy.mean_random_cosine": (0.05, 0.1)}))
    assert base.verdict == "ok" and strict["anisotropy"].verdict == "fail"
    assert strict["anisotropy"].warn_at == 0.05
    with pytest.raises(ValueError):
        audit(x, k=5, thresholds={"nope": (1, 2)})
    with pytest.raises(ValueError):  # direction flipped
        audit(x, k=5, thresholds={"ann_recall.recall_at_k": (0.5, 0.9)})


def test_findings_carry_actions():
    rng = np.random.default_rng(0)
    ev = audit(rng.standard_normal((300, 16)).astype(np.float32), k=5)
    for f in ev.findings:
        assert f.likely_causes and f.suggested_action and f.verify_by
    assert "hubness.skewness" in ev.thresholds


def test_connectivity_handles_long_chains():
    from vecscope.diagnostics.geometry import connectivity

    n = 6000  # a path graph: diameter n, one component, no islands
    knn = np.stack([np.clip(np.arange(n) - 1, 0, n - 1), np.clip(np.arange(n) + 1, 0, n - 1)], axis=1)
    c = connectivity(knn)
    assert c["n_components"] == 1 and c["island_ratio"] == 0.0


def test_sweep_reaches_full_recall_with_duplicates_and_zero_rows():
    pytest = __import__("pytest")
    pytest.importorskip("faiss")
    from vecscope.sweep import sweep

    rng = np.random.default_rng(0)
    base = rng.standard_normal((600, 16)).astype(np.float32)
    res = sweep(np.concatenate([base, base]), k=5, n_queries=100, device="cpu")
    assert max(p.recall_at_k for p in res.points) >= 0.99  # duplicates no longer cap recall
    # zero rows have no meaningful neighbors (all ties); the sweep must still run cleanly
    res = sweep(np.concatenate([base, np.zeros((50, 16), np.float32)]), k=5, n_queries=100, device="cpu")
    assert all(0.0 <= p.recall_at_k <= 1.0 for p in res.points)


def test_ann_validation_and_self_ids():
    import pytest

    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 8)).astype(np.float32)
    with pytest.raises(ValueError):
        audit(x, k=5, ann_neighbors=np.zeros((50, 5), int))
    with pytest.raises(ValueError):
        audit(x, k=5, ann_neighbors=np.full((200, 5), 999))
    with pytest.raises(ValueError):
        audit(x, k=5, ann_neighbors=np.zeros((200, 5), int), sample=100)
    from vecscope._knn import exact_knn

    exact, _ = exact_knn(x, 5)
    with_self = np.concatenate([np.arange(200)[:, None], exact], axis=1)  # index returned self first
    r = _by(audit(x, k=5, ann_neighbors=with_self, sample=None))["ann_recall"]
    assert r.value == 1.0


def test_parquet_picks_vector_column_by_type(tmp_path):
    pa = __import__("pytest").importorskip("pyarrow")
    import pyarrow.parquet as pq

    from vecscope.io import load_vectors

    t = pa.table({"id": [1, 2], "embedding_model": ["m", "m"], "vector_id": [7, 8],
                  "embedding": [[1.0, 2.0], [3.0, 4.0]]})
    pq.write_table(t, tmp_path / "t.parquet")
    assert load_vectors(tmp_path / "t.parquet").shape == (2, 2)


def test_library_sample_zero_means_all_rows_and_bad_args_raise():
    import pytest

    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 8)).astype(np.float32)
    assert not audit(x, k=5, sample=0).sampled
    for bad in ({"k": 0}, {"k": 2.5}, {"sample": 3}, {"clusters": -1}, {"near_dup_cos": 0},
                {"thresholds": [("hubness.skewness", (1, 2))]}, {"thresholds": {"hubness.skewness": {"warn": 1}}}):
        with pytest.raises((ValueError, TypeError)):
            audit(x, **{"k": 5, **bad})
