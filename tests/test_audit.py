import json

import numpy as np
import pytest

from vecscope import audit, compare
from vecscope.diagnostics import ann_recall


def test_isotropic_gaussian_has_no_signal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 64)).astype(np.float32)
    ev = audit(x, k=10)
    by = {f.name: f for f in ev.findings}
    assert by["anisotropy"].verdict == "ok"
    assert by["collapse"].verdict == "ok"
    # pure noise has no neighbor structure: the contrast equals the noise baseline
    assert abs(by["distance_concentration"].value - 1.0) < 0.1
    assert by["distance_concentration"].verdict == "fail"
    assert all(f.verdict != "fail" for f in ev.findings if f.name != "distance_concentration")


def test_contrast_over_noise_is_size_free():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((12000, 128)).astype(np.float32)
    f = {f.name: f for f in audit(x, k=5, sample=None).findings}["distance_concentration"]
    assert abs(f.value - 1.0) < 0.05  # the raw contrast of noise rises with n (1.44 -> 1.55)
    assert f.details["baseline_rows"] == 4000 and f.verdict == "fail"
    with pytest.raises(ValueError, match="renamed"):
        audit(x[:500], k=5, thresholds={"distance_concentration.relative_contrast": [1.5, 1.3]})


def test_collapse_jittered_boilerplate_is_one_group():
    rng = np.random.default_rng(0)
    c = rng.standard_normal((40, 32))
    base = c[rng.integers(0, 40, 1500)] + rng.standard_normal((1500, 32)) * 0.8
    boiler = np.ones((300, 32)) + rng.standard_normal((300, 32)) * 0.01  # cos ~0.9999, not exact
    x = np.concatenate([base, boiler]).astype(np.float32)
    d = {f.name: f for f in audit(x, k=10).findings}["collapse"].details
    assert d["largest_exact_duplicate_groups"] == []
    assert d["largest_near_duplicate_groups"][0] >= 290  # linked through k neighbors, not 1-NN
    assert d["sample_pairs"][0][0] >= 1500 and d["sample_pairs"][0][2] <= 1.0


def test_contrast_over_noise_is_dimension_free():
    rng = np.random.default_rng(1)
    vals = []
    for d in (32, 256, 1024):
        x = rng.standard_normal((1500, d)).astype(np.float32)
        vals.append({f.name: f for f in audit(x, k=5).findings}["distance_concentration"].value)
    assert max(vals) - min(vals) < 0.1  # raw relative contrast would span 2.4 to 1.1


def test_shared_offset_triggers_anisotropy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 64)).astype(np.float32) + 5.0
    ev = audit(x, k=10)
    by = {f.name: f for f in ev.findings}
    assert by["anisotropy"].verdict == "fail"


def test_collapse_reports_group_structure():
    rng = np.random.default_rng(0)
    c = rng.standard_normal((40, 16))
    base = c[rng.integers(0, 40, 600)] + rng.standard_normal((600, 16)) * 0.8
    x = np.concatenate([base, base[:100], np.tile(base[:1], (30, 1))]).astype(np.float32)
    d = {f.name: f for f in audit(x, k=5).findings}["collapse"].details
    assert d["largest_exact_duplicate_groups"][0] == 32  # base[0], its copy and 30 tiles
    assert d["largest_near_duplicate_groups"][0] == 32
    assert d["near_duplicate_groups"] >= 100
    assert d["largest_near_duplicate_groups"][1] == 2  # the other copies pair up
    assert d["sample_pairs"] and all(i != j and 0.99 <= cos <= 1.0 for i, j, cos in d["sample_pairs"])
    assert len({tuple(p[:2]) for p in d["sample_pairs"]}) == len(d["sample_pairs"])
    big = {0, 600} | set(range(700, 730))  # rows of the 32-row group
    assert sum(p[0] in big or p[1] in big for p in d["sample_pairs"]) == 1  # one pair per group
    clean = {f.name: f for f in audit(base.astype(np.float32), k=5).findings}["collapse"]
    assert clean.details["largest_near_duplicate_groups"] == []
    assert "no row has a near-duplicate" in clean.explanation


def test_compare_two_spaces():
    from vecscope.compare import render_compare

    rng = np.random.default_rng(0)
    c = rng.standard_normal((30, 16))
    a = (c[rng.integers(0, 30, 500)] + rng.standard_normal((500, 16)) * 0.5).astype(np.float32)
    a = np.concatenate([a, a[:50]])  # 50 duplicate items
    same = compare(a, a, k=5)
    assert same.nn1_agreement == 1.0 and same.topk_overlap == 1.0
    assert same.near_duplicate_both == same.near_duplicate_ratio_a > 0
    assert same.same_nearest_among_both_flagged == 1.0
    assert same.near_duplicate_only_a == 0.0
    # a rotation keeps every neighbor; an orthogonal transform of A is a different "model"
    q, _ = np.linalg.qr(rng.standard_normal((16, 16)))
    rot = compare(a, a @ q.astype(np.float32), k=5)
    assert rot.topk_overlap > 0.95
    noise = compare(a, rng.standard_normal(a.shape).astype(np.float32), k=5)
    assert noise.topk_overlap < 0.1 and noise.near_duplicate_only_a > 0
    assert noise.same_nearest_among_both_flagged is None
    assert len(noise.sample_disagreements) == 10
    assert "nn1_agreement" in render_compare(noise) and '"topk_overlap"' in json.dumps(noise.to_dict())
    sampled = compare(a, a, k=5, sample=200)
    assert sampled.sampled and sampled.n == 200 and sampled.n_total == 550
    with pytest.raises(ValueError):
        compare(a, a[:-1])
    with pytest.raises(ValueError):
        compare(a, a, k=0)
    with pytest.raises(ValueError, match="exceed k"):
        compare(a, a, k=5, sample=1)
    assert compare(a, a, k=5, sample=0).sampled is False  # 0 = every row, as in audit
    with pytest.raises(ValueError, match="non-negative"):
        compare(a, a, k=5, sample=-1)
    with pytest.raises(ValueError, match="NaN"):
        compare(a, np.full_like(a, np.nan))


def test_ann_neighbors_without_columns_is_rejected():
    from vecscope.diagnostics import ann_recall

    x = np.random.default_rng(0).standard_normal((50, 8)).astype(np.float32)
    with pytest.raises(ValueError, match="neighbor column"):
        audit(x, k=5, sample=None, ann_neighbors=np.zeros((50, 0), dtype=np.int64))
    with pytest.raises(ValueError, match="neighbor column"):
        ann_recall(np.zeros((10, 3), np.int64), np.zeros((10, 0), np.int64))
    # a row whose index returned nothing (all -1) simply scores 0, it is not an error
    ev = audit(x, k=5, sample=None, ann_neighbors=np.full((50, 5), -1))
    assert {f.name: f for f in ev.findings}["ann_recall"].value == 0.0


def test_diff_evidence_and_baseline_cli(tmp_path, capsys):
    from vecscope.cli import main
    from vecscope.evidence import SCHEMA_VERSION, diff_evidence

    rng = np.random.default_rng(0)
    c = rng.standard_normal((30, 16))
    good = (c[rng.integers(0, 30, 800)] + rng.standard_normal((800, 16)) * 0.6).astype(np.float32)
    bad = np.concatenate([good, good[:200]])  # duplicates: collapse gets worse
    old, new = audit(good, k=5).to_dict(), audit(bad, k=5).to_dict()
    assert old["schema_version"] == SCHEMA_VERSION
    d = diff_evidence(new, old)
    assert "collapse" in d["regressions"] and not d["improvements"]
    assert d["comparable"] is False  # n differs
    assert diff_evidence(old, old)["comparable"] is True
    loose = audit(good, k=5, thresholds={"collapse.near_duplicate_ratio": (0.5, 0.9)}).to_dict()
    assert diff_evidence(loose, old)["comparable"] is False  # thresholds differ
    by = {c["name"]: c for c in d["changes"]}
    assert by["collapse"]["delta"] == pytest.approx(new["findings"][4]["value"] - old["findings"][4]["value"])
    assert diff_evidence(old, old)["regressions"] == []
    with pytest.raises(ValueError, match="schema_version"):
        diff_evidence(new, {**old, "schema_version": 99})
    with pytest.raises(ValueError, match="findings"):
        diff_evidence(new, {"verdict": "ok"})

    base = tmp_path / "base.json"
    base.write_text(json.dumps(old), encoding="utf-8")
    np.save(tmp_path / "bad.npy", bad)
    np.save(tmp_path / "good.npy", good)
    assert main(["audit", str(tmp_path / "bad.npy"), "-k", "5", "--baseline", str(base)]) == 1
    out = capsys.readouterr().out
    assert "WORSE" in out and "regressions: collapse" in out
    # same data as the baseline: no regression, exit code follows --fail-on alone
    code = main(["audit", str(tmp_path / "good.npy"), "-k", "5", "--baseline", str(base), "--json"])
    # no regression, yet exit 1: the baseline itself is FAIL and --fail-on fail (the default)
    # still trips, as the README states; regressions alone are not what decides the code
    assert old["verdict"] == "fail" and code == 1
    assert json.loads(capsys.readouterr().out)["baseline_diff"]["regressions"] == []


def test_golden_evidence_for_shipped_examples():
    """The committed JSON pins values and verdicts for the example files; a change here is a
    change in what users see and must be deliberate (regenerate with examples/make_examples.py)."""
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("make_examples", root / "examples" / "make_examples.py")
    make_examples = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_examples)
    for name, x in make_examples.make_examples().items():  # the .npy files are not in git
        ev = audit(x, k=10).to_dict()
        gold = json.loads((root / "tests" / "golden" / f"{name}.json").read_text(encoding="utf-8"))
        assert ev["schema_version"] == gold["schema_version"]
        assert [f["name"] for f in ev["findings"]] == [f["name"] for f in gold["findings"]]
        assert set(ev) == set(gold)
        for f, g in zip(ev["findings"], gold["findings"]):
            assert f["verdict"] == g["verdict"], (name, f["name"])
            # a different BLAS can flip a near-tied neighbor, which moves the count-based
            # findings slightly; verdicts and the other values are pinned tightly
            loose = f["name"] in {"hubness", "antihubs", "outliers", "connectivity"}
            # distance_concentration depends on a median of float32 distances, which a
            # different BLAS accumulation order moves by a few 1e-6 relative
            tol = (5e-2, 1e-3) if loose else (1e-4, 1e-8) if f["name"] == "distance_concentration" else (1e-5, 1e-8)
            assert f["value"] == pytest.approx(g["value"], rel=tol[0], abs=tol[1]), (name, f["name"])
            assert set(f["details"]) == set(g["details"]), (name, f["name"])


def test_neighbor_ids_written_as_floats_and_bad_baseline_file(tmp_path, capsys):
    from vecscope.cli import main
    from vecscope.io import load_neighbors

    (tmp_path / "nb.csv").write_text("1.0,2.0\n0.0,2.0\n", encoding="utf-8")
    assert load_neighbors(tmp_path / "nb.csv").tolist() == [[1, 2], [0, 2]]
    (tmp_path / "frac.csv").write_text("1.5,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integers"):
        load_neighbors(tmp_path / "frac.csv")
    with pytest.raises(FileNotFoundError):
        load_neighbors(tmp_path / "missing.csv")
    np.save(tmp_path / "x.npy", np.random.default_rng(0).standard_normal((50, 8)).astype(np.float32))
    assert main(["audit", str(tmp_path / "x.npy"), "-k", "5", "--baseline", str(tmp_path / "x.npy")]) == 2
    assert "--baseline" in capsys.readouterr().err


def test_sampled_audit_reports_input_row_ids():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3000, 16)).astype(np.float32)
    x[2500:2510] = x[2500]  # a duplicate group that lives far beyond the sample size
    ev = audit(x, k=5, sample=1000, clusters=2)
    assert ev.sampled
    pairs = next(f for f in ev.findings if f.name == "collapse").details["sample_pairs"]
    assert pairs and all(2500 <= a <= 2509 and 2500 <= b <= 2509 for a, b, _ in pairs)
    assert max(i for c in ev.clusters for i in c["sample_ids"]) >= 1000
    y = x.copy()
    y[2600] = rng.standard_normal(16)
    r = compare(x, y, k=5, sample=1000)
    assert max(d["nearest_a"] for d in r.sample_disagreements) >= 1000


def test_cli_reports_missing_gpu_as_user_error(tmp_path, capsys, monkeypatch):
    pytest.importorskip("faiss")
    import vecscope.sweep as sw
    from vecscope.cli import main

    monkeypatch.setattr(sw, "gpu_available", lambda faiss: False)
    np.save(tmp_path / "x.npy", np.random.default_rng(0).standard_normal((100, 8)).astype(np.float32))
    assert main(["sweep", str(tmp_path / "x.npy"), "--device", "gpu"]) == 2
    assert "no GPU" in capsys.readouterr().err


def test_loader_rejects_empty_text_and_non_integer_ids_cleanly(tmp_path):
    import warnings

    from vecscope.io import load_neighbors, load_vectors

    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    (tmp_path / "blank.txt").write_text("\n  \n", encoding="utf-8")
    (tmp_path / "null.json").write_text("[[1, 2], [null, 2]]", encoding="utf-8")
    (tmp_path / "inf.csv").write_text("1,inf\n", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a numpy warning here is a failure
        for name in ("empty.csv", "blank.txt"):
            with pytest.raises(ValueError, match="empty"):
                load_vectors(tmp_path / name)
            with pytest.raises(ValueError, match="empty"):
                load_neighbors(tmp_path / name)
        (tmp_path / "huge.csv").write_text("1e30,1\n", encoding="utf-8")
        (tmp_path / "nested.json").write_text("[[[1, 2]], [[3, 4]]]", encoding="utf-8")
        (tmp_path / "objects.json").write_text('[[1, {"a": 2}]]', encoding="utf-8")
        for name in ("null.json", "inf.csv", "huge.csv"):
            with pytest.raises(ValueError, match="integers|numbers only"):
                load_neighbors(tmp_path / name)
        for name in ("nested.json", "objects.json"):
            with pytest.raises(ValueError, match="flat lists|numbers only"):
                load_vectors(tmp_path / name)


def test_equal_warn_and_fail_override_keeps_the_default_direction():
    from vecscope.evidence import _verdict

    th = {"ann_recall.recall_at_k": (0.9, 0.9), "hubness.skewness": (3.0, 3.0)}
    assert _verdict("ann_recall.recall_at_k", 0.99, th) == "ok"  # lower is worse: 0.99 is fine
    assert _verdict("ann_recall.recall_at_k", 0.5, th) == "fail"
    assert _verdict("hubness.skewness", 2.0, th) == "ok"
    assert _verdict("hubness.skewness", 3.0, th) == "fail"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 16)).astype(np.float32)
    f = next(f for f in audit(x, k=5, thresholds={"metric_mismatch.topk_overlap": (0.8, 0.8)}).findings
             if f.name == "metric_mismatch")
    assert f.lower_is_worse and f.healthy_range.startswith("ok >=")


def test_ann_file_with_self_in_k_columns_is_explained(tmp_path):
    from vecscope._knn import exact_knn
    from vecscope.evidence import diff_evidence
    from vecscope.io import load_neighbors

    x = np.random.default_rng(0).standard_normal((200, 8)).astype(np.float32)
    idx, _ = exact_knn(x, 5)
    ann = np.concatenate([np.arange(200)[:, None], idx[:, :4]], axis=1)  # self + 4 true ids
    f = audit(x, k=5, sample=None, ann_neighbors=ann).findings[-1]
    assert f.value == pytest.approx(0.8)
    assert f.details["rows_with_fewer_than_k_ids_ratio"] == 1.0 and "k + 1" in f.explanation
    (tmp_path / "ids.jsonl").write_text('{"ids": [1.5, 2]}\n{"ids": [0, 2]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="integers"):
        load_neighbors(tmp_path / "ids.jsonl")
    nameless = {"schema_version": 1, "findings": [{"verdict": "ok", "value": 1.0}]}
    with pytest.raises(ValueError, match="malformed"):
        diff_evidence(nameless, nameless)


def test_all_rows_duplicated_is_reported_as_collapse_not_outliers():
    x = np.repeat(np.random.default_rng(0).standard_normal((300, 16)).astype(np.float32), 4, axis=0)
    by = {f.name: f for f in audit(x, k=5).findings}
    assert by["collapse"].verdict == "fail"
    assert by["outliers"].value == 0.0 and by["outliers"].details["distinct_neighbors"] is False
    assert by["distance_concentration"].value == 0.0
    assert by["distance_concentration"].details["distinct_neighbors"] is False


def test_fully_duplicated_corpus_does_not_flag_every_query_as_ood():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((100, 16)).astype(np.float32)
    x = np.repeat(base, 3, axis=0)
    q = base + rng.standard_normal(base.shape).astype(np.float32) * 0.05
    f = next(f for f in audit(x, k=5, queries=q).findings if f.name == "ood_queries")
    assert f.value == 0.0 and f.details["corpus_has_distinct_neighbors"] is False
    with pytest.raises(ValueError, match="no rows"):
        audit(x, k=5, queries=np.zeros((0, 16), np.float32))
    with pytest.raises(ValueError, match="pair"):
        audit(x, k=5, thresholds={"hubness.skewness": "35"})


def test_unreadable_npy_and_npz_are_user_errors(tmp_path, capsys):
    from vecscope.cli import main

    (tmp_path / "empty.npy").write_bytes(b"")
    np.savez(tmp_path / "t.npz", x=np.ones((5, 3), np.float32))
    data = (tmp_path / "t.npz").read_bytes()
    (tmp_path / "t.npz").write_bytes(data[:-30])
    for name in ("empty.npy", "t.npz"):
        assert main(["audit", str(tmp_path / name)]) == 2
        assert "not a readable" in capsys.readouterr().err


def test_cluster_near_dup_rule_matches_collapse_and_float32_options_serialize():
    import warnings

    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 16)).astype(np.float32)
    x[:40] = 0.0
    ev = audit(x, k=5, clusters=2, near_dup_cos=np.float32(0.99))
    json.loads(ev.to_json())  # numpy scalars must not leak into the evidence
    collapse_value = next(f.value for f in ev.findings if f.name == "collapse")
    weighted = sum(c["near_duplicate_ratio"] * c["size"] for c in ev.clusters) / 400
    assert weighted == pytest.approx(collapse_value)
    big = rng.standard_normal((50, 4)) * 1e300  # float64, finite, beyond float32
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="Rescale"):
            audit(big, k=3)


def test_short_distinct_rows_are_not_exact_copies_and_loader_messages(tmp_path):
    from vecscope.io import load_neighbors, load_vectors

    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 16)).astype(np.float32)
    x[:40] *= 1e-7  # tiny but distinct directions: not copies of each other
    c = next(f for f in audit(x, k=5).findings if f.name == "collapse")
    assert c.details["exact_duplicate_ratio"] == 0.0 and c.details["largest_exact_duplicate_groups"] == []
    (tmp_path / "scalars.jsonl").write_text("1\n2\n3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="list of numbers"):
        load_vectors(tmp_path / "scalars.jsonl")
    (tmp_path / "empty.json").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty.json"):
        load_vectors(tmp_path / "empty.json")
    np.save(tmp_path / "bool.npy", np.ones((5, 3), bool))
    with pytest.raises(ValueError, match="boolean"):
        load_neighbors(tmp_path / "bool.npy")


def test_tie_aware_recall_still_counts_a_missed_unique_neighbor():
    from vecscope._knn import exact_knn
    from vecscope.diagnostics.ann_recall import tie_aware_hits

    assert tie_aware_hits([0.1, 0.2, 0.2, 0.2, 0.3], [0.2, 0.2, 0.2, 0.3, 0.9]) == 4  # 0.1 missed
    # slots 1e-5 apart are distinct neighbors (tolerance ~1.2e-6 at this distance): five
    # copies of 0.10004 fill only its own slot
    assert tie_aware_hits([0.1, 0.10001, 0.10002, 0.10003, 0.10004], [0.10004] * 5) == 1
    # slots 1e-6 apart are within float32 rounding and do tie: the copies fill the slots
    # within tolerance of 0.100004 (itself and 0.100003), never the farther ones
    assert tie_aware_hits([0.1, 0.100001, 0.100002, 0.100003, 0.100004], [0.100004] * 5) == 2
    # an 11th neighbor 1e-5 farther than the 10th is a miss, not a tie (recall inflation seen
    # with the earlier 1e-4 relative slack: docs/verification.md)
    exact = np.array([0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.705, 0.708, 0.709, 0.7100])
    assert tie_aware_hits(exact, np.r_[exact[:9], 0.71001]) == 9
    # nor is one 2.1e-6 farther at distance 0.39, the gap that a 1e-5 relative slack
    # credited on 100k x 768 clustered vectors (IVF nprobe=1)
    exact = np.array([0.30, 0.32, 0.34, 0.35, 0.36, 0.37, 0.38, 0.385, 0.387, 0.38757522])
    assert tie_aware_hits(exact, np.r_[exact[:9], 0.38757733]) == 9
    assert tie_aware_hits([0.1, 0.2, 0.3, 0.4, 0.5], [0.5] * 5) == 1
    # a hit must not be wasted on a slot another id could still fill (maximum matching)
    assert tie_aware_hits([0.1, 0.100005, 0.100010], [0.100010, 0.100005, 0.1]) == 3
    assert tie_aware_hits([0.1, 0.2, 0.2, 0.2, 0.3], [0.2, 0.2, 0.2, 0.2, 0.2]) == 3  # tie set capped
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 32)).astype(np.float32)
    x[1] = x[0] + rng.standard_normal(32).astype(np.float32) * 0.05  # unique nearest of row 0
    x[2] = x[0] + rng.standard_normal(32).astype(np.float32) * 0.3  # second nearest ...
    x[3:40] = x[2]  # ... copied: a tie set at the k-th distance
    idx, _ = exact_knn(x, 5)
    ann = idx.copy()
    ann[0] = [10, 11, 12, 13, 14]  # members of the tie set only: the nearest (row 1) is missed
    r = audit(x, k=5, sample=None, ann_neighbors=ann).findings[-1]
    assert r.details["min_recall"] == pytest.approx(0.8)
    r = audit(x, k=5, sample=None, ann_neighbors=idx.astype(np.uint64)).findings[-1]
    assert r.value == 1.0  # unsigned ids are accepted


def test_compare_agrees_on_nearest_within_copy_groups():
    from vecscope import compare

    rng = np.random.default_rng(0)
    for seed in range(30):
        r = np.random.default_rng(seed)
        a = r.standard_normal((300, 32)).astype(np.float32)
        b = r.standard_normal((300, 16)).astype(np.float32)
        a[1:6], b[1:6] = a[0], b[0]  # the same copy group in both spaces
        assert compare(a, b, k=5).same_nearest_among_both_flagged == 1.0, seed
    del rng


def test_zero_width_input_is_rejected(tmp_path, capsys):
    from vecscope import compare
    from vecscope.cli import main

    empty = np.zeros((30, 0), np.float32)
    with pytest.raises(ValueError, match="column"):
        audit(empty, k=5)
    with pytest.raises(ValueError, match="column"):
        compare(empty, empty, k=5)
    np.save(tmp_path / "d0.npy", empty)
    assert main(["audit", str(tmp_path / "d0.npy")]) == 2
    assert "vecscope: error" in capsys.readouterr().err


def test_loader_rejects_string_dtype_and_unsigned_ids_beyond_int64(tmp_path):
    from vecscope.io import load_neighbors, load_vectors

    np.save(tmp_path / "s.npy", np.array([["a", "b"], ["c", "d"], ["e", "f"]]))
    with pytest.raises(ValueError, match="real numbers"):
        load_vectors(tmp_path / "s.npy")
    with pytest.raises(ValueError, match="real numbers"):
        audit(np.array([["a", "b"]] * 3), k=1)
    np.save(tmp_path / "u.npy", np.array([[2**63, 1], [0, 1], [1, 0]], np.uint64))
    with pytest.raises(ValueError, match="int64"):
        load_neighbors(tmp_path / "u.npy")


def test_exact_copy_groups_match_float64_reference_and_signed_zero():
    from vecscope._knn import exact_copy_groups

    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 24)).astype(np.float32)
    x[10:20] = x[10]
    x[30] = -0.0
    x[31] = 0.0
    x[40] = x[41] * 1e-7  # same direction, different length: not a copy
    x[42] = x[41] * 3e-7  # nor of row 40, although both are tiny
    x[50] = x[51] * 2.0
    _, inverse, counts = exact_copy_groups(x)
    assert counts[inverse[10]] == 10 and inverse[30] == inverse[31]
    assert inverse[40] != inverse[41] and inverse[50] != inverse[51] and inverse[40] != inverse[42]
    assert (counts[inverse] > 1).sum() == 12


def test_near_dup_cutoff_is_part_of_baseline_comparability(tmp_path, capsys):
    from vecscope import compare
    from vecscope.cli import main
    from vecscope.evidence import diff_evidence

    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 16)).astype(np.float32)
    old = audit(x, k=5).to_dict()
    assert old["near_dup_cos"] == 0.99
    assert diff_evidence(audit(x, k=5, near_dup_cos=0.5).to_dict(), old)["comparable"] is False
    (tmp_path / "b.json").write_text(json.dumps(old), encoding="utf-8")
    np.save(tmp_path / "x.npy", x)
    main(["audit", str(tmp_path / "x.npy"), "-k", "5", "--near-dup", "0.5", "--baseline", str(tmp_path / "b.json")])
    assert "near-dup cutoff differ" in capsys.readouterr().out
    big = rng.standard_normal((50, 4)) * 1e300
    with pytest.raises(ValueError, match="Rescale"):
        compare(big, big, k=3)
    z = np.concatenate([rng.integers(-5, 5, (20, 4)), np.full((3, 4), -2**63, np.int64)])
    with pytest.raises(ValueError, match="Rescale"):
        audit(z, k=3)
    y = np.repeat(rng.standard_normal((50, 1536)).astype(np.float32), 2, axis=0)
    c = next(f for f in audit(y, k=3).findings if f.name == "collapse")
    assert c.details["nn_cosine_p95"] <= 1.0 and c.details["nn_cosine_median"] <= 1.0


def test_sweep_rejects_bad_queries_and_mislabelled_npy(tmp_path):
    pytest.importorskip("faiss")
    from vecscope.compare import compare
    from vecscope.io import load_vectors
    from vecscope.sweep import sweep

    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 8)).astype(np.float32)
    q = x[:20].copy()
    q[3, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sweep(x, k=5, queries=q, timing_repeats=1)
    with pytest.raises(ValueError, match="Rescale"):
        sweep(x, k=5, queries=x[:20].astype(np.float64) * 1e300, timing_repeats=1)
    with pytest.raises(ValueError, match="smaller than the number of rows"):
        compare(x[:10], x[:10], k=10)
    np.savez(tmp_path / "a.npz", x=x)
    (tmp_path / "zip_as.npy").write_bytes((tmp_path / "a.npz").read_bytes())
    np.save(tmp_path / "b.npy", x)
    (tmp_path / "array_as.npz").write_bytes((tmp_path / "b.npy").read_bytes())
    for name in ("zip_as.npy", "array_as.npz"):
        with pytest.raises(ValueError, match="suffix"):
            load_vectors(tmp_path / name)


def test_verify_by_names_the_metric_the_verdict_uses():
    from vecscope.evidence import _FINDING_KEYS, ACTIONS

    for name in ("antihubs", "rogue_dimensions", "hubness", "collapse", "outliers"):
        metric = _FINDING_KEYS[name].split(".")[1]
        assert metric in ACTIONS[name][2], name


def test_distance_concentration_needs_x_above_baseline_rows():
    from vecscope.diagnostics import distance_concentration

    nn = np.random.default_rng(0).uniform(0.1, 0.5, 5000)
    with pytest.raises(ValueError, match="needs x"):
        distance_concentration(nn, 0.0, dim=16)
    assert "contrast_over_noise" in distance_concentration(nn[:4000], 0.0, dim=16)


def test_exact_ids_score_full_recall_on_tight_high_dim_neighbors():
    from vecscope._knn import exact_knn

    rng = np.random.default_rng(0)
    for spread in (0.01, 0.005):
        c = rng.standard_normal((30, 1536))
        x = c[rng.integers(0, 30, 3000)] + rng.standard_normal((3000, 1536)) * spread
        x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
        idx, _ = exact_knn(x, 10)
        r = audit(x, k=10, sample=None, ann_neighbors=idx).findings[-1]
        assert r.value == 1.0 and r.details["min_recall"] == 1.0, spread


def test_csv_with_huge_values_and_json_booleans_are_rejected_clearly(tmp_path, capsys):
    from vecscope.cli import main
    from vecscope.io import load_vectors

    (tmp_path / "huge.csv").write_text("1e300,1\n2,3\n4,5\n", encoding="utf-8")
    assert main(["audit", str(tmp_path / "huge.csv"), "-k", "1"]) == 2
    assert "Rescale" in capsys.readouterr().err
    (tmp_path / "bools.json").write_text("[[true, 1.0], [0, 1], [1, 0]]", encoding="utf-8")
    with pytest.raises(ValueError, match="numbers only"):
        load_vectors(tmp_path / "bools.json")


def test_query_and_json_values_beyond_float32_get_the_rescale_message(tmp_path, capsys):
    import warnings

    from vecscope.cli import main

    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 8)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="Rescale"):
            audit(x, k=5, queries=rng.standard_normal((4, 8)) * 1e300)
        with pytest.raises(ValueError, match="real numbers"):
            audit(x, k=5, queries=np.array([["a"] * 8]))
    (tmp_path / "big.json").write_text("[[1e39,1,2,3,4,5,6,7],[1,2,3,4,5,6,7,8],[2,3,4,5,6,7,8,9]]",
                                       encoding="utf-8")
    assert main(["audit", str(tmp_path / "big.json"), "-k", "1"]) == 2
    assert "Rescale" in capsys.readouterr().err


def test_zero_vectors_count_as_collapse():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 16)).astype(np.float32)
    x[:40] = 0.0  # 10% zero vectors: cosine 0 with everything, including each other
    c = next(f for f in audit(x, k=5).findings if f.name == "collapse")
    assert c.value == pytest.approx(0.1) and c.verdict == "fail"
    assert c.details["largest_near_duplicate_groups"][0] == 40
    row_a, row_b, cos = c.details["sample_pairs"][0]
    assert row_a < 40 and row_b < 40 and cos == 1.0  # the partner is another copy, not a random row
    ev = audit(np.zeros((50, 8)), k=5)
    assert [f.verdict for f in ev.findings if f.name == "collapse"] == ["fail"]
    with pytest.raises(ValueError, match="near_dup_cos"):
        compare(x, x, k=3, near_dup_cos=1.5)


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
    back = json.loads(s)  # strict JSON: no NaN/Infinity tokens
    assert back == ev.to_dict() and back["tool"] == "vecscope" and "verdict" in back


def test_hub_cluster_raises_k_occurrence():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1500, 32)).astype(np.float32)
    plain = {f.name: f for f in audit(x, k=10).findings}["hubness"].details
    center = rng.standard_normal((1, 32)).astype(np.float32)
    x[:100] = center + rng.standard_normal((100, 32)).astype(np.float32) * 0.05
    hub = {f.name: f for f in audit(x, k=10).findings}["hubness"].details
    # a tight cluster makes its members everyone's neighbors inside it: more skew, bigger hubs
    assert hub["max_k_occurrence"] > plain["max_k_occurrence"]
    assert hub["skewness"] > plain["skewness"]


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
    from vecscope.sweep import render_sweep, sweep

    rng = np.random.default_rng(0)
    x = rng.standard_normal((3000, 32)).astype(np.float32)
    res = sweep(x, k=5, n_queries=200)
    assert res.points and res.recommendation["index"] in {"hnsw", "ivf"}
    assert all(0.0 <= p.recall_at_k <= 1.0 for p in res.points)
    r = res.recommendation
    assert abs(r["margin"] - (r["recall_at_k"] - res.target_recall)) < 1e-9
    assert len(res.caveats) == 5 and "corpus itself" in res.caveats[0]
    assert "note:" in render_sweep(res) and "margin=" in render_sweep(res)
    assert res.latency_queries == 200  # capped by n_queries
    for p in res.points:
        assert p.index_bytes > 0 and p.latency_ms_p50 <= p.latency_ms_p95 <= p.latency_ms_p99
    assert r["latency_ms_p95"] is not None and r["index_bytes"] > 0
    assert "p95 ms" in render_sweep(res) and "MiB" in render_sweep(res)
    skipped = sweep(x, k=5, n_queries=50, latency_queries=0)
    assert skipped.latency_queries == 0 and all(p.latency_ms_p95 is None for p in skipped.points)
    assert "skipped" in skipped.caveats[3] and "-" in render_sweep(skipped)
    with pytest.raises(ValueError, match="latency_queries"):
        sweep(x, k=5, n_queries=50, latency_queries=-1)


def test_sweep_recommendation_margin_is_deterministic():
    from vecscope.sweep import SweepPoint, _recommend

    pts = [SweepPoint("hnsw", {"M": 8}, 0.961, 0.1, 900.0),
           SweepPoint("hnsw", {"M": 16}, 0.983, 0.1, 700.0),
           SweepPoint("hnsw", {"M": 32}, 0.990, 0.1, 500.0),
           SweepPoint("ivf", {"nprobe": 1}, 0.90, 0.1, 2000.0)]
    r = _recommend(pts, 10, 0.95, 0.02)
    assert r["params"] == {"M": 8} and r["tight"] is True
    assert abs(r["margin"] - 0.011) < 1e-9
    assert r["safer_alternative"]["params"] == {"M": 16}  # fastest with >= 0.02 headroom
    assert "M=16" in r["reason"] and "larger than this one" in r["reason"]
    r = _recommend(pts, 10, 0.95, 0.01)
    assert r["params"] == {"M": 8} and r["tight"] is False and "safer_alternative" not in r
    r = _recommend(pts, 10, 0.985, 0.02)
    assert r["params"] == {"M": 32} and r["tight"] is True
    assert "no setting in the grid" in r["reason"] and "safer_alternative" not in r
    r = _recommend(pts, 10, 0.995, 0.02)
    assert r["params"] == {"M": 32} and r["tight"] is True and "no configuration" in r["reason"]


def test_sweep_rejects_bad_repeats_and_margin():
    pytest.importorskip("faiss")
    from vecscope.sweep import sweep

    x = np.random.default_rng(0).standard_normal((300, 8)).astype(np.float32)
    with pytest.raises(ValueError):
        sweep(x, k=5, timing_repeats=0)
    with pytest.raises(ValueError):
        sweep(x, k=5, margin=1.0)


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
    assert set(by) >= {"audit_embeddings", "sweep_index", "describe_findings", "compare_embeddings"}
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
    assert set(desc) == {f["name"] for f in full["findings"]} | {
        "ann_recall", "ood_queries", "query_gap", "query_shift", "query_hubness"}


def test_mcp_baseline_by_audit_id_and_summary(tmp_path):
    import json

    pytest = __import__("pytest")
    pytest.importorskip("mcp")
    from vecscope.mcp_server import audit_embeddings

    rng = np.random.default_rng(1)
    c = rng.standard_normal((20, 16))
    x = (c[rng.integers(0, 20, 600)] + rng.standard_normal((600, 16)) * 0.5).astype(np.float32)
    x *= rng.uniform(0.3, 3.0, (600, 1)).astype(np.float32)  # unnormalized
    q = x[:40] + rng.standard_normal((40, 16)).astype(np.float32) * 0.1
    np.save(tmp_path / "a.npy", x)
    np.save(tmp_path / "q.npy", q)
    (tmp_path / "qrels.json").write_text(json.dumps({str(i): {str(i): 1} for i in range(40)}))
    args = {"queries_path": str(tmp_path / "q.npy"), "qrels_path": str(tmp_path / "qrels.json")}
    first = audit_embeddings(str(tmp_path / "a.npy"), k=5, **args)
    assert first["audit_id"] == audit_embeddings(str(tmp_path / "a.npy"), k=5, **args)["audit_id"]
    assert "retrieval nDCG@5=" in first["summary"] and "plan step 1 of" in first["summary"]
    assert first["plan"][0]["group"] in first["summary"]
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    np.save(tmp_path / "b.npy", unit)
    np.save(tmp_path / "qb.npy", q / np.linalg.norm(q, axis=1, keepdims=True))
    second = audit_embeddings(str(tmp_path / "b.npy"), k=5, baseline=first["audit_id"],
                              **{**args, "queries_path": str(tmp_path / "qb.npy")})
    d = second["baseline_diff"]
    assert "metric_mismatch" in d["improvements"] and d["retrieval_comparable"]
    assert "vs baseline: verdict" in second["summary"]
    # normalized rows are another corpus: its query distances are no reference for query_shift
    assert not any(f["name"] == "query_shift" for f in second["findings"])
    assert "query_shift skipped" in second["query_reference_note"]
    assert "query_shift skipped" in second["summary"]
    # the same corpus with new queries takes the baseline as the reference
    same = audit_embeddings(str(tmp_path / "a.npy"), k=5, baseline=first["audit_id"], **args)
    assert any(f["name"] == "query_shift" for f in same["findings"])
    assert "query_reference_note" not in same
    # a saved response works as a baseline file too
    (tmp_path / "first.json").write_text(json.dumps(first))
    again = audit_embeddings(str(tmp_path / "b.npy"), k=5, baseline=str(tmp_path / "first.json"))
    assert again["baseline_diff"]["baseline_verdict"] == first["verdict"]
    with pytest.raises(Exception, match="not an audit_id of this server session"):
        audit_embeddings(str(tmp_path / "b.npy"), baseline="0123456789abcdef")
    with pytest.raises(Exception, match="tolerances need baseline"):
        audit_embeddings(str(tmp_path / "b.npy"), tolerances={"collapse": 0.01})
    (tmp_path / "binary.tsv").write_bytes((tmp_path / "q.npy").read_bytes())
    with pytest.raises(Exception, match="not a text qrels file"):
        audit_embeddings(str(tmp_path / "a.npy"), queries_path=str(tmp_path / "q.npy"),
                         qrels_path=str(tmp_path / "binary.tsv"))
    with pytest.raises(Exception, match="unsupported qrels file"):
        audit_embeddings(str(tmp_path / "a.npy"), queries_path=str(tmp_path / "q.npy"),
                         qrels_path=str(tmp_path / "q.npy"))


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
    # 8 copies per item at k=5: every neighbor set is a tie set, any member must count
    res = sweep(np.repeat(base[:150], 8, axis=0), k=5, n_queries=100, device="cpu", timing_repeats=1)
    assert max(p.recall_at_k for p in res.points) >= 0.99
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
    # ties: 8 copies per item, an exact index that breaks ties differently is still perfect
    y = np.repeat(x[:50], 8, axis=0)
    idx, _ = exact_knn(y, 6)
    other = (idx // 8) * 8 + (idx + 3) % 8  # another member of the same copy group
    r = _by(audit(y, k=5, ann_neighbors=other, sample=None))["ann_recall"]
    assert r.value == 1.0
    wrong = (idx + 8) % len(y)  # a different group: a genuine miss
    r = _by(audit(y, k=5, ann_neighbors=wrong, sample=None))["ann_recall"]
    assert r.value < 0.2


def test_parquet_neighbor_ids_keep_values_above_2_24(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from vecscope.io import load_neighbors

    ids = [[16777217, 16777219], [33554433, 1]]
    pq.write_table(pa.table({"neighbors": ids}), tmp_path / "nb.parquet")
    assert load_neighbors(tmp_path / "nb.parquet").tolist() == ids


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


def test_sample_repeats_reports_ranges_and_stability():
    x = np.random.default_rng(3).standard_normal((2000, 32)).astype(np.float32)
    ev = audit(x, sample=500, sample_repeats=3)
    assert ev.sampled and ev.sample_repeats == 3
    for f in ev.findings:
        assert f.value_range is not None and f.value_range[0] <= f.value <= f.value_range[1]
        assert f.verdict_stable is not None
    assert set(ev.unstable_findings) <= {f.name for f in ev.findings}
    assert ev.to_json() == audit(x, sample=500, sample_repeats=3).to_json()
    ev2 = audit(x, sample_repeats=3)  # nothing sampled: repeats ignored, no ranges
    assert ev2.sample_repeats == 1 and all(f.value_range is None for f in ev2.findings)
    with pytest.raises(ValueError):
        audit(x, sample=500, sample_repeats=0)


def test_value_regressions_with_tolerances(tmp_path, capsys):
    """A finding that drifts in the worse direction inside one verdict band is reported as a
    value regression only when a tolerance is set and exceeded."""
    from vecscope._knn import normalize
    from vecscope.cli import main
    from vecscope.evidence import diff_evidence

    rng = np.random.default_rng(1)
    c = rng.standard_normal((30, 16))
    good = normalize(c[rng.integers(0, 30, 800)] + rng.standard_normal((800, 16)) * 0.6)
    worse = np.concatenate([good[:-4], good[:4]])  # 4 extra copies: collapse rises, same verdict
    old, new = audit(good, k=5).to_dict(), audit(worse, k=5).to_dict()
    by_old = {f["name"]: f for f in old["findings"]}
    by_new = {f["name"]: f for f in new["findings"]}
    assert by_new["collapse"]["value"] > by_old["collapse"]["value"]
    assert by_new["collapse"]["verdict"] == by_old["collapse"]["verdict"]

    d = diff_evidence(new, old)
    row = {c["name"]: c for c in d["changes"]}["collapse"]
    assert row["worsened"] is True and row["worsening"] == pytest.approx(row["delta"])
    assert row["lower_is_worse"] is False and row["tolerance"] is None
    assert d["value_regressions"] == [] and d["tolerances"] == {}  # no tolerance, no verdict change
    assert "collapse" not in d["regressions"]

    d = diff_evidence(new, old, tolerances={"collapse": 0.0})
    assert d["value_regressions"] == ["collapse"] and "collapse" not in d["regressions"]
    assert diff_evidence(new, old, tolerances={"collapse": 1.0})["value_regressions"] == []
    # lower-is-worse findings count a drop as worsening
    lowered = json.loads(json.dumps(old))
    f = {f["name"]: f for f in lowered["findings"]}["effective_rank"]
    f["value"] -= 0.01
    d = diff_evidence(lowered, old, tolerances={"effective_rank": 0.005})
    row = {c["name"]: c for c in d["changes"]}["effective_rank"]
    assert row["lower_is_worse"] is True and row["worsening"] == pytest.approx(0.01)
    assert d["value_regressions"] == ["effective_rank"]
    # a finding that improves is never a value regression, whatever the tolerance
    f["value"] += 0.02
    assert diff_evidence(lowered, old, tolerances={"effective_rank": 0.0})["value_regressions"] == []
    with pytest.raises(ValueError, match="unknown finding"):
        diff_evidence(new, old, tolerances={"collapse.near_duplicate_ratio": 0.1})
    with pytest.raises(ValueError, match="finite number >= 0"):
        diff_evidence(new, old, tolerances={"collapse": -1})
    with pytest.raises(TypeError, match="mapping"):
        diff_evidence(new, old, tolerances=[("collapse", 0.1)])

    base = tmp_path / "base.json"
    base.write_text(json.dumps(old), encoding="utf-8")
    tol = tmp_path / "tol.json"
    tol.write_text(json.dumps({"collapse": 0.0}), encoding="utf-8")
    np.save(tmp_path / "worse.npy", worse)
    assert old["verdict"] != "fail"
    assert main(["audit", str(tmp_path / "worse.npy"), "-k", "5", "--baseline", str(base),
                 "--tolerances", str(tol)]) == 1
    out = capsys.readouterr().out
    assert "VALUE WORSE" in out and "value regressions (beyond tolerance): collapse" in out
    # the same run without tolerances passes: verdicts did not change
    assert main(["audit", str(tmp_path / "worse.npy"), "-k", "5", "--baseline", str(base)]) == 0
    capsys.readouterr()
    assert main(["audit", str(tmp_path / "worse.npy"), "-k", "5", "--tolerances", str(tol)]) == 2
    assert "need --baseline" in capsys.readouterr().err


def test_require_comparable_refuses_a_mismatched_baseline(tmp_path, capsys):
    from vecscope.cli import main

    rng = np.random.default_rng(2)
    x = rng.standard_normal((300, 8)).astype(np.float32)
    np.save(tmp_path / "x.npy", x)
    base = tmp_path / "base.json"
    base.write_text(audit(x, k=5).to_json(), encoding="utf-8")
    assert main(["audit", str(tmp_path / "x.npy"), "-k", "5", "--baseline", str(base),
                 "--require-comparable", "--fail-on", "fail"]) in (0, 1)
    capsys.readouterr()
    # a different k makes the runs incomparable: a plain diff notes it, the strict flag refuses
    main(["audit", str(tmp_path / "x.npy"), "-k", "6", "--baseline", str(base)])
    assert "not directly comparable" in capsys.readouterr().out
    assert main(["audit", str(tmp_path / "x.npy"), "-k", "6", "--baseline", str(base),
                 "--require-comparable"]) == 2
    assert "--require-comparable" in capsys.readouterr().err


def test_mcp_tools_refuse_paths_outside_allowed_dirs(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    import os

    from vecscope import mcp_server

    inside = tmp_path / "data"
    inside.mkdir()
    outside = tmp_path / "secret"
    outside.mkdir()
    x = np.random.default_rng(0).standard_normal((100, 8)).astype(np.float32)
    np.save(inside / "v.npy", x)
    np.save(outside / "v.npy", x)
    monkeypatch.delenv(mcp_server.ALLOWED_DIRS_ENV, raising=False)
    assert mcp_server.allowed_dirs() == []
    assert mcp_server.audit_embeddings(str(outside / "v.npy"), k=5)["n"] == 100  # unrestricted
    monkeypatch.setenv(mcp_server.ALLOWED_DIRS_ENV, os.pathsep.join([str(inside), ""]))
    assert mcp_server.allowed_dirs() == [inside.resolve()]
    assert mcp_server.audit_embeddings(str(inside / "v.npy"), k=5)["n"] == 100
    with pytest.raises(Exception, match="outside the directories"):
        mcp_server.audit_embeddings(str(outside / "v.npy"), k=5)
    with pytest.raises(Exception, match="queries_path"):
        mcp_server.audit_embeddings(str(inside / "v.npy"), k=5, queries_path=str(outside / "v.npy"))
    with pytest.raises(Exception, match="path_b"):
        mcp_server.compare_embeddings(str(inside / "v.npy"), str(outside / "v.npy"), k=5)
    # a ".." path that escapes the allowed directory is resolved before the check
    with pytest.raises(Exception, match="outside"):
        mcp_server.audit_embeddings(str(inside / ".." / "secret" / "v.npy"), k=5)
    with pytest.raises(SystemExit):
        mcp_server.main(["--allow-dir", str(tmp_path / "missing")])


def test_mcp_server_round_trip_over_stdio(tmp_path):
    """Start the server as a subprocess and call tools through the mcp client, so the
    version-specific server class, the tool schema and the error path run end to end."""
    pytest.importorskip("mcp")
    import asyncio
    import os
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from vecscope.mcp_server import ALLOWED_DIRS_ENV

    x = np.random.default_rng(0).standard_normal((120, 8)).astype(np.float32)
    inside = tmp_path / "data"
    inside.mkdir()
    outside = tmp_path / "secret"
    outside.mkdir()
    np.save(inside / "v.npy", x)
    np.save(outside / "v.npy", x)
    from pathlib import Path

    import vecscope

    src = str(Path(vecscope.__file__).resolve().parents[1])  # the checkout's src, or site-packages
    env = {**os.environ, ALLOWED_DIRS_ENV: str(inside),
           "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    params = StdioServerParameters(command=sys.executable, args=["-m", "vecscope.mcp_server"],
                                   env=env)

    async def run():
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            init = await s.initialize()
            tools = {t.name: t for t in (await s.list_tools()).tools}
            ok = await s.call_tool("audit_embeddings", {"path": str(inside / "v.npy"), "k": 5})
            bad = await s.call_tool("audit_embeddings", {"path": str(outside / "v.npy"), "k": 5})
            desc = await s.call_tool("describe_findings", {})
            return init, tools, ok, bad, desc

    def attr(obj, *names):  # mcp 1.x uses camelCase fields, 2.x snake_case
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(names)

    init, tools, ok, bad, desc = asyncio.run(run())
    assert attr(init, "serverInfo", "server_info").name == "vecscope"
    assert {"audit_embeddings", "sweep_index", "compare_embeddings", "describe_findings"} <= set(tools)
    assert not attr(ok, "isError", "is_error") and not attr(desc, "isError", "is_error")
    payload = json.loads(ok.content[0].text)
    assert payload["n"] == 120 and payload["summary"].startswith("verdict=")
    assert attr(bad, "isError", "is_error") and "outside the directories" in bad.content[0].text


def test_public_entry_points_validate_input_alike():
    """audit, compare and sweep refuse the same malformed inputs with the same messages."""
    from vecscope.compare import compare

    good = np.random.default_rng(0).standard_normal((40, 8)).astype(np.float32)
    nan = good.copy()
    nan[3, 2] = np.nan
    huge = good.copy()
    huge[0, 0] = 1e17
    cases = [
        (nan, "NaN or Inf"),
        (huge, "float32 dot products overflow"),
        (good.ravel(), "2-[dD]"),
        (good[:, :0], "column"),
        (good.astype(str), "real numbers"),
    ]
    entry_points = [lambda x: audit(x, k=5), lambda x: compare(x, good, k=5)]
    if pytest.importorskip("faiss", reason="sweep needs faiss") is not None:
        from vecscope.sweep import sweep

        entry_points.append(lambda x: sweep(x, k=5, n_queries=5))
    for x, message in cases:
        for call in entry_points:
            with pytest.raises(ValueError, match=message):
                call(x)
