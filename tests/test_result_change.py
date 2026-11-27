"""Label-free result change against a baseline: a fixed sample of queries and rows keeps its
exact top-k, and the diff says how much of it changed."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vecscope import audit
from vecscope.cli import main
from vecscope.evidence import diff_evidence
from vecscope.results import pick_rows


def _data(seed=0, n=1500, d=32):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((30, d))
    x = (c[rng.integers(0, 30, n)] + 0.5 * rng.standard_normal((n, d))).astype(np.float32)
    q = (c[rng.integers(0, 30, 200)] + 0.5 * rng.standard_normal((200, d))).astype(np.float32)
    return x, q


def test_pick_rows_keeps_the_old_range_under_appended_rows():
    old, new = pick_rows(1000, 100), pick_rows(1200, 100)
    assert len(old) == len(new) == 100
    assert set(new[new < 1000]) <= set(old)
    assert len(set(new) & set(old)) >= 75
    assert np.array_equal(pick_rows(1000, 100), old)
    assert len(pick_rows(50, 100)) == 50


def test_identical_runs_agree_and_a_lost_id_mapping_does_not():
    """10% of rows point at another document's vector: no finding moves, the results do."""
    x, q = _data()
    base = audit(x, k=10, queries=q, result_sample=200).to_dict()
    same = audit(x, k=10, queries=q, result_sample=200).to_dict()
    d = diff_evidence(same, base, result_tolerance=0.0)
    assert d["result_agreement"]["queries"]["changed_share"] == 0.0
    assert d["result_agreement"]["rows"]["changed_share"] == 0.0
    assert d["result_regressions"] == []

    rng = np.random.default_rng(1)
    rows = rng.choice(len(x), len(x) // 10, replace=False)
    moved = x.copy()
    moved[rows] = x[np.roll(rows, 1)]
    new = audit(moved, k=10, queries=q, result_sample=200).to_dict()
    d = diff_evidence(new, base, result_tolerance=0.0)
    assert d["regressions"] == []
    assert d["result_agreement"]["queries"]["changed_share"] > 0.03
    assert set(d["result_regressions"]) == {"queries", "rows"}
    # without a tolerance the change is reported, never a regression
    d = diff_evidence(new, base)
    assert d["result_regressions"] == [] and d["result_agreement"]
    # and without labels the change decision cites it
    reasons = " ".join(d["change_decision"]["reasons"])
    assert d["change_decision"]["decision"] == "needs_review"
    assert "changed" in reasons


def test_unchecked_when_a_run_has_no_sample():
    x, q = _data()
    base = audit(x, k=10, queries=q).to_dict()
    new = audit(x, k=10, queries=q, result_sample=100).to_dict()
    d = diff_evidence(new, base, result_tolerance=0.0)
    assert d["result_agreement"] == {}
    assert d["result_unchecked"] == ["results"]
    assert diff_evidence(new, base)["result_unchecked"] == []


def test_result_sample_is_deterministic_and_off_by_default():
    x, q = _data()
    assert audit(x, k=10).result_sample == {}
    a = audit(x, k=10, queries=q, result_sample=50).result_sample
    assert a == audit(x, k=10, queries=q, result_sample=50).result_sample
    assert len(a["rows"]) == 50 and len(a["query_results"][0]) == 10
    assert all(r not in nb for r, nb in zip(a["rows"], a["row_neighbors"]))
    with pytest.raises(ValueError, match="result_sample"):
        audit(x, k=10, result_sample=-1)


def test_cli_result_tolerance_gate(tmp_path, capsys):
    x, q = _data()
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "q.npy", q)
    moved = x.copy()
    moved[[0, 1, 2, 3]] = x[[3, 0, 1, 2]]
    moved[10:160] = x[np.roll(np.arange(10, 160), 1)]
    np.save(tmp_path / "moved.npy", moved)
    common = ["--queries", str(tmp_path / "q.npy"), "--result-sample", "200", "--fail-on", "none"]
    base = tmp_path / "base.json"
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["audit", str(tmp_path / "x.npy"), "--json", *common]) == 0
    base.write_text(buf.getvalue())
    assert json.loads(base.read_text())["result_sample"]["k"] == 10
    gate = ["--baseline", str(base), "--result-tolerance", "0"]
    assert main(["audit", str(tmp_path / "x.npy"), *common, *gate]) == 0
    assert main(["audit", str(tmp_path / "moved.npy"), *common, *gate]) == 1
    capsys.readouterr()
    x_path = str(tmp_path / "x.npy")
    assert main(["audit", x_path, "--baseline", str(base), "--result-tolerance", "0"]) == 2
    assert "needs --result-sample" in capsys.readouterr().err
    assert main(["audit", x_path, "--result-sample", "10", "--result-tolerance", "0"]) == 2
    assert "need --baseline" in capsys.readouterr().err


def test_mcp_keeps_the_sample_for_the_baseline_but_not_in_the_response(tmp_path, monkeypatch):
    monkeypatch.setenv("VECSCOPE_ALLOWED_DIRS", str(tmp_path))
    pytest.importorskip("mcp")
    from vecscope.mcp_server import audit_embeddings

    x, q = _data()
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "q.npy", q)
    moved = x.copy()
    moved[10:160] = x[np.roll(np.arange(10, 160), 1)]
    np.save(tmp_path / "moved.npy", moved)
    qp = str(tmp_path / "q.npy")
    base = audit_embeddings(str(tmp_path / "x.npy"), queries_path=qp, result_sample=100)
    assert base["result_sample"] == {
        "kept_for_baseline": True,
        "k": 10,
        "rows": 100,
        "queries": 100,
    }
    res = audit_embeddings(
        str(tmp_path / "moved.npy"),
        queries_path=qp,
        result_sample=100,
        baseline=base["audit_id"],
        result_tolerance=0.0,
    )
    assert set(res["baseline_diff"]["result_regressions"]) == {"queries", "rows"}
    assert "RESULTS CHANGED" in res["summary"]
    ref = audit_embeddings(str(tmp_path / "x.npy"), reference_path=str(tmp_path / "x.npy"))
    assert ref["reference"]["dim"] == 32
