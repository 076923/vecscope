"""query_gap and query_shift: the ood_queries replacement from the redesign run on
19 models x 3 datasets (docs/verification.md), and the tighter ANN recall tie slack."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vecscope import audit
from vecscope.cli import main
from vecscope.diagnostics.ann_recall import tie_tolerance
from vecscope.evidence import THRESHOLDS, query_reference


def _unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope="module")
def space():
    rng = np.random.default_rng(11)
    centers = rng.standard_normal((30, 48))
    corpus = _unit(centers[rng.integers(0, 30, 2500)] + rng.standard_normal((2500, 48)) * 0.6)
    queries = _unit(centers[rng.integers(0, 30, 200)] + rng.standard_normal((200, 48)) * 0.6)
    rot, _ = np.linalg.qr(rng.standard_normal((48, 48)))
    return corpus, queries, _unit(queries @ rot.astype(np.float32))


def _by(ev):
    return {f.name: f for f in ev.findings}


def test_query_gap_is_the_ratio_of_medians(space):
    corpus, queries, _ = space
    f = _by(audit(corpus, k=10, queries=queries))["query_gap"]
    d = f.details
    assert f.value == pytest.approx(d["median_query_nn_distance"] / d["median_corpus_nn_distance"])
    assert (f.warn_at, f.fail_at) == THRESHOLDS["queries.query_gap_ratio"] == (2.6, 3.0)
    assert d["query_nn_mad"] >= 0 and f.verdict == "ok"


def test_query_gap_without_distinct_corpus_neighbors_is_zero():
    rng = np.random.default_rng(0)
    base = _unit(rng.standard_normal((5, 16)))
    corpus = np.repeat(base, 20, axis=0)  # every row has an exact copy
    f = _by(audit(corpus, k=5, queries=_unit(rng.standard_normal((30, 16)))))["query_gap"]
    assert f.value == 0.0 and f.details["corpus_has_distinct_neighbors"] is False


def test_query_shift_needs_a_baseline_and_compares_against_it(space):
    corpus, queries, other = space
    base = audit(corpus, k=10, queries=queries).to_dict()
    assert "query_shift" not in {f["name"] for f in base["findings"]}
    same = _by(audit(corpus, k=10, queries=queries, query_baseline=base))["query_shift"]
    moved = _by(audit(corpus, k=10, queries=other, query_baseline=base))["query_shift"]
    med, mad = query_reference(base)
    assert same.details["shift_cutoff_distance"] == pytest.approx(med + 3 * mad)
    assert same.verdict == "ok" and same.value < 0.1
    assert moved.verdict == "fail" and moved.value > 0.9
    assert moved.details["shifted_sample_ids"]


def test_query_shift_round_trips_through_json(space):
    corpus, queries, other = space
    base = json.loads(audit(corpus, k=10, queries=queries).to_json())
    assert _by(audit(corpus, k=10, queries=other, query_baseline=base))["query_shift"].value > 0.9


@pytest.mark.parametrize("bad, err, match", [
    ([], TypeError, "findings list"),
    ({"findings": []}, ValueError, "no query_gap finding"),
    ({"findings": [{"name": "query_gap", "details": {"median_query_nn_distance": 0.2}}]},
     ValueError, "no query_gap finding"),
    ({"findings": [{"name": "query_gap", "details": {"median_query_nn_distance": 0.2,
                                                     "query_nn_mad": float("nan")}}]},
     ValueError, "no query_gap finding"),
])
def test_query_baseline_validation(space, bad, err, match):
    corpus, queries, _ = space
    with pytest.raises(err, match=match):
        audit(corpus, k=10, queries=queries, query_baseline=bad)


def test_query_baseline_needs_queries_and_the_same_dim(space):
    corpus, queries, _ = space
    base = audit(corpus, k=10, queries=queries).to_dict()
    with pytest.raises(ValueError, match="needs queries"):
        audit(corpus, k=10, query_baseline=base)
    with pytest.raises(ValueError, match="dim=48"):
        audit(corpus[:, :40], k=10, queries=queries[:, :40], query_baseline=base)


def test_cli_baseline_with_queries_adds_query_shift(tmp_path, capsys, space):
    corpus, queries, other = space
    np.save(tmp_path / "c.npy", corpus)
    np.save(tmp_path / "q.npy", queries)
    np.save(tmp_path / "o.npy", other)
    main(["audit", str(tmp_path / "c.npy"), "--queries", str(tmp_path / "q.npy"), "--json"])
    (tmp_path / "base.json").write_text(capsys.readouterr().out)
    code = main(["audit", str(tmp_path / "c.npy"), "--queries", str(tmp_path / "o.npy"),
                 "--baseline", str(tmp_path / "base.json"), "--json"])
    out = json.loads(capsys.readouterr().out)
    names = {f["name"]: f for f in out["findings"]}
    assert code == 1 and names["query_shift"]["verdict"] == "fail"
    assert "query_shift" in out["baseline_diff"]["only_in_new"]
    assert "query_gap" in out["baseline_diff"]["regressions"]
    # a baseline without query distances still diffs; query_shift is skipped
    main(["audit", str(tmp_path / "c.npy"), "--json"])
    (tmp_path / "noq.json").write_text(capsys.readouterr().out)
    main(["audit", str(tmp_path / "c.npy"), "--queries", str(tmp_path / "q.npy"),
          "--baseline", str(tmp_path / "noq.json"), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert "query_shift" not in {f["name"] for f in out["findings"]}


def test_mcp_query_baseline_path(tmp_path, space):
    pytest.importorskip("mcp")
    from vecscope.mcp_server import audit_embeddings

    corpus, queries, other = space
    np.save(tmp_path / "c.npy", corpus)
    np.save(tmp_path / "q.npy", queries)
    np.save(tmp_path / "o.npy", other)
    base = audit_embeddings(str(tmp_path / "c.npy"), queries_path=str(tmp_path / "q.npy"))
    (tmp_path / "base.json").write_text(json.dumps(base))
    res = audit_embeddings(str(tmp_path / "c.npy"), queries_path=str(tmp_path / "o.npy"),
                           query_baseline_path=str(tmp_path / "base.json"))
    assert "query_shift=" in res["summary"] and "query_gap=" in res["summary"]


def test_tie_tolerance_covers_float32_rounding_but_not_the_gaps_060_credited():
    # 768-d cosine distances in float32 were off float64 by at most 5.9e-7 (bench run); a
    # 1e-5 relative slack credited 11th neighbors 2.10e-6 and 2.68e-6 farther than the 10th at d 0.388 and 0.356
    assert 5.9e-7 < tie_tolerance(0.0) <= tie_tolerance(0.37)
    assert tie_tolerance(0.38757522) < 2.10e-6 and tie_tolerance(0.35610206) < 2.68e-6
    assert tie_tolerance(1.0) == pytest.approx(3e-6)
