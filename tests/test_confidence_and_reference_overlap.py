"""Two ways to read a change with fewer false alarms: a labeled gate judged against the noise
of the labeled queries (pass, fail or inconclusive), and the label-free overlap with a
reference model's results (the top hits both models agreed on were lost, or kept)."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import numpy as np
import pytest

from vecscope import audit
from vecscope.cli import main
from vecscope.evidence import diff_evidence


def _data(seed=0, n=1200, d=32, nq=300):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((40, d))
    lab = rng.integers(0, 40, n)
    x = (c[lab] + 0.4 * rng.standard_normal((n, d))).astype(np.float32)
    qlab = rng.integers(0, 40, nq)
    q = (c[qlab] + 0.4 * rng.standard_normal((nq, d))).astype(np.float32)
    # each query's relevant rows: the 3 corpus rows of its cluster nearest to it
    qrels = {}
    for i in range(nq):
        rows = np.flatnonzero(lab == qlab[i])
        best = rows[np.argsort(-(x[rows] @ q[i]))[:3]]
        qrels[i] = {int(r): 1.0 for r in best}
    # a reference model: the same rows through another random rotation
    rot, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return x, q, qrels, (x @ rot).astype(np.float32), (q @ rot).astype(np.float32)


def test_per_query_values_are_kept_and_paired():
    x, q, qrels, *_ = _data()
    r = audit(x, k=10, queries=q, qrels=qrels).retrieval
    pq = r["per_query"]
    assert len(pq["query_rows"]) == len(pq["exact"]["ndcg_at_k"]) == r["judged_queries"]
    assert np.mean(pq["exact"]["ndcg_at_k"]) == pytest.approx(r["exact"]["ndcg_at_k"])
    assert pq["ranked"] is None


def test_confidence_splits_fail_pass_and_inconclusive():
    x, q, qrels, *_ = _data()
    base = audit(x, k=10, queries=q, qrels=qrels).to_dict()
    same = diff_evidence(
        base, base, retrieval_tolerances={"ndcg_at_k": 0.01}, retrieval_confidence=0.95
    )
    c = same["retrieval_confidence_checks"]["exact.ndcg_at_k"]
    assert c["decision"] == "pass" and c["mean_drop"] == 0.0
    assert same["retrieval_regressions"] == [] and same["retrieval_inconclusive"] == []

    # a large fault on every query: a real drop beyond the tolerance
    broken = audit(x[:, :4].copy(), k=10, queries=q[:, :4].copy(), qrels=qrels).to_dict()
    d = diff_evidence(
        broken, base, retrieval_tolerances={"ndcg_at_k": 0.01}, retrieval_confidence=0.95
    )
    assert d["retrieval_regressions"] == ["exact.ndcg_at_k"]
    assert d["change_decision"]["decision"] == "reject"

    # a fault on a handful of queries, judged on few labels: the mean is beyond the
    # tolerance, the interval is not settled
    few = {i: qrels[i] for i in range(20)}
    qq = q.copy()
    qq[:1] = -qq[:1]
    a = audit(x, k=10, queries=q, qrels=few).to_dict()
    b = audit(x, k=10, queries=qq, qrels=few).to_dict()
    plain = diff_evidence(b, a, retrieval_tolerances={"ndcg_at_k": 0.01})
    assert plain["retrieval_regressions"] == ["exact.ndcg_at_k"]
    d = diff_evidence(b, a, retrieval_tolerances={"ndcg_at_k": 0.01}, retrieval_confidence=0.95)
    assert d["retrieval_regressions"] == []
    assert d["retrieval_inconclusive"] == ["exact.ndcg_at_k"]
    c = d["retrieval_confidence_checks"]["exact.ndcg_at_k"]
    assert c["low"] <= 0 < c["mean_drop"] and c["queries_needed"] > 20
    assert d["change_decision"]["decision"] == "needs_review"
    assert "label more queries" in " ".join(d["change_decision"]["reasons"])
    with pytest.raises(ValueError, match="retrieval_confidence"):
        diff_evidence(b, a, retrieval_tolerances={"ndcg_at_k": 0.01}, retrieval_confidence=1.5)


def test_without_per_query_values_the_mean_rule_applies():
    x, q, qrels, *_ = _data()
    base = audit(x, k=10, queries=q, qrels=qrels).to_dict()
    old = {**base, "retrieval": {k: v for k, v in base["retrieval"].items() if k != "per_query"}}
    d = diff_evidence(base, old, retrieval_tolerances={"ndcg_at_k": 0.01}, retrieval_confidence=0.9)
    assert d["retrieval_confidence_checks"] == {}
    assert d["retrieval_regressions"] == [] and d["retrieval_unchecked"] == []


def test_agreed_top_hits_separate_a_lost_mapping_from_a_harmless_reorder():
    x, q, _, rx, rq = _data()
    kw = {"k": 10, "queries": q, "result_sample": 200}
    base = audit(x, reference_embeddings=rx, reference_queries=rq, **kw)
    ro = base.result_sample["reference_overlap"]
    assert len(ro["queries"]) == len(base.result_sample["queries"]) == 200
    assert np.mean(ro["queries"]) > 0.5  # a rotation keeps cosine neighbors
    first = base.result_sample["reference_first"]
    assert len(first["queries"]) == 200 and len(first["rows"]) == 200
    base = base.to_dict()

    # harmless: a tiny perturbation changes some results but keeps the agreed top hits
    jit = (x + 1e-3 * np.random.default_rng(3).standard_normal(x.shape)).astype(np.float32)
    ok = audit(jit, reference_embeddings=rx, reference_queries=rq, **kw).to_dict()
    d = diff_evidence(ok, base, reference_tolerance=0.05)
    assert d["reference_regressions"] == []
    hit = d["result_agreement"]["queries"]["reference_top_hit"]
    assert hit["agreed"] > 150 and hit["lost_share"] < 0.05
    assert abs(d["result_agreement"]["queries"]["reference"]["drop"]) < 0.02
    assert "were kept" in " ".join(d["change_decision"]["reasons"])

    # harmful: 30% of rows hold another document's vector, the reference (from the texts)
    # does not
    rng = np.random.default_rng(1)
    rows = rng.choice(len(x), len(x) * 3 // 10, replace=False)
    moved = x.copy()
    moved[rows] = x[np.roll(rows, 1)]
    bad = audit(moved, reference_embeddings=rx, reference_queries=rq, **kw).to_dict()
    d = diff_evidence(bad, base, reference_tolerance=0.05)
    hit = d["result_agreement"]["queries"]["reference_top_hit"]
    assert hit["lost_share"] > 0.1 and hit["z"] > 2
    assert d["reference_regressions"] == ["queries"]
    assert "lost the top hits" in " ".join(d["change_decision"]["reasons"])

    # only the baseline needs the reference
    plain = audit(moved, **kw).to_dict()
    d = diff_evidence(plain, base, reference_tolerance=0.05)
    assert d["reference_regressions"] == ["queries"] and d["reference_unchecked"] == []
    # a baseline without a reference cannot be checked
    d = diff_evidence(bad, plain, reference_tolerance=0.05)
    assert d["reference_unchecked"] == ["reference"]


def test_top_hit_loss_counts_only_agreed_items():
    from vecscope.results import _top_hit, reference_regressions

    # item 0: agreed (first 5 in the baseline) and lost; item 1: agreed, kept;
    # item 2: not agreed (first 9 was never in the baseline), so it does not count
    hit = _top_hit(
        [0, 1, 2], [[5, 6], [7, 8], [1, 2]], [5, 7, 9], [0, 1, 2], [[1, 6], [7, 3], [9, 4]]
    )
    assert hit["agreed"] == 2 and hit["lost"] == 1 and hit["lost_share"] == 0.5
    agreement = {"queries": {"reference_top_hit": hit}, "rows": {"reference_top_hit": hit}}
    assert reference_regressions(agreement, 0.6) == []
    # se of [1, 0] is 0.5, z = 1: not beyond 2 standard errors
    assert reference_regressions(agreement, 0.05) == []


def test_cli_confidence_and_reference_gates(tmp_path, capsys):
    x, q, qrels, rx, rq = _data()
    for name, arr in (("x", x), ("q", q), ("rx", rx), ("rq", rq)):
        np.save(tmp_path / f"{name}.npy", arr)
    (tmp_path / "qrels.json").write_text(json.dumps({str(k): v for k, v in qrels.items()}))
    rows = np.random.default_rng(1).choice(len(x), len(x) * 3 // 10, replace=False)
    moved = x.copy()
    moved[rows] = x[np.roll(rows, 1)]
    np.save(tmp_path / "moved.npy", moved)
    p = {n: str(tmp_path / f"{n}.npy") for n in ("x", "q", "rx", "rq", "moved")}
    common = [
        "--queries", p["q"], "--qrels", str(tmp_path / "qrels.json"), "--result-sample", "200",
        "--reference", p["rx"], "--reference-queries", p["rq"], "--fail-on", "none",
    ]  # fmt: skip
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(["audit", p["x"], "--json", *common]) == 0
    base = tmp_path / "base.json"
    base.write_text(buf.getvalue())
    gate = [
        "--baseline", str(base), "--retrieval-tolerance", "ndcg_at_k=0.01",
        "--retrieval-confidence", "0.95", "--reference-tolerance", "0.05",
    ]  # fmt: skip
    assert main(["audit", p["x"], *common, *gate]) == 0
    capsys.readouterr()
    assert main(["audit", p["moved"], *common, *gate]) == 1
    out = capsys.readouterr().out
    assert "top hits agreed with the reference model lost" in out
    assert "AGREED TOP HITS LOST" in out
    assert main(["audit", p["x"], "--retrieval-confidence", "0.9"]) == 2
    assert main(["audit", p["x"], "--baseline", str(base), "--reference-tolerance", "0.05"]) == 2
    err = capsys.readouterr().err
    assert "needs --result-sample on this run" in err


def test_mcp_keeps_per_query_values_for_the_baseline_only(tmp_path, monkeypatch):
    monkeypatch.setenv("VECSCOPE_ALLOWED_DIRS", str(tmp_path))
    pytest.importorskip("mcp")
    from vecscope.mcp_server import audit_embeddings

    x, q, qrels, rx, rq = _data()
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "q.npy", q)
    np.save(tmp_path / "rx.npy", rx)
    np.save(tmp_path / "rq.npy", rq)
    (tmp_path / "qrels.json").write_text(json.dumps({str(k): v for k, v in qrels.items()}))
    kw = {
        "queries_path": str(tmp_path / "q.npy"),
        "qrels_path": str(tmp_path / "qrels.json"),
        "reference_path": str(tmp_path / "rx.npy"),
        "reference_queries_path": str(tmp_path / "rq.npy"),
        "result_sample": 100,
    }
    base = audit_embeddings(str(tmp_path / "x.npy"), **kw)
    assert base["retrieval"]["per_query"] == {"kept_for_baseline": True, "queries": 300}
    res = audit_embeddings(
        str(tmp_path / "x.npy"),
        baseline=base["audit_id"],
        retrieval_tolerances={"ndcg_at_k": 0.01},
        retrieval_confidence=0.95,
        reference_tolerance=0.05,
        **kw,
    )
    d = res["baseline_diff"]
    assert d["retrieval_confidence_checks"]["exact.ndcg_at_k"]["decision"] == "pass"
    assert d["reference_regressions"] == []
    assert "top hits agreed with the reference lost" in res["summary"]


def test_a_small_loss_of_agreed_top_hits_is_listed_for_review_not_failed():
    from vecscope.results import reference_regressions, reference_review, result_agreement

    n = 200
    ids = list(range(n))
    res = [[i * 10 + j for j in range(10)] for i in ids]
    old = {
        "k": 10,
        "rows": [],
        "row_neighbors": [],
        "queries": ids,
        "n_queries": n,
        "query_results": res,
        "reference_first": {"queries": [r[0] for r in res]},
    }
    # 5 of 200 agreed top hits lost: a net drop of 2.5%, between the review share (0.015)
    # and the default tolerance (0.03)
    new_res = [([-1] + r[1:]) if i < 5 else r for i, r in zip(ids, res)]
    new = {**old, "query_results": new_res}
    del new["reference_first"]
    agreement = result_agreement(new, old)
    hit = agreement["queries"]["reference_top_hit"]
    assert hit["lost"] == 5 and hit["gained"] == 0 and hit["net_drop"] == 0.025
    assert hit["net_z"] > 2
    assert reference_regressions(agreement, 0.03) == []
    assert reference_review(agreement, 0.03) == ["queries"]
    assert reference_review(agreement, None) == ["queries"]
    # at a tolerance below the drop it is a regression, not a review
    assert reference_regressions(agreement, 0.02) == ["queries"]
    assert reference_review(agreement, 0.02) == []


def test_agreed_top_hits_gained_offset_the_lost_ones():
    """A new model loses some agreed top hits and finds the reference's first result where
    the baseline missed it: the net drop, not the lost share, decides."""
    from vecscope.results import reference_regressions, reference_review, result_agreement

    n = 200
    ids = list(range(n))
    res = [[i * 10 + j for j in range(10)] for i in ids]
    # the reference's first result is in the baseline's top-k for the first 150 queries only
    first = [r[0] if i < 150 else -2 for i, r in enumerate(res)]
    old = {
        "k": 10,
        "rows": [],
        "row_neighbors": [],
        "queries": ids,
        "n_queries": n,
        "query_results": res,
        "reference_first": {"queries": first},
    }
    # 20 agreed hits lost (13% of 150), 30 gained on the queries the baseline missed
    new_res = [
        [-1] + r[1:] if i < 20 else [-2] + r[1:] if i >= 170 else r for i, r in zip(ids, res)
    ]
    new = {**old, "query_results": new_res}
    del new["reference_first"]
    agreement = result_agreement(new, old)
    hit = agreement["queries"]["reference_top_hit"]
    assert (hit["agreed"], hit["lost"], hit["gained"]) == (150, 20, 30)
    assert hit["lost_share"] > 0.13 and hit["z"] > 2
    assert hit["net_drop"] == -0.05 and hit["net_z"] < 0
    assert reference_regressions(agreement, 0.03) == []
    assert reference_review(agreement, None) == []
