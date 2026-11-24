"""Safe remediation: a change is tried on a copy and adopted only when labeled retrieval did
not drop; without labels a person decides; an unnecessary fix on a healthy corpus is
rejected; an agent that keeps trying fixes for one step is told to stop."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vecscope import audit
from vecscope.evidence import change_decision, diff_evidence
from vecscope.plan import ATTEMPT_LIMIT


def _unit(v):
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def _healthy(seed=1, n=1200, d=32):
    """Clustered vectors with a mild shared offset, labeled queries whose relevant documents
    are the query's whole cluster. With seed 1 every finding is ok, and centering, a fix no
    finding asks for here, lowers nDCG@10."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((40, d))
    lab = rng.integers(0, 40, n)
    x = centers[lab] + rng.standard_normal((n, d)) * 0.6
    u = _unit(rng.standard_normal((1, d)))[0] * np.sqrt(d) * 0.3
    qi = rng.choice(n, 100, replace=False)
    q = centers[lab[qi]] + rng.standard_normal((100, d)) * 1.1 + u
    qrels = {i: {int(j): 1 for j in np.flatnonzero(lab == lab[k])} for i, k in enumerate(qi)}
    return _unit(x + u), _unit(q), qrels


@pytest.fixture(scope="module")
def healthy():
    return _healthy()


def _center(x, q):
    mu = x.mean(axis=0, keepdims=True)
    return _unit(x - mu), _unit(q - mu)


def test_centering_a_healthy_corpus_is_rejected(healthy):
    x, q, qrels = healthy
    base = audit(x, k=10, queries=q, qrels=qrels, sample=0)
    assert base.verdict == "ok" and base.plan == []  # nothing asks for a fix
    cx, cq = _center(x, q)
    new = audit(cx, k=10, queries=cq, qrels=qrels, sample=0).to_dict()
    d = diff_evidence(new, base.to_dict())
    nd = next(
        c for c in d["retrieval_changes"] if c["ranking"] == "exact" and c["metric"] == "ndcg_at_k"
    )
    assert nd["delta"] < -0.005  # the "fix" costs retrieval
    cd = d["change_decision"]
    assert cd["decision"] == "reject"
    assert cd["reasons"][0].startswith("retrieval dropped: exact.ndcg_at_k")
    # the original is untouched: the copy was transformed, not x
    assert np.array_equal(x, healthy[0])


def test_a_change_that_does_nothing_is_rejected(healthy):
    x, q, qrels = healthy
    ev = audit(x, k=10, queries=q, qrels=qrels, sample=0).to_dict()
    again = audit(_unit(x.astype(np.float64)), k=10, queries=q, qrels=qrels, sample=0).to_dict()
    cd = diff_evidence(again, ev)["change_decision"]
    assert cd["decision"] == "reject" and "no measured gain" in cd["reasons"][0]


def test_without_labels_a_person_decides(healthy):
    x, q, _ = healthy
    base = audit(x, k=10, queries=q, sample=0).to_dict()
    cx, cq = _center(x, q)
    new = audit(cx, k=10, queries=cq, sample=0).to_dict()
    cd = diff_evidence(new, base)["change_decision"]
    assert cd["decision"] == "needs_review" and "let a person decide" in cd["reasons"][0]


def _diff(changes, improvements=(), regressions=(), **extra):
    return {
        "retrieval_comparable": True,
        "retrieval_changes": [
            {
                "ranking": r,
                "metric": m,
                "value": b + w * -1,
                "baseline_value": b,
                "worsening": w,
            }
            for r, m, b, w in changes
        ],
        "improvements": list(improvements),
        "regressions": list(regressions),
        "value_regressions": [],
        **extra,
    }


@pytest.mark.parametrize(
    "diff,decision,watch",
    [
        # nDCG up, a finding fixed
        (_diff([("exact", "ndcg_at_k", 0.5, -0.02)], ["anisotropy"]), "adopt", []),
        # a finding fixed, retrieval unchanged: adopt
        (_diff([("exact", "ndcg_at_k", 0.5, 0.0)], ["metric_mismatch"]), "adopt", []),
        # fixed, retrieval held, another finding got worse: adopt and watch it
        (
            _diff([("exact", "ndcg_at_k", 0.5, 0.0)], ["anisotropy"], ["outliers"]),
            "adopt",
            ["outliers"],
        ),
        # a finding fixed but recall fell: reject
        (_diff([("exact", "recall_at_k", 0.7, 0.01)], ["anisotropy"]), "reject", []),
        # the ranked results fell while exact search held: reject
        (
            _diff(
                [("exact", "ndcg_at_k", 0.5, 0.0), ("ranked", "ndcg_at_k", 0.5, 0.03)], ["hubness"]
            ),
            "reject",
            [],
        ),
        # float noise is not a drop
        (_diff([("exact", "ndcg_at_k", 0.5, 1e-12)], ["anisotropy"]), "adopt", []),
        # a duplicate-share rise alone does not decide it without a tolerance
        (_diff([("exact", "duplicate_result_share", 0.1, 0.05)], ["collapse"]), "adopt", []),
        # with tolerances, the gate's own lists decide
        (
            _diff(
                [("exact", "ndcg_at_k", 0.5, 0.005)],
                ["anisotropy"],
                retrieval_tolerances={"ndcg_at_k": 0.01},
                retrieval_regressions=[],
                retrieval_unchecked=[],
            ),
            "adopt",
            [],
        ),
        (
            _diff(
                [("exact", "ndcg_at_k", 0.5, 0.0)],
                ["anisotropy"],
                retrieval_tolerances={"ranked.ndcg_at_k": 0.01},
                retrieval_regressions=[],
                retrieval_unchecked=["ranked.ndcg_at_k"],
            ),
            "reject",
            [],
        ),
        ({"retrieval_comparable": False, "improvements": ["anisotropy"]}, "needs_review", []),
    ],
)
def test_change_decision_table(diff, decision, watch):
    cd = change_decision(diff)
    assert cd["decision"] == decision, cd
    assert cd["watch"] == watch and cd["reasons"]


def test_adopt_names_what_is_still_not_ok():
    d = _diff([("exact", "ndcg_at_k", 0.22, -0.01)], ["query_gap"])
    d["changes"] = [
        {"name": "query_gap", "verdict": "warn"},
        {"name": "anisotropy", "verdict": "ok"},
    ]
    cd = change_decision(d)
    assert cd["decision"] == "adopt" and cd["still_not_ok"] == ["query_gap"]
    assert "still not ok: query_gap" in cd["reasons"][-1]


def test_mcp_stops_an_agent_after_repeated_rejected_fixes(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from vecscope import mcp_server
    from vecscope.mcp_server import audit_embeddings

    monkeypatch.setattr(mcp_server, "_ATTEMPTS", {})
    # a corpus with a uniform offset on every row, so its plan has steps to try
    x, q, qrels = _healthy()
    rng = np.random.default_rng(0)
    u = _unit(rng.standard_normal((1, x.shape[1])))[0]
    x = _unit(x + 2.5 * u)
    q = _unit(q + 2.5 * u)
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "q.npy", q)
    (tmp_path / "qrels.json").write_text(
        json.dumps({str(a): {str(b): g for b, g in v.items()} for a, v in qrels.items()})
    )
    args = {"queries_path": str(tmp_path / "q.npy"), "qrels_path": str(tmp_path / "qrels.json")}
    monkeypatch.chdir(tmp_path)
    first = audit_embeddings(str(tmp_path / "x.npy"), k=10, sample=0, **args)
    assert first["plan"] and "adopt_when" in first["plan"][0]
    step = first["plan"][0]["group"]

    def tries():
        """Three attempts at the step, each with a mistake agents make: center the corpus but
        not the queries, keep only half the dimensions, subtract the queries' mean instead of
        the corpus mean from the corpus."""
        mu = x.mean(axis=0, keepdims=True)
        yield _unit(x - mu), q
        yield _unit((x - mu)[:, ::2]), _unit((q - mu)[:, ::2])
        yield _unit(x - 3 * q.mean(axis=0, keepdims=True)), q

    seen = []
    for i, (cx, cq) in enumerate(tries()):
        np.save(tmp_path / f"c{i}.npy", cx)
        np.save(tmp_path / f"cq{i}.npy", cq)
        rep = audit_embeddings(
            str(tmp_path / f"c{i}.npy"),
            k=10,
            sample=0,
            baseline=first["audit_id"],
            **{**args, "queries_path": str(tmp_path / f"cq{i}.npy")},
        )
        d = rep["baseline_diff"]
        seen.append((d["change_decision"]["decision"], d["attempts"], rep["summary"]))
    decisions = [s[0] for s in seen]
    assert "adopt" not in decisions, decisions
    assert [s[1]["not_adopted"] for s in seen] == [1, 2, 3]
    assert all(s[1]["step"] == step and s[1]["limit"] == ATTEMPT_LIMIT for s in seen)
    assert [s[1]["stop"] for s in seen] == [False, False, True]
    assert "STOP: 3 changes for step" in seen[-1][2] and "STOP" not in seen[0][2]
    # re-auditing the same candidate does not count twice
    again = audit_embeddings(
        str(tmp_path / "c0.npy"),
        k=10,
        sample=0,
        baseline=first["audit_id"],
        **{**args, "queries_path": str(tmp_path / "cq0.npy")},
    )
    assert again["baseline_diff"]["attempts"]["not_adopted"] == 3
