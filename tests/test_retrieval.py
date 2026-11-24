"""Retrieval quality next to the findings: exact nDCG/recall/MRR against hand-computed
values, duplicate share, a supplied ranking, qrels formats, the baseline diff, the CLI."""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

from vecscope import audit
from vecscope.cli import main
from vecscope.evidence import diff_evidence
from vecscope.retrieval import evaluate_retrieval, load_qrels, parse_qrels


def _unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope="module")
def labeled():
    """Queries that sit next to a known document; the qrels mark that document."""
    rng = np.random.default_rng(5)
    corpus = _unit(rng.standard_normal((400, 24)))
    targets = rng.choice(400, 60, replace=False)
    queries = _unit(corpus[targets] + rng.standard_normal((60, 24)).astype(np.float32) * 0.15)
    qrels = {q: {int(d): 1} for q, d in enumerate(targets)}
    return corpus, queries, qrels


def test_scores_match_a_hand_computed_ranking():
    corpus = _unit(np.eye(5, dtype=np.float32) + 0.01)
    queries = corpus[[0, 1]].copy()
    # query 0: relevant docs 0 (grade 2) and 3 (grade 1); query 1: doc 4 (not in its top 2)
    qrels = {0: {0: 2, 3: 1}, 1: {4: 1}}
    r = evaluate_retrieval(corpus, queries, qrels, k=2)
    # exact top-2 of query 0 is doc 0 then a tied doc; doc 3 is found only if it is second
    top0 = r["exact"]
    dcg_best = (2**2 - 1) / math.log2(2)
    idcg0 = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert r["judged_queries"] == 2 and r["k"] == 2
    assert top0["mrr_at_k"] == pytest.approx((1.0 + 0.0) / 2)
    assert top0["hit_rate_at_k"] == pytest.approx(0.5)
    assert top0["ndcg_at_k"] >= (dcg_best / idcg0) / 2 - 1e-12  # query 1 scores 0
    # a supplied ranking that puts the relevant docs first is perfect
    ranked = np.array([[0, 3], [4, 1]])
    p = evaluate_retrieval(corpus, queries, qrels, k=2, ranked=ranked)["ranked"]
    assert p["ndcg_at_k"] == pytest.approx(1.0) and p["recall_at_k"] == pytest.approx(1.0)
    assert p["mrr_at_k"] == pytest.approx(1.0)


def test_exact_search_finds_the_planted_documents(labeled):
    corpus, queries, qrels = labeled
    r = evaluate_retrieval(corpus, queries, qrels, k=10)
    assert r["exact"]["recall_at_k"] > 0.95 and r["exact"]["ndcg_at_k"] > 0.9
    assert r["exact"]["duplicate_result_share"] == 0.0


def test_duplicate_share_rises_with_copies_and_falls_after_dedup(labeled):
    corpus, queries, qrels = labeled
    copies = np.repeat(corpus, 3, axis=0)  # every document three times
    dup_qrels = {q: {3 * d: 1 for d in docs} for q, docs in qrels.items()}
    dup = evaluate_retrieval(copies, queries, dup_qrels, k=9)["exact"]
    assert dup["duplicate_result_share"] == pytest.approx(2 / 3, abs=1e-9)
    clean = evaluate_retrieval(corpus, queries, qrels, k=9)["exact"]
    assert clean["duplicate_result_share"] == 0.0
    assert clean["recall_at_k"] >= dup["recall_at_k"]


def test_ranked_overlap_falls_with_a_worse_index(labeled):
    corpus, queries, qrels = labeled
    rng = np.random.default_rng(0)
    exact = evaluate_retrieval(corpus, queries, qrels, k=10)
    from vecscope.diagnostics.queries import query_knn

    top, _ = query_knn(corpus, queries, 10)
    noisy = top.copy()
    noisy[:, 5:] = rng.integers(0, 400, (60, 5))  # half of each list replaced
    good = evaluate_retrieval(corpus, queries, qrels, k=10, ranked=top)["ranked"]
    bad = evaluate_retrieval(corpus, queries, qrels, k=10, ranked=noisy)["ranked"]
    assert good["overlap_with_exact"] == pytest.approx(1.0)
    assert good["ndcg_at_k"] == pytest.approx(exact["exact"]["ndcg_at_k"])
    assert bad["overlap_with_exact"] < 0.6


@pytest.mark.parametrize(
    "obj",
    [
        {"0": {"3": 1, "5": 0}},
        [[0, 3, 1], [0, 5, 0]],
        [{"query": 0, "doc": 3, "grade": 1}, {"query-id": 0, "corpus-id": 5, "score": 0}],
    ],
)
def test_parse_qrels_shapes(obj):
    assert parse_qrels(obj) == {0: {3: 1.0, 5: 0.0}}


@pytest.mark.parametrize(
    "obj, err",
    [
        ({}, ValueError),
        ([[0, 1, 0]], ValueError),
        ([[0.5, 1]], ValueError),
        ([[-1, 2]], ValueError),
        ([[True, 1]], TypeError),
        ("x", TypeError),
        ({"0": [1]}, TypeError),
        ([[0, 1, "nan"]], ValueError),
    ],
)
def test_parse_qrels_rejects(obj, err):
    with pytest.raises(err):
        parse_qrels(obj)


def test_load_qrels_formats(tmp_path):
    want = {0: {3: 1.0}, 2: {7: 2.0}}
    (tmp_path / "q.json").write_text(json.dumps({"0": {"3": 1}, "2": {"7": 2}}))
    (tmp_path / "q.jsonl").write_text(
        '{"query": 0, "doc": 3}\n{"query": 2, "doc": 7, "grade": 2}\n'
    )
    (tmp_path / "q.tsv").write_text("query-id\tcorpus-id\tscore\n0\t3\t1\n2\t7\t2\n")
    (tmp_path / "q.csv").write_text("0,3,1\n2,7,2\n")
    for name in ("q.json", "q.jsonl", "q.tsv", "q.csv"):
        assert load_qrels(tmp_path / name) == want, name
    with pytest.raises(FileNotFoundError):
        load_qrels(tmp_path / "missing.tsv")


def test_queries_without_a_relevant_judgment_are_counted_not_scored():
    corpus = _unit(np.eye(4, dtype=np.float32) + 0.01)
    queries = corpus[[0, 1, 2]].copy()
    # query 0 has a relevant doc, query 1 only a grade-0 judgment, query 2 none at all
    qrels = parse_qrels({0: {0: 1}, 1: {1: 0}})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no division by zero, no mean of an empty list
        r = evaluate_retrieval(corpus, queries, qrels, k=2)
    assert r["judged_queries"] == 1 and r["unscored_queries"] == 2
    assert r["exact"]["ndcg_at_k"] == pytest.approx(1.0)
    assert "2 of 3 queries have no relevant judgment and are not scored" in r["explanation"]
    with pytest.raises(ValueError, match="no relevant"):
        evaluate_retrieval(corpus, queries, {1: {1: 0}}, k=2)  # unparsed, nothing relevant


def test_qrels_must_index_existing_rows(labeled):
    corpus, queries, _ = labeled
    with pytest.raises(ValueError, match="query row 60"):
        evaluate_retrieval(corpus, queries, {60: {1: 1}})
    with pytest.raises(ValueError, match="corpus row 400"):
        evaluate_retrieval(corpus, queries, {0: {400: 1}})
    with pytest.raises(ValueError, match="ranked"):
        evaluate_retrieval(corpus, queries, {0: {1: 1}}, ranked=np.zeros((5, 10), dtype=int))


def test_audit_attaches_retrieval_and_diff_reports_its_change(labeled):
    corpus, queries, qrels = labeled
    base = audit(corpus, k=10, queries=queries, qrels=qrels).to_dict()
    assert base["retrieval"]["exact"]["recall_at_k"] > 0.95
    # the same corpus with a rogue dimension: a finding fails and quality moves with it
    hurt = corpus.copy()
    hurt[:, 0] += 3.0
    new = audit(hurt, k=10, queries=queries, qrels=qrels).to_dict()
    d = diff_evidence(new, base)
    assert d["retrieval_comparable"] is True
    rows = {(r["ranking"], r["metric"]): r for r in d["retrieval_changes"]}
    assert rows[("exact", "ndcg_at_k")]["delta"] < 0
    assert "rogue_dimensions" in d["regressions"] or "anisotropy" in d["regressions"]
    other = audit(corpus, k=10, queries=queries, qrels={0: qrels[0]}).to_dict()
    assert diff_evidence(other, base)["retrieval_comparable"] is False


def test_audit_scores_every_row_even_when_findings_are_sampled(labeled):
    corpus, queries, qrels = labeled
    full = audit(corpus, k=10, queries=queries, qrels=qrels, sample=0)
    sampled = audit(corpus, k=10, queries=queries, qrels=qrels, sample=200)
    assert sampled.sampled and sampled.retrieval["exact"] == full.retrieval["exact"]


def test_audit_qrels_needs_queries(labeled):
    corpus, _, qrels = labeled
    with pytest.raises(ValueError, match="need queries"):
        audit(corpus, k=10, qrels=qrels)


def test_cli_qrels_and_ranked(tmp_path, capsys, labeled):
    corpus, queries, qrels = labeled
    from vecscope.diagnostics.queries import query_knn

    top, _ = query_knn(corpus, queries, 10)
    np.save(tmp_path / "c.npy", corpus)
    np.save(tmp_path / "q.npy", queries)
    np.save(tmp_path / "r.npy", top)
    (tmp_path / "qrels.json").write_text(
        json.dumps({str(q): {str(d): g for d, g in v.items()} for q, v in qrels.items()})
    )
    assert (
        main(
            [
                "audit",
                str(tmp_path / "c.npy"),
                "--queries",
                str(tmp_path / "q.npy"),
                "--qrels",
                str(tmp_path / "qrels.json"),
                "--ranked",
                str(tmp_path / "r.npy"),
                "--fail-on",
                "none",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "retrieval (60 judged queries, k=10):" in out and "overlap_with_exact=1.0000" in out
    main(
        [
            "audit",
            str(tmp_path / "c.npy"),
            "--queries",
            str(tmp_path / "q.npy"),
            "--qrels",
            str(tmp_path / "qrels.json"),
            "--json",
        ]
    )
    (tmp_path / "base.json").write_text(capsys.readouterr().out)
    main(
        [
            "audit",
            str(tmp_path / "c.npy"),
            "--queries",
            str(tmp_path / "q.npy"),
            "--qrels",
            str(tmp_path / "qrels.json"),
            "--baseline",
            str(tmp_path / "base.json"),
        ]
    )
    assert "retrieval vs baseline:" in capsys.readouterr().out


# -- the retrieval regression gate -----------------------------------------------------


def _write_labeled(tmp_path, corpus, queries, qrels, name="c"):
    np.save(tmp_path / f"{name}.npy", corpus)
    np.save(tmp_path / "q.npy", queries)
    (tmp_path / "qrels.json").write_text(
        json.dumps({str(q): {str(d): g for d, g in v.items()} for q, v in qrels.items()})
    )


def _permuted(corpus, seed=0):
    """The same vectors in another row order: what a reindex that loses the id mapping does.
    Every finding is invariant to row order, so only labeled retrieval can see it."""
    return corpus[np.random.default_rng(seed).permutation(len(corpus))]


def test_findings_cannot_see_a_lost_id_mapping_but_the_retrieval_gate_can(labeled):
    corpus, queries, qrels = labeled
    base = audit(corpus, k=10, queries=queries, qrels=qrels, sample=0).to_dict()
    new = audit(_permuted(corpus), k=10, queries=queries, qrels=qrels, sample=0).to_dict()
    plain = diff_evidence(new, base)
    assert plain["regressions"] == [] and plain["value_regressions"] == []
    assert plain["retrieval_regressions"] == [] and plain["retrieval_unchecked"] == []
    d = diff_evidence(new, base, retrieval_tolerances={"ndcg_at_k": 0.01})
    assert d["retrieval_regressions"] == ["exact.ndcg_at_k"]
    row = next(r for r in d["retrieval_changes"] if r["metric"] == "ndcg_at_k")
    assert row["worsening"] > 0.5 and row["tolerance"] == 0.01
    # unchanged vectors stay inside a zero tolerance
    same = diff_evidence(base, base, retrieval_tolerances={"ndcg_at_k": 0, "recall_at_k": 0})
    assert same["retrieval_regressions"] == [] and same["retrieval_unchecked"] == []


def test_retrieval_tolerance_direction_and_rankings(labeled):
    corpus, queries, qrels = labeled
    from vecscope.diagnostics.queries import query_knn

    top, _ = query_knn(corpus, queries, 10)
    worse = top[:, ::-1].copy()  # same documents, the relevant one pushed to the bottom
    base = audit(corpus, k=10, queries=queries, qrels=qrels, ranked=top).to_dict()
    new = audit(corpus, k=10, queries=queries, qrels=qrels, ranked=worse).to_dict()
    # a bare key covers both rankings; recall@10 is order-blind, nDCG and MRR are not
    d = diff_evidence(new, base, retrieval_tolerances={"ndcg_at_k": 0.01, "recall_at_k": 0})
    assert d["retrieval_regressions"] == ["ranked.ndcg_at_k"]
    # a prefixed key checks only its ranking; overlap_with_exact is ranked-only
    d = diff_evidence(new, base, retrieval_tolerances={"exact.mrr_at_k": 0})
    assert d["retrieval_regressions"] == []
    d = diff_evidence(new, base, retrieval_tolerances={"overlap_with_exact": 0})
    assert d["retrieval_regressions"] == [] and d["retrieval_unchecked"] == []
    # an improvement is never a regression, a higher duplicate share is
    d = diff_evidence(base, new, retrieval_tolerances={"ndcg_at_k": 0})
    assert d["retrieval_regressions"] == []
    dup = json.loads(json.dumps(new))
    dup["retrieval"]["exact"]["duplicate_result_share"] += 0.2
    d = diff_evidence(dup, new, retrieval_tolerances={"duplicate_result_share": 0.1})
    assert d["retrieval_regressions"] == ["exact.duplicate_result_share"]


def test_retrieval_tolerance_that_cannot_be_checked_is_reported(labeled):
    corpus, queries, qrels = labeled
    base = audit(corpus, k=10, queries=queries, qrels=qrels).to_dict()
    unlabeled = audit(corpus, k=10, queries=queries).to_dict()
    other_qrels = audit(corpus, k=10, queries=queries, qrels={0: qrels[0]}).to_dict()
    rtol = {"ndcg_at_k": 0.01}
    for new, old in ((base, unlabeled), (unlabeled, base), (other_qrels, base)):
        d = diff_evidence(new, old, retrieval_tolerances=rtol)
        assert d["retrieval_regressions"] == [] and d["retrieval_unchecked"] == ["exact.ndcg_at_k"]
    # a ranking the tolerance names explicitly must exist on both runs
    d = diff_evidence(base, base, retrieval_tolerances={"ranked.ndcg_at_k": 0.01})
    assert d["retrieval_unchecked"] == ["ranked.ndcg_at_k"]
    # ... while a bare key skips a ranking neither run supplied
    assert diff_evidence(base, base, retrieval_tolerances=rtol)["retrieval_unchecked"] == []


@pytest.mark.parametrize(
    "tol,err",
    [
        ({"ndcg": 0.01}, "unknown retrieval metric"),
        ({"fast.ndcg_at_k": 0.01}, "unknown retrieval metric"),
        ({"exact.overlap_with_exact": 0.01}, "ranked results only"),
        ({"ndcg_at_k": -0.01}, ">= 0"),
        ({"ndcg_at_k": float("nan")}, ">= 0"),
        ({"ndcg_at_k": True}, ">= 0"),
    ],
)
def test_retrieval_tolerances_are_validated(tol, err):
    from vecscope.evidence import resolve_retrieval_tolerances

    with pytest.raises(ValueError, match=err):
        resolve_retrieval_tolerances(tol)


def test_parse_retrieval_tolerance_flags():
    from vecscope.evidence import parse_retrieval_tolerance

    assert parse_retrieval_tolerance(["ndcg_at_k=0.01", "ranked.recall_at_k=0.02"]) == {
        "ndcg_at_k": 0.01,
        "ranked.recall_at_k": 0.02,
    }
    assert parse_retrieval_tolerance(None) == {}
    for bad in ("ndcg_at_k", "=0.1", "ndcg_at_k=x"):
        with pytest.raises(ValueError, match="--retrieval-tolerance"):
            parse_retrieval_tolerance([bad])


def test_cli_retrieval_tolerance_gates_the_run(tmp_path, capsys, labeled):
    corpus, queries, qrels = labeled
    _write_labeled(tmp_path, corpus, queries, qrels)
    np.save(tmp_path / "moved.npy", _permuted(corpus))
    common = ["--queries", str(tmp_path / "q.npy"), "--qrels", str(tmp_path / "qrels.json")]
    common += ["--sample", "0", "--fail-on", "none"]
    assert main(["audit", str(tmp_path / "c.npy"), "--json", *common]) == 0
    (tmp_path / "base.json").write_text(capsys.readouterr().out)
    gate = ["--baseline", str(tmp_path / "base.json")]
    # without a retrieval tolerance the lost mapping passes: no finding moved
    assert main(["audit", str(tmp_path / "moved.npy"), *common, *gate]) == 0
    capsys.readouterr()
    gate += ["--retrieval-tolerance", "ndcg_at_k=0.01"]
    assert main(["audit", str(tmp_path / "moved.npy"), *common, *gate]) == 1
    out = capsys.readouterr().out
    assert "retrieval worse than baseline beyond tolerance: exact.ndcg_at_k" in out
    assert "WORSE by" in out
    assert "change decision: reject; retrieval dropped: exact.ndcg_at_k" in out
    assert main(["audit", str(tmp_path / "c.npy"), *common, *gate]) == 0
    assert "within 0.01" in capsys.readouterr().out
    assert main(["audit", str(tmp_path / "moved.npy"), "--json", *common, *gate]) == 1
    g = json.loads(capsys.readouterr().out)["gate"]
    assert g["retrieval_regressions"] == ["exact.ndcg_at_k"] and g["exit_code"] == 1


def test_cli_retrieval_tolerance_unchecked_fails_and_needs_its_inputs(tmp_path, capsys, labeled):
    corpus, queries, qrels = labeled
    _write_labeled(tmp_path, corpus, queries, qrels)
    c, q = str(tmp_path / "c.npy"), ["--queries", str(tmp_path / "q.npy")]
    assert main(["audit", c, *q, "--json", "--fail-on", "none"]) == 0
    (tmp_path / "unlabeled.json").write_text(capsys.readouterr().out)
    run = [*q, "--qrels", str(tmp_path / "qrels.json"), "--fail-on", "none"]
    run += ["--baseline", str(tmp_path / "unlabeled.json")]
    run += ["--retrieval-tolerance", "ndcg_at_k=0.01"]
    assert main(["audit", c, *run]) == 1
    assert "retrieval tolerance not checked: exact.ndcg_at_k" in capsys.readouterr().out
    assert main(["audit", c, *run, "--require-comparable"]) == 2
    assert "cannot be compared on exact.ndcg_at_k" in capsys.readouterr().err
    # the flag needs a baseline and labels
    assert main(["audit", c, *q, "--retrieval-tolerance", "ndcg_at_k=0.01"]) == 2
    assert "need --baseline" in capsys.readouterr().err
    no_qrels = [*q, "--baseline", str(tmp_path / "unlabeled.json")]
    assert main(["audit", c, *no_qrels, "--retrieval-tolerance", "ndcg_at_k=0.01"]) == 2
    assert "needs --qrels" in capsys.readouterr().err


def test_gate_decision_reports_retrieval_regressions():
    from vecscope.cli import gate_decision

    d = {
        "regressions": [],
        "value_regressions": [],
        "retrieval_regressions": ["exact.ndcg_at_k"],
        "retrieval_unchecked": [],
        "retrieval_changes": [
            {
                "ranking": "exact",
                "metric": "ndcg_at_k",
                "value": 0.5,
                "baseline_value": 0.9,
                "worsening": 0.4,
                "tolerance": 0.01,
            }
        ],
    }
    g = gate_decision("ok", "none", d)
    assert g["exit_code"] == 1 and g["retrieval_regressions"] == ["exact.ndcg_at_k"]
    assert "0.9000 -> 0.5000 (worse by 0.4000 > 0.01)" in g["reasons"][0]
    g = gate_decision(
        "ok", "none", {**d, "retrieval_regressions": [], "retrieval_unchecked": ["x"]}
    )
    assert g["exit_code"] == 1 and "not checked: x" in g["reasons"][0]
