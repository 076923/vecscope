"""Retrieval quality next to the findings: exact nDCG/recall/MRR against hand-computed
values, duplicate share, a supplied ranking, qrels formats, the baseline diff, the CLI."""

from __future__ import annotations

import json
import math

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


@pytest.mark.parametrize("obj", [
    {"0": {"3": 1, "5": 0}},
    [[0, 3, 1], [0, 5, 0]],
    [{"query": 0, "doc": 3, "grade": 1}, {"query-id": 0, "corpus-id": 5, "score": 0}],
])
def test_parse_qrels_shapes(obj):
    assert parse_qrels(obj) == {0: {3: 1.0, 5: 0.0}}


@pytest.mark.parametrize("obj, err", [
    ({}, ValueError), ([[0, 1, 0]], ValueError), ([[0.5, 1]], ValueError),
    ([[-1, 2]], ValueError), ([[True, 1]], TypeError), ("x", TypeError),
    ({"0": [1]}, TypeError), ([[0, 1, "nan"]], ValueError),
])
def test_parse_qrels_rejects(obj, err):
    with pytest.raises(err):
        parse_qrels(obj)


def test_load_qrels_formats(tmp_path):
    want = {0: {3: 1.0}, 2: {7: 2.0}}
    (tmp_path / "q.json").write_text(json.dumps({"0": {"3": 1}, "2": {"7": 2}}))
    (tmp_path / "q.jsonl").write_text('{"query": 0, "doc": 3}\n{"query": 2, "doc": 7, "grade": 2}\n')
    (tmp_path / "q.tsv").write_text("query-id\tcorpus-id\tscore\n0\t3\t1\n2\t7\t2\n")
    (tmp_path / "q.csv").write_text("0,3,1\n2,7,2\n")
    for name in ("q.json", "q.jsonl", "q.tsv", "q.csv"):
        assert load_qrels(tmp_path / name) == want, name
    with pytest.raises(FileNotFoundError):
        load_qrels(tmp_path / "missing.tsv")


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
    (tmp_path / "qrels.json").write_text(json.dumps({str(q): {str(d): g for d, g in v.items()}
                                                     for q, v in qrels.items()}))
    assert main(["audit", str(tmp_path / "c.npy"), "--queries", str(tmp_path / "q.npy"),
                 "--qrels", str(tmp_path / "qrels.json"), "--ranked", str(tmp_path / "r.npy"),
                 "--fail-on", "none"]) == 0
    out = capsys.readouterr().out
    assert "retrieval (60 judged queries, k=10):" in out and "overlap_with_exact=1.0000" in out
    main(["audit", str(tmp_path / "c.npy"), "--queries", str(tmp_path / "q.npy"),
          "--qrels", str(tmp_path / "qrels.json"), "--json"])
    (tmp_path / "base.json").write_text(capsys.readouterr().out)
    main(["audit", str(tmp_path / "c.npy"), "--queries", str(tmp_path / "q.npy"),
          "--qrels", str(tmp_path / "qrels.json"), "--baseline", str(tmp_path / "base.json")])
    assert "retrieval vs baseline:" in capsys.readouterr().out
