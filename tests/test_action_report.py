"""The GitHub Action's report step: step outputs and the job summary written from the
evidence JSON, including the retrieval gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from vecscope.cli import main

_spec = importlib.util.spec_from_file_location(
    "vecscope_action_report", Path(__file__).parents[1] / "action" / "report.py"
)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def _outputs(tmp_path, monkeypatch, evidence_path, code):
    out, summary = tmp_path / "out.txt", tmp_path / "summary.md"
    out.write_text("")
    summary.write_text("")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert report.main([str(evidence_path), str(code)]) == 0
    pairs = dict(ln.split("=", 1) for ln in out.read_text().splitlines())
    return pairs, summary.read_text()


def test_report_names_retrieval_regressions(tmp_path, capsys, monkeypatch):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((300, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q = x[:40] + rng.standard_normal((40, 16)).astype(np.float32) * 0.1
    np.save(tmp_path / "a.npy", x)
    np.save(tmp_path / "moved.npy", x[rng.permutation(len(x))])
    np.save(tmp_path / "q.npy", q)
    (tmp_path / "qrels.json").write_text(json.dumps({str(i): {str(i): 1} for i in range(40)}))
    common = ["--queries", str(tmp_path / "q.npy"), "--qrels", str(tmp_path / "qrels.json")]
    common += ["-k", "5", "--sample", "0", "--fail-on", "none", "--json"]
    main(["audit", str(tmp_path / "a.npy"), *common])
    (tmp_path / "base.json").write_text(capsys.readouterr().out)
    gate = ["--baseline", str(tmp_path / "base.json"), "--retrieval-tolerance", "ndcg_at_k=0.01"]
    code = main(["audit", str(tmp_path / "moved.npy"), *common, *gate])
    (tmp_path / "moved.json").write_text(capsys.readouterr().out)
    assert code == 1
    pairs, summary = _outputs(tmp_path, monkeypatch, tmp_path / "moved.json", code)
    assert pairs["retrieval-regressions"] == "exact.ndcg_at_k"
    assert pairs["regressions"] == "" and pairs["exit-code"] == "1"
    assert "Retrieval vs baseline:" in summary and "**exceeded**" in summary
    assert "retrieval worse than baseline beyond tolerance" in summary


def test_report_marks_unchecked_tolerances(tmp_path, monkeypatch):
    ev = {
        "verdict": "ok",
        "findings": [],
        "gate": {"retrieval_regressions": [], "retrieval_unchecked": ["exact.ndcg_at_k"]},
        "baseline_diff": {
            "verdict": "ok",
            "baseline_verdict": "ok",
            "retrieval_changes": [],
            "retrieval_unchecked": ["exact.ndcg_at_k"],
        },
    }
    (tmp_path / "ev.json").write_text(json.dumps(ev))
    pairs, summary = _outputs(tmp_path, monkeypatch, tmp_path / "ev.json", 1)
    assert pairs["retrieval-regressions"] == "unchecked:exact.ndcg_at_k"
    assert "Retrieval tolerances not checked: exact.ndcg_at_k" in summary


def test_report_without_evidence(tmp_path, monkeypatch):
    pairs, summary = _outputs(tmp_path, monkeypatch, tmp_path / "missing.json", 2)
    assert pairs["verdict"] == "error" and pairs["retrieval-regressions"] == ""
    assert "without evidence" in summary
