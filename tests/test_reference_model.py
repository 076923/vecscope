"""Judging findings against a reference model that embedded the same rows: a finding both
models reach comes from the rows and reads ok; a finding only this model reaches stays."""

from __future__ import annotations

import numpy as np
import pytest

from vecscope import audit
from vecscope.cli import main


def _embed(latent, dim, rng, noise=0.01):
    """A "model": a random linear map of the rows' latent content plus a little noise."""
    w = rng.standard_normal((latent.shape[1], dim))
    return (latent @ w + noise * rng.standard_normal((latent.shape[0], dim))).astype(np.float32)


def _f(ev, name):
    return next(f for f in ev.findings if f.name == name)


def test_low_rank_rows_are_explained_by_the_reference():
    """Rows with three degrees of freedom (templated texts) give every model a low effective
    rank; the reference reaches it too, so it reads ok and keeps the verdict without it."""
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((1500, 3))
    x, ref = _embed(latent, 128, rng), _embed(latent, 96, rng)
    plain = _f(audit(x, k=10), "effective_rank")
    assert plain.verdict == "fail"
    ev = audit(x, k=10, reference_embeddings=ref)
    f = _f(ev, "effective_rank")
    assert f.verdict == "ok"
    assert f.details["reference"]["verdict_without_reference"] == "fail"
    assert f.details["reference"]["verdict"] == "fail"
    assert "EXPLAINED BY THE ROWS" in f.explanation
    assert "effective_rank" in ev.reference["explained"]
    assert ev.reference["dim"] == 96
    assert ev.reference["findings"]["effective_rank"]["verdict"] == "fail"
    # explained findings leave the plan
    assert all("effective_rank" not in [g["name"] for g in s["findings"]] for s in ev.plan)


def test_a_fault_only_this_model_has_stays_flagged():
    """20% of rows mapped to one vector by this model only: the reference has distinct rows,
    so collapse is not explained."""
    rng = np.random.default_rng(1)
    latent = rng.standard_normal((1500, 64))
    x, ref = _embed(latent, 128, rng, 0.1), _embed(latent, 96, rng, 0.1)
    x[:300] = x[0]
    ev = audit(x, k=10, reference_embeddings=ref)
    f = _f(ev, "collapse")
    assert f.verdict == "fail"
    assert f.details["reference"]["verdict"] == "ok"
    assert "collapse" not in ev.reference["explained"]


def test_output_format_findings_are_never_explained():
    """Unnormalized rows of varying length in both models: norms describes the model output,
    not the rows, so the reference cannot explain it."""
    rng = np.random.default_rng(2)
    latent = rng.standard_normal((1000, 32))
    scale = rng.uniform(0.2, 3.0, (1000, 1)).astype(np.float32)
    x, ref = _embed(latent, 64, rng) * scale, _embed(latent, 64, rng) * scale
    ev = audit(x, k=10, reference_embeddings=ref)
    f = _f(ev, "norms")
    assert f.verdict != "ok"
    assert "reference" not in f.details
    assert "norms" not in ev.reference["explained"]


def test_reference_query_side():
    rng = np.random.default_rng(3)
    latent = rng.standard_normal((1200, 3))
    qlat = rng.standard_normal((300, 3))
    wa, wb = rng.standard_normal((3, 64)), rng.standard_normal((3, 48))
    x, ref = (latent @ wa).astype(np.float32), (latent @ wb).astype(np.float32)
    q, rq = (qlat @ wa).astype(np.float32), (qlat @ wb).astype(np.float32)
    ev = audit(x, k=10, queries=q, reference_embeddings=ref, reference_queries=rq)
    assert ev.reference["n_queries"] == 300
    assert "reference" in _f(ev, "query_gap").details


def test_reference_inputs_are_checked():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((200, 16)).astype(np.float32)
    with pytest.raises(ValueError, match="same 200 rows"):
        audit(x, k=5, reference_embeddings=x[:150])
    with pytest.raises(ValueError, match="need reference_embeddings"):
        audit(x, k=5, queries=x[:10], reference_queries=x[:10])
    with pytest.raises(ValueError, match="need queries"):
        audit(x, k=5, reference_embeddings=x, reference_queries=x[:10])
    with pytest.raises(ValueError, match="same 10 queries"):
        audit(x, k=5, queries=x[:10], reference_embeddings=x, reference_queries=x[:9])


def test_cli_reference(tmp_path, capsys):
    rng = np.random.default_rng(5)
    latent = rng.standard_normal((800, 3))
    np.save(tmp_path / "x.npy", _embed(latent, 64, rng))
    np.save(tmp_path / "ref.npy", _embed(latent, 48, rng))
    main(["audit", str(tmp_path / "x.npy"), "--reference", str(tmp_path / "ref.npy")])
    assert "EXPLAINED BY THE ROWS" in capsys.readouterr().out
    capsys.readouterr()
    assert (
        main(["audit", str(tmp_path / "x.npy"), "--reference-queries", str(tmp_path / "x.npy")])
        == 2
    )
    assert "needs --reference" in capsys.readouterr().err
