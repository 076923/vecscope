"""Exactness and recall scoring beyond the upstream tests' n=1,500 x 32: several corpus
shapes, float64 references, the faiss CPU/GPU baselines the sweep uses, and an independent
recall scorer. ``slow`` tests take minutes (pytest -m slow). The 100k x 768 numbers from an
independent run are in docs/verification.md."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from bench_common import CORPORA, corpus
from exactness_helpers import d64, drop_self, strict_hits, truth

from vecscope._knn import exact_knn
from vecscope.diagnostics.ann_recall import tie_aware_hits, tie_tolerance
from vecscope.diagnostics.queries import query_knn

faiss = pytest.importorskip("faiss")


def _gpu():
    from vecscope.sweep import gpu_available
    return gpu_available(faiss)


def _check(ids, ref_dist, c64, qids):
    got = np.sort(d64(c64, qids, ids), axis=1)
    gap = np.abs(got - ref_dist)
    return int((gap > 1e-6).any(axis=1).sum()), float(gap.max())


@pytest.mark.parametrize("kind", CORPORA)
def test_exact_paths_match_float64_medium(kind):
    """20k x 128: exact_knn, query_knn and faiss flat agree with float64 up to ties."""
    c = corpus(kind, 20_000, 128, seed=3)
    c64 = c.astype(np.float64)
    qids = np.arange(300)
    _, ref_dist = truth(c, qids, 10)
    ki, kd = exact_knn(c, 10)
    assert _check(ki[qids], ref_dist, c64, qids)[0] == 0
    # reported float32 distances are within float32 rounding of float64
    assert np.abs(kd[qids].astype(np.float64) - np.sort(d64(c64, qids, ki[qids]), axis=1)).max() < 5e-6
    qi, _ = query_knn(c, c[qids], 11)
    assert _check(drop_self(qi, qids, 10), ref_dist, c64, qids)[0] == 0
    flat = faiss.IndexFlatIP(128)
    flat.add(c)
    _, fi = flat.search(c[qids], 11)
    assert _check(drop_self(fi, qids, 10), ref_dist, c64, qids)[0] == 0


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu(), reason="no faiss GPU")
@pytest.mark.parametrize("kind", ["clustered", "duplicates", "isotropic"])
def test_gpu_flat_baseline_matches_float64(kind):
    """The sweep's GPU exact baseline must be exact too (no TF32 / fp16 shortcuts)."""
    c = corpus(kind, 50_000, 256, seed=1)
    c64 = c.astype(np.float64)
    qids = np.arange(500)
    _, ref_dist = truth(c, qids, 10)
    res = faiss.StandardGpuResources()
    g = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(256))
    g.add(c)
    _, gi = g.search(c[qids], 11)
    errors, gap = _check(drop_self(gi, qids, 10), ref_dist, c64, qids)
    assert errors == 0, f"{errors} rows differ from float64 beyond ties (max gap {gap:.2e})"


@pytest.mark.slow
@pytest.mark.parametrize("kind", CORPORA)
def test_exact_knn_matches_float64_at_scale(kind):
    """100k x 384, 500 query rows: the blocked float32 kernel makes no real rank errors."""
    c = corpus(kind, 100_000, 384, seed=0)
    c64 = c.astype(np.float64)
    qids = np.arange(500)
    _, ref_dist = truth(c, qids, 10)
    ki, _ = exact_knn(c, 10)
    errors, gap = _check(ki[qids], ref_dist, c64, qids)
    assert errors == 0, f"{errors} rows with rank errors, max gap {gap:.2e}"


@pytest.mark.parametrize("kind", ["clustered", "duplicates", "zipf"])
def test_sweep_recall_matches_independent_scorer(kind):
    """sweep._recall vs a strict max matching written independently: equal on corpora whose
    neighbor distances are not concentrated (the tolerance below does not come into play)."""
    from vecscope.sweep import _recall

    c = corpus(kind, 20_000, 64, seed=5)
    c64 = c.astype(np.float64)
    qids = np.arange(400)
    _, ref_dist = truth(c, qids, 10)
    exq, _ = query_knn(c, c[qids], 11)
    ex = drop_self(exq, qids, 10)
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(64), 64, 256, faiss.METRIC_INNER_PRODUCT)
    ivf.train(c)
    ivf.add(c)
    ivf.nprobe = 2
    _, a = ivf.search(c[qids], 11)
    a = drop_self(a, qids, 10)
    mine = _recall(a, ex, c[qids], c)
    got = d64(c64, qids, a)
    strict = np.mean([strict_hits(r, g) for r, g in zip(ref_dist, got)]) / 10
    assert 0.2 < strict < 0.99  # the setting must actually miss neighbors to test anything
    assert mine - strict <= 0.002, (mine, strict)
    assert mine >= strict - 1e-12  # vecscope never scores below the strict count


def test_tie_tolerance_does_not_credit_a_farther_neighbor():
    exact = np.array([0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.705, 0.708, 0.709, 0.7100])
    # the index returned the 11th neighbor (distance 0.71001, 1e-5 farther: not a tie under
    # float32 rounding, ~1e-7) instead of the 10th
    returned = np.r_[exact[:9], 0.71001]
    assert tie_tolerance(0.71) < 1e-5  # the tolerance is narrower than this gap
    assert tie_aware_hits(exact, returned) == 9
