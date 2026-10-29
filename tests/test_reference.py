"""Reference checks: each measurement against an independent computation or a known answer,
and the audit against the invariances cosine findings must have (row order, rotation,
uniform scale)."""

from __future__ import annotations

import numpy as np
import pytest

from vecscope import audit
from vecscope._knn import exact_knn, normalize, row_overlap
from vecscope.diagnostics import ann_recall, collapse, distance_concentration, hubness
from vecscope.diagnostics.anisotropy import anisotropy


def _clustered(rng, n=1500, d=32, centers=25, spread=0.7):
    c = rng.standard_normal((centers, d))
    return (c[rng.integers(0, centers, n)] + rng.standard_normal((n, d)) * spread).astype(np.float32)


def _values(ev):
    return {f.name: f.value for f in ev.findings}


# ----------------------------------------------------------------------------- k-NN kernel


@pytest.mark.parametrize("metric", ["cosine", "dot", "euclidean"])
def test_exact_knn_matches_naive_full_matrix(metric):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((700, 12)).astype(np.float32) * rng.uniform(0.5, 2.0, (700, 1))
    idx, dist = exact_knn(x, 7, metric=metric, block=128)  # several blocks
    xd = x.astype(np.float64)
    if metric == "cosine":
        q = xd / np.linalg.norm(xd, axis=1, keepdims=True)
        full = 1.0 - q @ q.T
    elif metric == "dot":
        full = -(xd @ xd.T)
    else:
        full = ((xd[:, None, :] - xd[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(full, np.inf)
    ref = np.argsort(full, axis=1, kind="stable")[:, :7]
    # ids may differ only where distances tie; compare the distances themselves
    np.testing.assert_allclose(dist, np.take_along_axis(full, ref, axis=1), rtol=1e-4, atol=1e-5)
    assert (idx == ref).mean() > 0.999


def test_row_overlap_matches_set_intersection():
    rng = np.random.default_rng(1)
    a = np.array([rng.choice(50, 6, replace=False) for _ in range(300)])
    b = np.array([rng.choice(50, 8, replace=False) for _ in range(300)])
    ref = np.array([len(set(r) & set(s[:6])) / 6 for r, s in zip(a, b)])
    np.testing.assert_allclose(row_overlap(a, b, block=64), ref)


# ----------------------------------------------------------------------------- measurements


def test_hubness_skewness_matches_scipy_and_counts_sum_to_nk():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    x = _clustered(rng)
    knn, _ = exact_knn(x, 10)
    h = hubness(x, 10, knn=knn)
    nk = np.bincount(knn.ravel(), minlength=len(x))
    assert nk.sum() == len(x) * 10
    assert h["skewness"] == pytest.approx(scipy_stats.skew(nk, bias=True), abs=1e-9)
    assert h["antihub_ratio"] == pytest.approx((nk == 0).mean())
    assert h["max_k_occurrence"] == nk.max()


def test_anisotropy_known_answers():
    rng = np.random.default_rng(0)
    iso = rng.standard_normal((20000, 64)).astype(np.float32)
    a = anisotropy(iso)
    assert abs(a["mean_random_cosine"]) < 0.01
    assert a["effective_rank"] == pytest.approx(64, rel=0.05)
    # every vector in one direction: random cosine 1, nothing left after centering
    same = np.tile(np.array([1.0, 2.0, 3.0, 4.0], np.float32), (500, 1))
    s = anisotropy(same)
    assert s["mean_random_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert s["effective_rank"] == 0.0


def test_effective_rank_of_rank_r_data_with_equal_variances_is_r():
    rng = np.random.default_rng(0)
    r, d = 5, 64
    w, _ = np.linalg.qr(rng.standard_normal((d, r)))  # r orthonormal directions
    x = (rng.standard_normal((20000, r)) @ w.T).astype(np.float32)
    a = anisotropy(x)
    # participation ratio 1 / sum(p_i^2) with equal p_i = 1/r is exactly r; sampling noise
    # and the normalization before the SVD leave a few percent
    assert a["effective_rank"] == pytest.approx(r, rel=0.05)
    assert a["effective_rank_ratio"] == pytest.approx(r / d, rel=0.05)


def test_ann_recall_hand_computed():
    exact = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    ann = np.array([[1, 2, 9], [-1, 0, 2], [3, 3, 3], [5, 6, 7]])  # 2/3, 2/3, 1/3, 0
    r = ann_recall(exact, ann, floor=0.5)
    assert r["recall_at_k"] == pytest.approx((2 / 3 + 2 / 3 + 1 / 3 + 0) / 4)
    assert r["min_recall"] == 0.0
    assert r["queries_below_floor_ratio"] == pytest.approx(0.5)


def test_collapse_ratios_hand_computed():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((100, 8)).astype(np.float32)
    x = np.concatenate([base, base[:10], base[:5]])  # rows 0-4 appear 3x, 5-9 twice
    c = collapse(x)
    assert c["n"] == 115
    assert c["exact_duplicate_ratio"] == pytest.approx(1 - 100 / 115)
    assert c["near_duplicate_ratio"] == pytest.approx(25 / 115)  # 5*3 + 5*2 rows have a copy
    assert c["largest_exact_duplicate_groups"] == [3, 3, 3, 3, 3]
    assert c["near_duplicate_groups"] == 10


def test_distance_concentration_formula():
    rng = np.random.default_rng(0)
    x = _clustered(rng, n=800)
    _, d = exact_knn(x, 1)
    q = normalize(x)
    mean_cos = float((q @ q.T)[~np.eye(len(q), dtype=bool)].mean())
    dc = distance_concentration(d[:, 0], mean_cos, dim=32, x=x)
    assert dc["relative_contrast"] == pytest.approx((1 - mean_cos) / np.median(d[:, 0]), rel=1e-6)
    assert dc["contrast_over_noise"] == pytest.approx(
        dc["relative_contrast"] / dc["noise_relative_contrast"], rel=1e-6)  # n <= baseline rows


# ----------------------------------------------------------------------------- invariances


def test_audit_is_invariant_to_row_order():
    rng = np.random.default_rng(0)
    x = _clustered(rng)
    perm = rng.permutation(len(x))
    a, b = _values(audit(x, k=10)), _values(audit(x[perm], k=10))
    for name in a:
        assert a[name] == pytest.approx(b[name], rel=1e-6, abs=1e-9), name


def test_audit_is_invariant_to_rotation_and_uniform_scale():
    rng = np.random.default_rng(0)
    x = _clustered(rng)
    q, _ = np.linalg.qr(rng.standard_normal((32, 32)))
    rotated = (x.astype(np.float64) @ q).astype(np.float32)
    a, b, c = _values(audit(x, k=10)), _values(audit(rotated, k=10)), _values(audit(x * 7.5, k=10))
    for scale in (1e-7, 1e7):  # far from unit size: the duplicate test must scale with the data
        for name, v in _values(audit(x * scale, k=10)).items():
            assert a[name] == pytest.approx(v, rel=1e-4, abs=1e-6), f"scale {scale} {name}"
    for name in a:
        # float32 rounding can flip a handful of near-tied neighbors; hubness is the most
        # sensitive to that, everything else must agree closely. rogue_dimensions is a
        # property of the coordinate system and is expected to change under rotation.
        tol = 0.05 if name in {"hubness", "antihubs"} else 2e-3
        # ratios of counted rows move by 1/n (or one island) when a single row crosses a cutoff
        abs_tol = 5e-3 if name in {"outliers", "connectivity", "collapse"} else tol * 1e-2
        if name != "rogue_dimensions":
            assert a[name] == pytest.approx(b[name], rel=tol, abs=abs_tol), f"rotation {name}"
        assert a[name] == pytest.approx(c[name], rel=1e-5, abs=1e-7), f"scale {name}"


def test_audit_is_deterministic_across_calls():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((12000, 16)).astype(np.float32)  # sampled path too
    assert audit(x, k=5, sample=4000, clusters=3).to_json() == audit(x, k=5, sample=4000, clusters=3).to_json()
