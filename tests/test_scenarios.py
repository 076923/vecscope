"""Scenario sweep: each synthetic defect must trip its own finding, healthy data must not fail,
and degenerate inputs must not crash. Expected verdicts list every acceptable outcome; the
tests document which secondary findings are allowed to react (e.g. anisotropy also raises
hubness, which matches the literature)."""

from __future__ import annotations

import numpy as np
import pytest

from vecscope import audit

N, D = 3000, 256


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def structured(rng, n=N, d=D, centers=60, spread=0.8):
    """Clustered base with real neighbor signal; pure noise has none and would fail
    distance_concentration on its own."""
    c = rng.standard_normal((centers, d))
    return c[rng.integers(0, centers, n)] + rng.standard_normal((n, d)) * spread


def build(name: str, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, set[str]]]:
    g = rng.standard_normal
    if name == "pure_noise":  # no neighbor structure at all: only distance_concentration may fail
        return unit(g((N, D))), {"distance_concentration": {"fail"}}
    if name == "healthy_lowdim":
        c = g((30, 64))
        return unit(c[rng.integers(0, 30, N)] + g((N, 64)) * 0.6), {"hubness": {"ok", "warn"}, "antihubs": {"ok", "warn"}}
    if name == "healthy_clustered":
        c = g((20, D))
        return unit(c[rng.integers(0, 20, N)] + g((N, D)) * 0.5), {"hubness": {"ok", "warn"}, "antihubs": {"ok", "warn"}}
    if name == "offset_anisotropic":
        return unit(structured(rng) * 0.3 + 2.0), {"anisotropy": {"fail"}, "hubness": {"ok", "warn", "fail"}, "antihubs": {"ok", "warn", "fail"}}
    if name == "low_rank":
        return unit(g((N, 8)) @ g((8, D))), {"effective_rank": {"fail"}}
    if name == "rogue_dim":
        return unit(np.c_[g((N, D - 1)), g((N, 1)) * 15]), {"rogue_dimensions": {"fail"}, "hubness": {"ok", "warn", "fail"}, "effective_rank": {"ok", "warn", "fail"}}
    if name == "duplicates":
        base = unit(structured(rng, int(N * 0.7)))
        return np.concatenate([base, base[: N - len(base)]]), {"collapse": {"fail"}}
    if name == "near_dup_clusters":
        c = g((150, D))
        return unit(c[rng.integers(0, 150, N)] + g((N, D)) * 0.02), {"collapse": {"fail"}, "connectivity": {"ok", "warn", "fail"}}
    if name == "norm_spread":
        return (structured(rng) * rng.uniform(0.1, 10, (N, 1))).astype(np.float32), {"norms": {"fail"}, "metric_mismatch": {"warn", "fail"}}
    if name == "zero_vectors":
        return np.concatenate([unit(structured(rng, N - 100)), np.zeros((100, D), np.float32)]), {"norms": {"warn", "fail"}, "hubness": {"ok", "warn", "fail"}, "antihubs": {"ok", "warn"}, "outliers": {"ok", "warn", "fail"}}
    if name == "junk_outliers":
        c = g((30, D))
        x = np.concatenate([c[rng.integers(0, 30, N - 120)] + g((N - 120, D)) * 0.15, g((120, D))])
        return unit(x), {"outliers": {"warn", "fail"}, "hubness": {"ok", "warn"}, "antihubs": {"ok", "warn"}}
    if name == "two_big_islands":
        a, b = g((N // 2, 64)), g((N // 2, 64))
        x = np.concatenate([np.c_[a, np.zeros((N // 2, 64))], np.c_[np.zeros((N // 2, 64)), b]])
        return unit(x), {"connectivity": {"ok"}}
    if name == "tiny_islands":
        return unit(np.repeat(g((250, D)), 12, axis=0) + g((3000, D)) * 0.3), {"connectivity": {"fail"}}
    raise KeyError(name)


SCENARIOS = [
    "pure_noise", "healthy_lowdim", "healthy_clustered", "offset_anisotropic", "low_rank",
    "rogue_dim", "duplicates", "near_dup_clusters", "norm_spread", "zero_vectors",
    "junk_outliers", "two_big_islands", "tiny_islands",
]


@pytest.mark.parametrize("name", SCENARIOS)
def test_scenario_trips_expected_findings(name):
    x, expect = build(name, np.random.default_rng(42))
    ev = audit(x, k=10)
    by = {f.name: f.verdict for f in ev.findings}
    for finding, allowed in expect.items():
        assert by[finding] in allowed, f"{name}: {finding}={by[finding]}, allowed {allowed}"
    unexpected = {k: v for k, v in by.items() if v == "fail" and k not in expect}
    assert not unexpected, f"{name}: unexpected FAIL {unexpected}"
    if not expect:
        assert ev.verdict == "ok", {k: v for k, v in by.items() if v != "ok"}


EDGE = {
    "dim_1": lambda rng: rng.standard_normal((50, 1)),
    "dim_2": lambda rng: rng.standard_normal((500, 2)),
    "float64": lambda rng: rng.standard_normal((300, 16)),
    "int": lambda rng: rng.integers(-5, 5, (300, 16)),
    "all_identical": lambda rng: np.ones((200, 16), np.float32),
    "all_zero": lambda rng: np.zeros((200, 16), np.float32),
    "one_zero_row": lambda rng: np.concatenate([unit(rng.standard_normal((199, 16))), np.zeros((1, 16), np.float32)]),
    "huge_values": lambda rng: rng.standard_normal((300, 16)) * 1e6,
    "tiny_values": lambda rng: rng.standard_normal((300, 16)) * 1e-6,
    "n_lt_dim": lambda rng: unit(rng.standard_normal((50, 512))),
    "list_input": lambda rng: rng.standard_normal((20, 4)).tolist(),
}


@pytest.mark.parametrize("name", sorted(EDGE))
def test_edge_inputs_do_not_crash(name):
    ev = audit(EDGE[name](np.random.default_rng(0)), k=5)
    assert all(np.isfinite(f.value) for f in ev.findings)
    ev.to_json()


@pytest.mark.parametrize(
    "bad", [np.zeros((2, 4)), np.zeros(10), np.full((10, 4), np.nan), np.full((10, 4), np.inf)]
)
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        audit(bad, k=3)


def test_same_input_same_json():
    x = unit(np.random.default_rng(1).standard_normal((1500, 64)))
    assert audit(x, k=10).to_json() == audit(x, k=10).to_json()
