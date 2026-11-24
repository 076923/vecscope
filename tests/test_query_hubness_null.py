"""query_hubness against chance: few queries over a large corpus leave most items with zero
appearances, so the k-occurrence skewness is high even when no item is a hub. The finding
reports the skewness a hub-free index would show (details.null_skewness) and gives no verdict
level that chance alone reaches."""

from __future__ import annotations

import numpy as np

from vecscope import audit
from vecscope.evidence import THRESHOLDS, _query_hubness_verdict, queries_needed


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def qh(ev):
    return next(f for f in ev.findings if f.name == "query_hubness")


def test_null_skewness_matches_hub_free_random_data():
    """Isotropic corpus and queries have no hubs; their measured skewness is the null."""
    rng = np.random.default_rng(0)
    for n, nq in ((5000, 300), (40000, 60)):
        corpus = unit(rng.standard_normal((n, 32)))
        queries = unit(rng.standard_normal((nq, 32)))
        f = qh(audit(corpus, k=10, queries=queries, sample=0))
        p = 10 / n
        expected = (1 - 2 * p) / np.sqrt(nq * p * (1 - p))
        assert abs(f.details["null_skewness"] - expected) < 1e-9
        assert abs(f.value - expected) / expected < 0.15


def test_hub_free_index_with_few_queries_is_not_flagged_but_says_not_judged():
    """60 queries over 40k items: chance gives a skewness near 8.2 (above warn_at 4), so the
    old verdict was warn on a corpus without hubs. Now warn is not judged and the finding
    says how many queries would judge it."""
    rng = np.random.default_rng(1)
    corpus = unit(rng.standard_normal((40000, 32)))
    queries = unit(rng.standard_normal((60, 32)))
    f = qh(audit(corpus, k=10, queries=queries, sample=0))
    assert f.value > 4.0
    assert f.verdict == "ok"
    assert f.details["not_judged"] == ["warn"]
    assert "NOT JUDGED at warn" in f.explanation
    assert f.details["queries_needed"] == queries_needed(40000, 10, 2.0)
    assert f.details["queries_needed"] > 60


def test_real_hub_is_still_flagged_when_enough_queries():
    """Generic items near the direction every query shares stay a fail when chance is far
    below fail_at."""
    rng = np.random.default_rng(2)
    base = unit(rng.standard_normal((1, 32)))
    corpus = unit(rng.standard_normal((5000, 32)))
    queries = unit(base + 0.5 * unit(rng.standard_normal((300, 32))))
    corpus[:20] = unit(base + 0.01 * rng.standard_normal((20, 32)))
    f = qh(audit(corpus, k=10, queries=queries, sample=0))
    assert f.details["null_skewness"] < 2.0
    assert f.details["not_judged"] == []
    assert f.verdict == "fail"


def test_levels_are_judged_only_above_chance():
    th = {"queries.query_hubness_skewness": THRESHOLDS["queries.query_hubness_skewness"]}
    warn, fail = th["queries.query_hubness_skewness"]
    assert _query_hubness_verdict(fail + 1, 1.0, th) == ("fail", [])
    assert _query_hubness_verdict(warn + 0.1, 1.0, th) == ("warn", [])
    # chance reaches warn: a warn-level value says nothing, a fail-level one still does
    assert _query_hubness_verdict(warn + 0.1, warn + 0.5, th) == ("ok", ["warn"])
    assert _query_hubness_verdict(fail + 1, warn + 0.5, th) == ("fail", ["warn"])
    # chance reaches fail: nothing is judged
    assert _query_hubness_verdict(fail + 5, fail + 1, th) == ("ok", ["warn", "fail"])


def test_queries_needed_brings_chance_to_target():
    n, k, target = 71277, 10, 2.0
    m = queries_needed(n, k, target)
    p = k / n
    assert (1 - 2 * p) / np.sqrt(m * p * (1 - p)) <= target
    assert (1 - 2 * p) / np.sqrt((m - 1) * p * (1 - p)) > target
