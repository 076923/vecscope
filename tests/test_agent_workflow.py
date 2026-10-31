"""The agent loop the skill describes, driven over the real MCP stdio protocol: audit, read the
plan, apply its first change, re-audit, and judge the change by the finding and by retrieval
quality on labeled queries."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest


def _attr(obj, *names):  # mcp 1.x uses camelCase fields, 2.x snake_case
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(names)


def test_diagnose_change_reverify_over_stdio(tmp_path):
    pytest.importorskip("mcp")
    import asyncio
    import os
    import sys
    from pathlib import Path

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    import vecscope
    from vecscope.mcp_server import ALLOWED_DIRS_ENV

    rng = np.random.default_rng(2)
    centers = rng.standard_normal((25, 32))
    labels = rng.integers(0, 25, 2000)
    clean = centers[labels] + rng.standard_normal((2000, 32)) * 0.7
    offset = rng.standard_normal(32) * 4.0  # a shared direction on top
    corpus = (clean + offset).astype(np.float32)
    targets = rng.choice(2000, 80, replace=False)
    queries = (clean[targets] + rng.standard_normal((80, 32)) * 0.3 + offset).astype(np.float32)
    data = tmp_path / "data"
    data.mkdir()
    np.save(data / "corpus.npy", corpus)
    np.save(data / "queries.npy", queries)
    (data / "qrels.json").write_text(json.dumps({str(i): {str(int(t)): 1}
                                                 for i, t in enumerate(targets)}))
    src = str(Path(vecscope.__file__).resolve().parents[1])
    env = {**os.environ, ALLOWED_DIRS_ENV: str(data),
           "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    params = StdioServerParameters(command=sys.executable, args=["-m", "vecscope.mcp_server"],
                                   env=env)
    audit_args = {"queries_path": str(data / "queries.npy"), "qrels_path": str(data / "qrels.json"),
                  "k": 10}

    # what the agent does for each plan group it meets here; the same transform goes to the
    # corpus and the queries, fitted on the corpus
    def unit(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def apply(group, c, q):
        if group == "metric":  # "L2-normalize if cosine is intended"
            return unit(c), unit(q)
        if group == "shared_direction":  # "subtract the corpus mean and re-normalize"
            mu = c.mean(axis=0, keepdims=True)
            return unit(c - mu), unit(q - mu)
        raise AssertionError(group)

    async def run():
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            c, q, groups, reports = corpus, queries, [], []
            for i in range(4):
                np.save(data / f"c{i}.npy", c.astype(np.float32))
                np.save(data / f"q{i}.npy", q.astype(np.float32))
                call = {"path": str(data / f"c{i}.npy"), **audit_args,
                        "queries_path": str(data / f"q{i}.npy")}
                if reports:  # judge each change against the audit before it
                    call["baseline"] = reports[-1]["audit_id"]
                res = await s.call_tool("audit_embeddings", call)
                assert not _attr(res, "isError", "is_error"), res.content[0].text
                rep = json.loads(res.content[0].text)
                reports.append(rep)
                if not rep["plan"] or rep["plan"][0]["group"] not in ("metric", "shared_direction"):
                    break  # the injected defects are gone; what remains is the data's own
                step = rep["plan"][0]
                groups.append(step["group"])
                c, q = apply(step["group"], c, q)
            return groups, reports

    groups, reports = asyncio.run(run())
    before, after = reports[0], reports[-1]
    assert groups == ["metric", "shared_direction"]  # normalize first, then center
    left = {s["group"] for s in after["plan"]}
    assert not left & {"metric", "shared_direction"}
    assert all(f["verdict"] == "ok" for f in after["findings"]
               if f["name"] in ("anisotropy", "metric_mismatch", "collapse", "norms"))
    find = {f["name"]: f for f in before["findings"]}
    assert find["anisotropy"]["verdict"] != "ok" and find["metric_mismatch"]["verdict"] != "ok"
    # each step's re-measure line names retrieval; the changes must not cost quality
    assert after["retrieval"]["exact"]["ndcg_at_k"] >= before["retrieval"]["exact"]["ndcg_at_k"] - 1e-9
    assert len({r["retrieval"]["qrels_fingerprint"] for r in reports}) == 1
    # each re-audit, diffed against the one before, shows the step's findings fixed and no
    # loss in retrieval; a finding the step did not target may still move (outliers here)
    for prev, rep in itertools.pairwise(reports):
        d = rep["baseline_diff"]
        targeted = {f["name"] for f in prev["plan"][0]["findings"]}
        verdicts = {f["name"]: f["verdict"] for f in rep["findings"]}
        assert all(n in d["improvements"] or verdicts[n] == "ok" for n in targeted), targeted
        assert not targeted & set(d["regressions"]) and d["retrieval_comparable"]
        assert all(c["delta"] >= -1e-9 for c in d["retrieval_changes"] if c["metric"] == "ndcg_at_k")
