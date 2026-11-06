"""The MCP server's guards: paths outside the allowed directories, the read, row and column
limits, network stores without opt-in, secrets in errors, the store timeout option, and a
MemoryError surfacing as a tool error with advice. All on small matrices."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mcp")

from vecscope import mcp_server


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """An allowed directory with a small corpus inside and a copy outside, every limit off."""
    inside, outside = tmp_path / "in", tmp_path / "out"
    inside.mkdir()
    outside.mkdir()
    x = np.random.default_rng(0).standard_normal((300, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    np.save(inside / "x.npy", x)
    np.save(outside / "x.npy", x)
    monkeypatch.setenv(mcp_server.ALLOWED_DIRS_ENV, str(inside))
    for env in (
        mcp_server.ALLOW_ANY_ENV,
        mcp_server.ALLOW_NETWORK_ENV,
        mcp_server.MAX_READ_ENV,
        mcp_server.MAX_ROWS_ENV,
        mcp_server.MAX_DIM_ENV,
    ):
        monkeypatch.delenv(env, raising=False)
    return inside, outside


def test_paths_outside_the_allowed_directories_are_refused(sandbox):
    inside, outside = sandbox
    with pytest.raises(Exception, match="outside the directories"):
        mcp_server.audit_embeddings(str(outside / "x.npy"))
    with pytest.raises(Exception, match="outside the directories"):
        mcp_server.audit_embeddings(str(inside / "x.npy"), queries_path=str(outside / "x.npy"))
    with pytest.raises(Exception, match="outside the directories"):
        mcp_server.compare_embeddings(str(inside / "x.npy"), str(outside / "x.npy"))
    res = mcp_server.audit_embeddings(str(inside / "x.npy"), k=5, verbose=False)
    assert res["verdict"] in ("ok", "warn", "fail") and res["evidence_scope"] == "full"


def test_a_directory_and_a_missing_file_are_user_errors(sandbox):
    inside, _ = sandbox
    with pytest.raises(Exception, match="cannot be read as a file|not a regular file"):
        mcp_server.audit_embeddings(str(inside))
    with pytest.raises(Exception, match="not found"):
        mcp_server.audit_embeddings(str(inside / "missing.npy"))


def test_read_limit_refuses_a_file_before_parsing_it(sandbox, monkeypatch):
    inside, _ = sandbox
    size_mb = (inside / "x.npy").stat().st_size / 2**20
    monkeypatch.setenv(mcp_server.MAX_READ_ENV, str(size_mb / 2))
    with pytest.raises(Exception, match="over the .* MiB this server reads"):
        mcp_server.audit_embeddings(str(inside / "x.npy"))
    monkeypatch.setenv(mcp_server.MAX_READ_ENV, "0")  # 0 lifts the limit
    assert mcp_server.audit_embeddings(str(inside / "x.npy"), k=5, verbose=False)["n"] == 300
    monkeypatch.setenv(mcp_server.MAX_READ_ENV, "lots")
    with pytest.raises(Exception, match="not a number of MiB"):
        mcp_server.audit_embeddings(str(inside / "x.npy"))


def test_row_and_column_limits(sandbox, monkeypatch):
    inside, _ = sandbox
    monkeypatch.setenv(mcp_server.MAX_ROWS_ENV, "299")
    with pytest.raises(Exception, match="300 rows, over the 299"):
        mcp_server.audit_embeddings(str(inside / "x.npy"))
    monkeypatch.setenv(mcp_server.MAX_ROWS_ENV, "300")
    monkeypatch.setenv(mcp_server.MAX_DIM_ENV, "15")
    with pytest.raises(Exception, match="16 columns, over the 15"):
        mcp_server.audit_embeddings(str(inside / "x.npy"))
    monkeypatch.setenv(mcp_server.MAX_DIM_ENV, "0")
    assert mcp_server.audit_embeddings(str(inside / "x.npy"), k=5, verbose=False)["dim"] == 16
    monkeypatch.setenv(mcp_server.MAX_ROWS_ENV, "many")
    with pytest.raises(Exception, match="not a whole number"):
        mcp_server.audit_embeddings(str(inside / "x.npy"))
    # the queries file is checked by the same limits
    monkeypatch.setenv(mcp_server.MAX_ROWS_ENV, "0")
    monkeypatch.setenv(mcp_server.MAX_DIM_ENV, "16")
    np.save(inside / "q32.npy", np.ones((4, 32), dtype=np.float32))
    with pytest.raises(Exception, match="queries_path has 32 columns"):
        mcp_server.audit_embeddings(str(inside / "x.npy"), queries_path=str(inside / "q32.npy"))


def test_the_server_options_set_the_limits(monkeypatch, tmp_path):
    for env in (mcp_server.MAX_ROWS_ENV, mcp_server.MAX_DIM_ENV, mcp_server.MAX_READ_ENV):
        monkeypatch.setenv(env, "")  # main() writes os.environ; this restores it afterwards
    monkeypatch.setattr(mcp_server.mcp, "run", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    mcp_server.main(["--max-rows", "1000", "--max-dim", "2048", "--max-read-mb", "64"])
    assert mcp_server.max_rows() == 1000
    assert mcp_server.max_dim() == 2048
    assert mcp_server.max_read_bytes() == 64 * 2**20
    with pytest.raises(SystemExit):
        mcp_server.main(["--max-rows", "-1"])


def test_memory_error_reaches_the_agent_with_advice(sandbox, monkeypatch):
    inside, _ = sandbox

    def exhausted(*a, **k):
        raise MemoryError

    monkeypatch.setattr(mcp_server, "audit", exhausted)
    with pytest.raises(Exception, match="MemoryError: .*sample=.*--max-rows") as err:
        mcp_server.audit_embeddings(str(inside / "x.npy"))
    assert "ran out of memory" in str(err.value)


def test_network_stores_need_opt_in_and_errors_carry_no_secrets(sandbox, monkeypatch):
    with pytest.raises(Exception, match="allow-network") as err:
        mcp_server.audit_embeddings("qdrant://alice:pw1234@host:6333/docs?api_key_env=VS_KEY")
    assert "pw1234" not in str(err.value)
    pytest.importorskip("qdrant_client")
    monkeypatch.setenv(mcp_server.ALLOW_NETWORK_ENV, "1")
    monkeypatch.setenv("VS_KEY", "s3cret-key")
    with pytest.raises(Exception) as err:
        mcp_server.audit_embeddings(
            "qdrant://alice:pw1234@127.0.0.1:1/docs?api_key_env=VS_KEY&timeout=1"
        )
    msg = str(err.value)
    assert "127.0.0.1:1/docs" in msg
    for secret in ("pw1234", "s3cret-key"):
        assert secret not in msg


def test_store_timeout_must_be_a_finite_positive_bounded_number():
    from vecscope.adapters import MAX_TIMEOUT, _timeout

    assert _timeout({}) == 60.0
    assert _timeout({"timeout": "2.5"}) == 2.5
    assert _timeout({"timeout": str(MAX_TIMEOUT)}) == MAX_TIMEOUT
    for bad in ("inf", "-inf", "nan", "0", "-1", "x", "", None, [1]):
        with pytest.raises(ValueError, match="timeout"):
            _timeout({"timeout": bad})
    with pytest.raises(ValueError, match="most a request may wait"):
        _timeout({"timeout": "1e9"})


def test_sampled_response_states_its_scope(sandbox):
    inside, _ = sandbox
    res = mcp_server.audit_embeddings(str(inside / "x.npy"), k=5, sample=100, verbose=False)
    assert res["evidence_scope"] == "sampled" and res["sampled"] is True
    assert res["summary"].startswith(f"verdict={res['verdict']} n=100 dim=16 k=5 scope=sampled")
    assert set(res["needs_full_audit"]) == set(res["inconclusive_findings"]) | set(
        res["unstable_findings"]
    )
    if res["needs_full_audit"]:
        assert "needs a full audit (sample=0)" in res["summary"]
    full = mcp_server.audit_embeddings(str(inside / "x.npy"), k=5, sample=0, verbose=False)
    assert full["evidence_scope"] == "full" and full["needs_full_audit"] == []
    assert "scope=full" in full["summary"]
    json.dumps(res)  # the response stays plain JSON
