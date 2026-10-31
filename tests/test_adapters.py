"""Vector store adapters: read vectors in id order from a faiss index, Qdrant, Chroma,
Milvus Lite (and Weaviate when opted in), query the store's own index for neighbors, and run
the audit on them. Each backend is skipped when its client is not installed."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from vecscope import audit
from vecscope._knn import exact_knn
from vecscope.adapters import annotate, is_source, load_source
from vecscope.cli import main


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    c = rng.standard_normal((12, 16))
    x = c[rng.integers(0, 12, 300)] + rng.standard_normal((300, 16)) * 0.5
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    ids = list(range(1000, 1300))[::-1]  # inserted in descending id order
    return x, ids


def _check(src, x, ids):
    order = np.argsort(ids)  # the source returns rows by ascending id
    assert src.ids == sorted(ids)
    np.testing.assert_allclose(src.vectors, x[order], atol=1e-6)
    assert src.info["n"] == 300 and src.info["dim"] == 16 and len(src.info["fingerprint"]) == 64
    exact, _ = exact_knn(src.vectors, 5)
    got = src.neighbors(5)
    assert got.shape == (300, 5) and (got != np.arange(300)[:, None]).all()  # self removed
    overlap = np.mean([len(set(a) & set(b)) / 5 for a, b in zip(got, exact)])
    assert overlap > 0.95  # each store's own (approximate or flat) index
    return got


def test_is_source():
    assert is_source("qdrant://h:6333/c") and is_source("x.faiss") and is_source("a/b.index")
    assert is_source("milvus+local:///tmp/m.db?collection=c")
    assert not is_source("x.npy") and not is_source("s3://bucket/x.npy")


def test_windows_paths_are_paths_not_schemes():
    from vecscope.adapters import local_path, scheme_of

    win = "C:\\Users\\me\\index\\flat.faiss"
    assert scheme_of(win) == "" and is_source(win)
    assert scheme_of("D:/data/x.npy") == "" and not is_source("D:/data/x.npy")
    assert local_path("qdrant+local://C:/data/qd?collection=docs") == "C:/data/qd"
    assert local_path("qdrant+local:///data/qd?collection=docs") == "/data/qd"
    assert local_path("faiss:///tmp/h.index") == "/tmp/h.index"


def test_faiss_index_files(tmp_path, data):
    faiss = pytest.importorskip("faiss")
    x, ids = data
    flat = faiss.IndexIDMap(faiss.IndexFlatIP(16))
    flat.add_with_ids(x, np.array(ids, dtype=np.int64))
    faiss.write_index(flat, str(tmp_path / "flat.faiss"))
    got = _check(load_source(str(tmp_path / "flat.faiss")), x, ids)
    exact, _ = exact_knn(x[np.argsort(ids)], 5)
    assert (got == exact).mean() > 0.99  # a flat index is exact
    hnsw = faiss.IndexHNSWFlat(16, 16, faiss.METRIC_INNER_PRODUCT)
    hnsw.add(x)
    faiss.write_index(hnsw, str(tmp_path / "h.index"))
    src = load_source(f"faiss://{tmp_path / 'h.index'}")
    assert src.ids == list(range(300)) and src.info["index_type"] == "IndexHNSWFlat"
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(16), 16, 4, faiss.METRIC_INNER_PRODUCT)
    ivf.train(x)
    ivf.add(x)
    faiss.write_index(ivf, str(tmp_path / "ivf.faiss"))
    np.testing.assert_allclose(load_source(str(tmp_path / "ivf.faiss")).vectors, x)


def test_qdrant_local(tmp_path, data):
    qc = pytest.importorskip("qdrant_client")
    from qdrant_client import models

    x, ids = data
    client = qc.QdrantClient(path=str(tmp_path / "qd"))
    client.create_collection("docs", vectors_config=models.VectorParams(
        size=16, distance=models.Distance.COSINE))
    client.upsert("docs", [models.PointStruct(id=i, vector=x[j].tolist())
                           for j, i in enumerate(ids)])
    client.close()
    src = load_source(f"qdrant+local://{tmp_path / 'qd'}?collection=docs")
    _check(src, x, ids)
    src.close()
    with pytest.raises(ValueError, match="collection"):
        load_source(f"qdrant+local://{tmp_path / 'qd'}")


def test_chroma_local(tmp_path, data):
    chromadb = pytest.importorskip("chromadb")
    x, ids = data
    col = chromadb.PersistentClient(path=str(tmp_path / "ch")).create_collection(
        "docs", metadata={"hnsw:space": "cosine"})
    col.add(ids=[f"d{i}" for i in ids], embeddings=x.tolist())
    src = load_source(f"chroma+local://{tmp_path / 'ch'}?collection=docs")
    str_ids = [f"d{i}" for i in ids]  # string ids sort as strings
    _check(src, x, str_ids)


def test_milvus_lite(tmp_path, data):
    pymilvus = pytest.importorskip("pymilvus")
    pytest.importorskip("milvus_lite")
    x, ids = data
    db = str(tmp_path / "m.db")
    client = pymilvus.MilvusClient(uri=db)
    client.create_collection("docs", dimension=16, metric_type="COSINE")
    client.insert("docs", [{"id": i, "vector": x[j].tolist()} for j, i in enumerate(ids)])
    client.release_collection("docs")
    client.close()
    src = load_source(f"milvus+local://{db}?collection=docs")
    assert src.info["loaded_by_vecscope"] is True and src.info["vector"] == "vector"
    _check(src, x, ids)
    src.close()
    client = pymilvus.MilvusClient(uri=db)
    assert "notload" in str(client.get_load_state("docs")["state"]).lower().replace("_", "")
    client.close()


@pytest.mark.skipif(os.environ.get("VECSCOPE_TEST_WEAVIATE") != "1",
                    reason="embedded Weaviate downloads a server binary; set VECSCOPE_TEST_WEAVIATE=1")
def test_weaviate_embedded(tmp_path, data):
    weaviate = pytest.importorskip("weaviate")
    from weaviate.classes.config import Configure

    x, _ = data
    client = weaviate.connect_to_embedded(persistence_data_path=str(tmp_path / "wv"),
                                          port=8079, grpc_port=50060)
    try:
        client.collections.create("Docs", vector_config=Configure.Vectors.self_provided())
        with client.collections.get("Docs").batch.fixed_size(100) as b:
            for v in x:
                b.add_object(properties={}, vector=v.tolist())
        src = load_source("weaviate://localhost:8079/Docs?grpc_port=50060")
        assert src.vectors.shape == (300, 16) and src.ids == sorted(src.ids)
        assert src.neighbors(5).shape == (300, 5)
        src.close()
    finally:
        client.close()


def test_api_key_comes_from_the_environment_only(monkeypatch):
    from vecscope.adapters import _api_key, _redact

    monkeypatch.setenv("VS_KEY", "s3cret")
    assert _api_key({"api_key_env": "VS_KEY"}) == "s3cret" and _api_key({}) is None
    monkeypatch.delenv("VS_KEY")
    with pytest.raises(ValueError, match="empty"):
        _api_key({"api_key_env": "VS_KEY"})
    assert _redact("qdrant://user:pw@host:6333/c?x=1") == "qdrant://host:6333/c?x=1"


def test_unknown_scheme_and_missing_index():
    with pytest.raises(ValueError, match="unsupported source"):
        load_source("redis://h/x")
    pytest.importorskip("faiss")
    with pytest.raises(FileNotFoundError):
        load_source("/nonexistent/x.faiss")


def test_audit_annotation_maps_rows_to_store_ids(tmp_path, data):
    faiss = pytest.importorskip("faiss")
    x, ids = data
    bad = x.copy()
    bad[:20] = bad[20]  # a duplicate group, so collapse lists rows
    index = faiss.IndexIDMap(faiss.IndexFlatIP(16))
    index.add_with_ids(bad, np.array(ids, dtype=np.int64))
    faiss.write_index(index, str(tmp_path / "f.faiss"))
    src = load_source(str(tmp_path / "f.faiss"))
    ev = audit(src.vectors, k=5, ann_neighbors=src.neighbors(5), sample=0)
    annotate(ev, src)
    assert ev.source["backend"] == "faiss"
    for f in ev.findings:
        if "sample_ids" in f.details:
            assert f.details["sample_source_ids"] == [src.ids[i] for i in f.details["sample_ids"]]
    recall = next(f for f in ev.findings if f.name == "ann_recall")
    assert recall.value == pytest.approx(1.0)  # a flat index returns exact neighbors


def test_cli_reads_a_faiss_index_and_its_neighbors(tmp_path, capsys, data):
    faiss = pytest.importorskip("faiss")
    x, _ = data
    index = faiss.IndexHNSWFlat(16, 4, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = 4  # a deliberately weak setting
    index.add(x)
    faiss.write_index(index, str(tmp_path / "h.faiss"))
    code = main(["audit", str(tmp_path / "h.faiss"), "-k", "10", "--ann-from-source",
                 "--save-ids", str(tmp_path / "ids.json"), "--json", "--fail-on", "none"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["source"]["backend"] == "faiss"
    assert json.loads((tmp_path / "ids.json").read_text()) == list(range(300))
    recall = next(f for f in out["findings"] if f["name"] == "ann_recall")
    assert recall["value"] < 1.0
    assert main(["audit", str(tmp_path / "h.faiss"), "--ann-from-source", "--ann", "x.npy"]) == 2
    np.save(tmp_path / "v.npy", x)
    assert main(["audit", str(tmp_path / "v.npy"), "--ann-from-source"]) == 2


def test_mcp_refuses_network_stores_without_opt_in(monkeypatch, tmp_path, data):
    pytest.importorskip("mcp")
    from vecscope import mcp_server

    monkeypatch.delenv(mcp_server.ALLOW_NETWORK_ENV, raising=False)
    with pytest.raises(Exception, match="allow-network"):
        mcp_server.audit_embeddings("qdrant://localhost:6333/docs")
    faiss = pytest.importorskip("faiss")
    x, _ = data
    inside, outside = tmp_path / "in", tmp_path / "out"
    inside.mkdir()
    outside.mkdir()
    index = faiss.IndexFlatIP(16)
    index.add(x)
    faiss.write_index(index, str(outside / "f.faiss"))
    faiss.write_index(index, str(inside / "f.faiss"))
    monkeypatch.setenv(mcp_server.ALLOWED_DIRS_ENV, str(inside))
    with pytest.raises(Exception, match="outside the directories"):
        mcp_server.audit_embeddings(str(outside / "f.faiss"))
    res = mcp_server.audit_embeddings(str(inside / "f.faiss"), k=5, ann_from_source=True)
    assert res["source"]["backend"] == "faiss" and any(
        f["name"] == "ann_recall" for f in res["findings"])
