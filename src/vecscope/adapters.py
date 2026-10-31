"""Read vectors straight from a vector index or database, and query its index for neighbors.

Optional: each backend imports its own client only when its URI is used, so the core keeps
depending on NumPy alone. Sources:

- ``path/to/index.faiss`` (or ``.index``, or ``faiss:///path``): a faiss index file
- ``qdrant://host:6333/collection``; ``qdrant+local:///dir?collection=c`` for local mode
- ``chroma://host:8000/collection``; ``chroma+local:///dir?collection=c`` for a persistent dir
- ``milvus://host:19530/collection``; ``milvus+local:///file.db?collection=c`` (Milvus Lite)
- ``weaviate://host:8080/Collection`` (``grpc_port=`` defaults to 50051)

Query options: ``vector=`` names the vector (named vectors in Qdrant/Weaviate, the field in
Milvus), ``https=1`` for TLS, ``api_key_env=NAME`` reads an API key from that environment
variable (keys are never taken from the URI itself), ``pk=`` the Milvus primary key field,
``timeout=`` seconds per request to a network store (default 60).

Secrets stay out of the evidence and of error messages: the recorded URI keeps the host, the
path and the options above, and replaces user info and any other query value with
``REDACTED``; a client error is re-raised with the API key and those values scrubbed.
``max_bytes`` caps the vectors read (as float32), so a store larger than the caller allows
fails while it is read instead of exhausting memory.

Rows are returned in ascending id order, so two reads of unchanged contents give the same
matrix; ``Source.info["fingerprint"]`` lets a baseline tell whether the contents changed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import numpy as np

SCHEMES = ("faiss", "qdrant", "qdrant+local", "chroma", "chroma+local", "milvus",
           "milvus+local", "weaviate")
FAISS_SUFFIXES = (".faiss", ".index")
_BATCH = 1024
DEFAULT_TIMEOUT = 60.0
# query options that are recorded as given; every other value is redacted
_SAFE_OPTS = ("collection", "vector", "https", "pk", "grpc_port", "api_key_env", "timeout")


def scheme_of(path: str) -> str:
    """The URI scheme, or "" for a file path. Only ``scheme://`` counts, so a Windows path
    such as ``C:\\data\\x.faiss`` is a path, not the scheme "c"."""
    s = str(path)
    return urlparse(s).scheme if "://" in s else ""


def local_path(uri: str) -> str:
    """The directory or file of a ``faiss://`` or ``+local`` URI: ``qdrant+local:///data/q``
    and ``qdrant+local://C:/data/q`` both name the path after the ``//``."""
    u = urlparse(str(uri))
    return u.netloc + u.path if u.netloc else u.path


def is_source(path: str) -> bool:
    """Whether ``path`` names a vector store rather than an embedding file."""
    s = str(path)
    return scheme_of(s) in SCHEMES or Path(s).suffix.lower() in FAISS_SUFFIXES


@dataclass
class Source:
    vectors: np.ndarray
    ids: list
    info: dict
    _search: object = field(default=None, repr=False)
    release: object = field(default=None, repr=False)  # close the client / undo a load

    def close(self) -> None:
        """Close the client and undo state changes made to read the store (a Milvus collection
        vecscope loaded); the source cannot be searched afterwards."""
        if self.release is not None:
            self.release()
            self.release = None

    def neighbors(self, k: int, block: int = 256) -> np.ndarray:
        """(n, k) row positions of the k nearest stored items the source's own index returns
        for each stored vector, self removed (-1 where it returned fewer). Queries the index
        with k + 1 so that dropping the item itself still leaves k."""
        if self._search is None:
            raise ValueError(f"{self.info['backend']} source cannot be searched")
        row_of = {i: r for r, i in enumerate(self.ids)}
        out = np.full((len(self.ids), k), -1, dtype=np.int64)
        for s in range(0, len(self.ids), block):
            e = min(s + block, len(self.ids))
            for j, hit_ids in enumerate(self._search(self.vectors[s:e], k + 1)):
                rows = [row_of[h] for h in hit_ids if h in row_of and row_of[h] != s + j][:k]
                out[s + j, :len(rows)] = rows
        return out


def _opts(u) -> dict:
    return {k: v[-1] for k, v in parse_qs(u.query).items()}


def _api_key(o: dict) -> str | None:
    name = o.get("api_key_env")
    if not name:
        return None
    key = os.environ.get(name)
    if not key:
        raise ValueError(f"api_key_env={name} is set but that environment variable is empty")
    return key


def _timeout(o: dict) -> float:
    try:
        t = float(o.get("timeout", DEFAULT_TIMEOUT))
    except ValueError:
        raise ValueError(f"timeout={o['timeout']!r} is not a number of seconds") from None
    if not t > 0:
        raise ValueError(f"timeout must be positive, got {t}")
    return t


class _Rows:
    """Ids and float32 vectors as they are read, refusing more than ``max_bytes`` of vectors
    (a client's list of Python floats is several times larger than the array it becomes)."""

    def __init__(self, max_bytes: int | None, what: str):
        self.ids, self.vectors, self.nbytes = [], [], 0
        self.max_bytes, self.what = max_bytes, what

    def add(self, i, v) -> None:
        v = np.asarray(v, dtype=np.float32).ravel()
        self.nbytes += v.nbytes
        if self.max_bytes is not None and self.nbytes > self.max_bytes:
            raise ValueError(f"{self.what} holds more than {self.max_bytes / 2**20:.0f} MiB of "
                             "vectors, the most this reader may load; audit a smaller collection "
                             "or raise the limit")
        self.ids.append(i)
        self.vectors.append(v)


def _sorted(ids: list, vectors: list) -> tuple[list, np.ndarray]:
    if not ids:
        raise ValueError("the source holds no vectors")
    key = (lambda i: (0, i, "")) if all(isinstance(i, int) for i in ids) else (lambda i: (1, 0, str(i)))
    order = sorted(range(len(ids)), key=lambda r: key(ids[r]))
    ids = [ids[r] for r in order]
    if len(set(ids)) != len(ids):
        raise ValueError("the source returned duplicate ids")
    lengths = {len(vectors[r]) for r in order}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(f"the source returned vectors of uneven or empty length {sorted(lengths)}")
    arr = np.stack([np.asarray(vectors[r], dtype=np.float32) for r in order])
    return ids, arr


def _pick_vector(v, name: str | None, backend: str):
    if isinstance(v, dict):
        if name:
            if name not in v:
                raise ValueError(f"{backend}: no vector named {name!r}; have {sorted(v)}")
            return v[name]
        if len(v) != 1:
            raise ValueError(f"{backend}: points have named vectors {sorted(v)}; pass vector=")
        return next(iter(v.values()))
    return v


def _redact(uri: str) -> str:
    """The URI without user info, and with every query value outside ``_SAFE_OPTS``
    replaced: what is recorded in evidence and shown in errors."""
    if not scheme_of(uri):
        return str(uri)
    u = urlparse(str(uri))
    netloc = u.hostname or ""
    if u.port:
        netloc += f":{u.port}"
    if u.scheme.endswith("+local"):
        netloc = u.netloc.rpartition("@")[2]  # a Windows drive is the "host" here
    query = urlencode([(k, v if k in _SAFE_OPTS else "REDACTED")
                       for k, vs in parse_qs(u.query, keep_blank_values=True).items() for v in vs])
    return u._replace(netloc=netloc, query=query).geturl()


def _secrets(uri: str) -> list[str]:
    """Strings that must not appear in a message about ``uri``."""
    u = urlparse(str(uri))
    out = [u.password or "", u.username or ""]
    o = _opts(u)
    if o.get("api_key_env"):
        out.append(os.environ.get(o["api_key_env"], ""))
    out += [v for k, v in o.items() if k not in _SAFE_OPTS]
    return [x for x in out if len(x) >= 4]


# ----------------------------------------------------------------------------- backends

def _faiss(path: str, data: bytes | None = None, max_bytes: int | None = None) -> Source:
    try:
        import faiss
    except ImportError as e:
        raise ImportError("reading a faiss index needs faiss: pip install 'vecscope[index]'") from e
    p = Path(local_path(path) if path.startswith("faiss://") else path)
    if data is not None:  # already read through a verified handle (MCP server)
        try:
            index = faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8))
        except RuntimeError as e:
            raise ValueError(f"{p}: not a readable faiss index ({e})") from None
    elif not p.exists():
        raise FileNotFoundError(f"faiss index not found: {p}")
    else:
        index = faiss.read_index(str(p))
    inner, ids = index, None
    if isinstance(index, (faiss.IndexIDMap, faiss.IndexIDMap2)) or hasattr(index, "id_map"):
        inner = faiss.downcast_index(index.index)
        ids = faiss.vector_to_array(index.id_map).astype(np.int64).tolist()
    try:
        ivf = faiss.extract_index_ivf(inner)
        ivf.make_direct_map()
    except RuntimeError:
        pass  # not an IVF index
    n = index.ntotal
    if n == 0:
        raise ValueError(f"{p}: the index is empty")
    if max_bytes is not None and n * index.d * 4 > max_bytes:
        raise ValueError(f"{p}: {n} vectors of {index.d} dims exceed the "
                         f"{max_bytes / 2**20:.0f} MiB this reader may load")
    try:
        vectors = inner.reconstruct_n(0, n)
    except RuntimeError as e:
        raise ValueError(f"{p}: this index type cannot return its vectors ({e}); export the "
                         "vectors to .npy instead") from None
    ids = ids if ids is not None else list(range(n))
    ids, arr = _sorted(ids, list(vectors))
    metric = "inner_product" if index.metric_type == faiss.METRIC_INNER_PRODUCT else "l2"

    def search(q, k):
        _, labels = index.search(np.ascontiguousarray(q, dtype=np.float32), k)
        return [[int(i) for i in row if i >= 0] for row in labels]

    return Source(arr, ids, {"backend": "faiss", "uri": str(p), "index_metric": metric,
                             "index_type": type(faiss.downcast_index(inner)).__name__}, search)


def _qdrant(uri: str, max_bytes: int | None = None) -> Source:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as e:
        raise ImportError("reading Qdrant needs qdrant-client: pip install 'vecscope[qdrant]'") from e
    u, o = urlparse(uri), _opts(urlparse(uri))
    if u.scheme == "qdrant+local":
        client, collection = QdrantClient(path=local_path(uri)), o.get("collection")
    else:
        client = QdrantClient(host=u.hostname, port=u.port or 6333, https=o.get("https") == "1",
                              api_key=_api_key(o), timeout=max(1, round(_timeout(o))))
        collection = u.path.strip("/")
    if not collection:
        raise ValueError(f"{_redact(uri)}: name the collection")
    name = o.get("vector")
    rows, offset = _Rows(max_bytes, f"qdrant collection {collection}"), None
    while True:
        points, offset = client.scroll(collection, limit=_BATCH, offset=offset, with_vectors=True,
                                       with_payload=False)
        for pt in points:
            rows.add(pt.id, _pick_vector(pt.vector, name, "qdrant"))
        if offset is None:
            break
    ids, arr = _sorted(rows.ids, rows.vectors)

    def search(q, k):
        reqs = [models.QueryRequest(query=v.tolist(), limit=k, using=name) for v in q]
        return [[p.id for p in r.points] for r in client.query_batch_points(collection, reqs)]

    return Source(arr, ids, {"backend": "qdrant", "uri": _redact(uri), "collection": collection,
                             "vector": name}, search, release=client.close)


def _chroma(uri: str, max_bytes: int | None = None) -> Source:
    try:
        import chromadb
    except ImportError as e:
        raise ImportError("reading Chroma needs chromadb: pip install 'vecscope[chroma]'") from e
    u, o = urlparse(uri), _opts(urlparse(uri))
    if u.scheme == "chroma+local":
        client, collection = chromadb.PersistentClient(path=local_path(uri)), o.get("collection")
    else:
        key = _api_key(o)
        timeout = _timeout(o)
        client = chromadb.HttpClient(host=u.hostname, port=u.port or 8000,
                                     ssl=o.get("https") == "1",
                                     headers={"x-chroma-token": key} if key else None)
        # the Chroma HTTP client takes no timeout; set it on its session for the reads
        session = getattr(getattr(client, "_server", None), "_session", None)
        if session is not None and hasattr(session, "timeout"):
            import httpx

            session.timeout = httpx.Timeout(timeout)
        collection = u.path.strip("/")
    if not collection:
        raise ValueError(f"{_redact(uri)}: name the collection")
    col = client.get_collection(collection)
    rows, offset = _Rows(max_bytes, f"chroma collection {collection}"), 0
    while True:
        got = col.get(include=["embeddings"], limit=_BATCH, offset=offset)
        if not got["ids"]:
            break
        for i, v in zip(got["ids"], got["embeddings"]):
            rows.add(i, v)
        offset += len(got["ids"])
    ids, arr = _sorted(rows.ids, rows.vectors)

    def search(q, k):
        res = col.query(query_embeddings=q.tolist(), n_results=k, include=[])
        return [list(r) for r in res["ids"]]

    return Source(arr, ids, {"backend": "chroma", "uri": _redact(uri), "collection": collection},
                  search)


def _milvus(uri: str, max_bytes: int | None = None) -> Source:
    try:
        from pymilvus import MilvusClient
    except ImportError as e:
        raise ImportError("reading Milvus needs pymilvus: pip install 'vecscope[milvus]'") from e
    u, o = urlparse(uri), _opts(urlparse(uri))
    if u.scheme == "milvus+local":
        client, collection = MilvusClient(uri=local_path(uri)), o.get("collection")
    else:
        scheme = "https" if o.get("https") == "1" else "http"
        client = MilvusClient(uri=f"{scheme}://{u.hostname}:{u.port or 19530}",
                              token=_api_key(o) or "", timeout=_timeout(o))
        collection = u.path.strip("/")
    if not collection:
        raise ValueError(f"{_redact(uri)}: name the collection")
    desc = client.describe_collection(collection)
    fields = {f["name"]: f for f in desc["fields"]}
    pk = o.get("pk") or next(f["name"] for f in desc["fields"] if f.get("is_primary"))
    vec_fields = [n for n, f in fields.items()
                  if "VECTOR" in getattr(f.get("type"), "name", str(f.get("type"))).upper()]
    name = o.get("vector") or (vec_fields[0] if len(vec_fields) == 1 else None)
    if name is None:
        raise ValueError(f"milvus: collection has vector fields {vec_fields}; pass vector=")
    # a released collection must be loaded to be read; load it only then, and release it
    # again afterwards so a production collection's load state is left as it was
    state = str(client.get_load_state(collection).get("state", "")).lower()
    loaded_here = "notload" in state.replace("_", "")
    if loaded_here:
        client.load_collection(collection)
    rows = _Rows(max_bytes, f"milvus collection {collection}")
    it = client.query_iterator(collection, batch_size=_BATCH, output_fields=[pk, name])
    try:
        while True:
            batch = it.next()
            if not batch:
                break
            for r in batch:
                rows.add(r[pk], r[name])
    except ValueError:
        if loaded_here:
            client.release_collection(collection)
        raise
    finally:
        it.close()
    ids, arr = _sorted(rows.ids, rows.vectors)

    def search(q, k):
        res = client.search(collection, data=q.tolist(), limit=k, anns_field=name,
                            output_fields=[])
        return [[h["id"] for h in hits] for hits in res]

    src = Source(arr, ids, {"backend": "milvus", "uri": _redact(uri), "collection": collection,
                            "vector": name, "loaded_by_vecscope": loaded_here}, search)
    def release():
        if loaded_here:
            client.release_collection(collection)
        client.close()

    src.release = release
    return src


def _weaviate(uri: str, max_bytes: int | None = None) -> Source:
    try:
        import weaviate
        from weaviate.classes.init import AdditionalConfig, Auth, Timeout
    except ImportError as e:
        raise ImportError("reading Weaviate needs weaviate-client: pip install "
                          "'vecscope[weaviate]'") from e
    u, o = urlparse(uri), _opts(urlparse(uri))
    secure = o.get("https") == "1"
    key = _api_key(o)
    t = _timeout(o)
    client = weaviate.connect_to_custom(
        http_host=u.hostname, http_port=u.port or 8080, http_secure=secure,
        grpc_host=u.hostname, grpc_port=int(o.get("grpc_port", 50051)), grpc_secure=secure,
        auth_credentials=Auth.api_key(key) if key else None,
        additional_config=AdditionalConfig(timeout=Timeout(init=t, query=t, insert=t)))
    collection = u.path.strip("/")
    if not collection:
        client.close()
        raise ValueError(f"{_redact(uri)}: name the collection")
    col = client.collections.get(collection)
    name = o.get("vector")
    rows = _Rows(max_bytes, f"weaviate collection {collection}")
    try:
        for obj in col.iterator(include_vector=True):
            rows.add(str(obj.uuid), _pick_vector(obj.vector, name, "weaviate"))
        ids, arr = _sorted(rows.ids, rows.vectors)
    except Exception:
        client.close()
        raise

    def search(q, k):
        out = []
        for v in q:
            res = col.query.near_vector(near_vector=v.tolist(), limit=k, target_vector=name)
            out.append([str(obj.uuid) for obj in res.objects])
        return out

    return Source(arr, ids, {"backend": "weaviate", "uri": _redact(uri), "collection": collection,
                             "vector": name}, search, release=client.close)


def load_source(uri: str, data: bytes | None = None, max_bytes: int | None = None) -> Source:
    """Vectors (ascending id order), their ids, and a search function from a vector store.
    ``data`` is a faiss index file's content when the caller has already read it;
    ``max_bytes`` caps the float32 vectors read."""
    from vecscope.incremental import fingerprint

    scheme = scheme_of(uri)
    if scheme == "faiss" or Path(str(uri)).suffix.lower() in FAISS_SUFFIXES and not scheme:
        read, args = _faiss, (str(uri), data, max_bytes)
    elif scheme.startswith("qdrant"):
        read, args = _qdrant, (uri, max_bytes)
    elif scheme.startswith("chroma"):
        read, args = _chroma, (uri, max_bytes)
    elif scheme.startswith("milvus"):
        read, args = _milvus, (uri, max_bytes)
    elif scheme == "weaviate":
        read, args = _weaviate, (uri, max_bytes)
    else:
        raise ValueError(f"unsupported source {_redact(uri)!r}; known schemes: {', '.join(SCHEMES)}")
    try:
        src = read(*args)
    except ImportError:
        raise
    except Exception as e:
        # client errors can carry what the client was given; keep the key and other secrets out
        msg = str(e)
        for secret in _secrets(uri):
            msg = msg.replace(secret, "REDACTED")
        if msg == str(e) and isinstance(e, (ValueError, FileNotFoundError, PermissionError)):
            raise
        raise ValueError(f"{_redact(uri)}: {type(e).__name__}: {msg}") from None
    src.info.update({"n": len(src.ids), "dim": int(src.vectors.shape[1]),
                     "id_order": "ascending id", "fingerprint": fingerprint(src.vectors)})
    return src


def annotate(ev, src: Source) -> None:
    """Record the source on Evidence and translate row ids in details to the store's ids
    (``sample_source_ids`` next to ``sample_ids``), so they can be looked up directly."""
    ev.source = dict(src.info)
    for f in ev.findings:  # query-side ids index the queries, not the store
        if "sample_ids" in f.details:
            f.details["sample_source_ids"] = [src.ids[i] for i in f.details["sample_ids"]]
    for c in ev.clusters:
        c["sample_source_ids"] = [src.ids[i] for i in c["sample_ids"]]
