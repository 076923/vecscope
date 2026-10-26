"""Load embeddings from common on-disk formats."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _is_numeric_list(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )


def _vector_key(row: dict) -> str:
    """The first field holding a non-empty list of numbers (ids, tags and text are skipped)."""
    for key, value in row.items():
        if _is_numeric_list(value):
            return key
    raise ValueError(
        f"no numeric list field found in JSON row with keys {list(row)}; "
        "expected something like {\"embedding\": [0.1, -0.3, ...]}"
    )


def _rows_to_array(rows: list, what: str) -> np.ndarray:
    if not rows:
        raise ValueError(f"{what} is empty")
    if isinstance(rows[0], dict):
        key = _vector_key(rows[0])
        vectors = []
        for i, r in enumerate(rows):
            if not isinstance(r, dict) or not _is_numeric_list(r.get(key)):
                raise ValueError(f"{what}: row {i} has no numeric list under {key!r}")
            vectors.append(r[key])
    else:
        vectors = rows
    widths = {len(v) for v in vectors}
    if len(widths) != 1:
        raise ValueError(f"{what}: vectors have different lengths {sorted(widths)}")
    return np.asarray(vectors, dtype=np.float32)


def _as_matrix(arr: np.ndarray, p: Path) -> np.ndarray:
    if arr.dtype == object or np.iscomplexobj(arr):
        raise ValueError(f"{p}: embeddings must be real numbers, got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(
            f"{p}: expected a 2-D array with one vector per row, got shape {arr.shape}"
        )
    return arr


def load_vectors(path: str | Path) -> np.ndarray:
    """Return a 2-D float array with one vector per row.

    Supported: .npy, .npz (first array), .parquet/.pq (column containing "emb" or "vec"),
    .jsonl (one object per line), .json (array of objects or array of lists), .csv/.tsv/.txt.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"embedding file not found: {p}")
    suf = p.suffix.lower()
    if suf == ".npy":
        return _as_matrix(np.load(p), p)
    if suf == ".npz":
        z = np.load(p)
        if not z.files:
            raise ValueError(f"{p}: the .npz archive is empty")
        return _as_matrix(z[z.files[0]], p)
    if suf in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:  # pragma: no cover
            raise ImportError("reading parquet needs pyarrow: pip install pyarrow") from e
        import pyarrow.types as pt

        t = pq.read_table(p)

        def is_vector(field) -> bool:
            typ = field.type
            return (pt.is_list(typ) or pt.is_large_list(typ) or pt.is_fixed_size_list(typ)) and (
                pt.is_floating(typ.value_type) or pt.is_integer(typ.value_type)
            )

        vector_cols = [f.name for f in t.schema if is_vector(f)]
        preferred = [c for c in vector_cols if "emb" in c.lower() or "vec" in c.lower()]
        cols = preferred or vector_cols
        if not cols:
            raise ValueError(
                f"{p}: no list-of-numbers column found (columns: {t.column_names}); "
                "store vectors as a list<float> column, ideally named 'embedding'"
            )
        return _rows_to_array(t.column(cols[0]).to_pylist(), f"{p} column {cols[0]!r}")
    if suf == ".jsonl":
        text = p.read_text(encoding="utf-8-sig")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return _rows_to_array(rows, str(p))
    if suf == ".json":
        rows = json.loads(p.read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list):
            raise ValueError(f"{p}: expected a JSON array of objects or of lists")
        return _rows_to_array(rows, str(p))
    if suf in {".csv", ".tsv", ".txt"}:
        try:
            arr = np.loadtxt(
                p, delimiter="," if suf == ".csv" else None, dtype=np.float32,
                encoding="utf-8-sig", ndmin=2,
            )
        except ValueError as e:
            raise ValueError(
                f"{p}: could not parse as numbers ({e}). Text formats take one vector per "
                "line, numbers only, no header or id column."
            ) from e
        return _as_matrix(arr, p)
    raise ValueError(f"unsupported embedding file: {p} (use .npy/.npz/.parquet/.jsonl/.json/.csv)")


def load_neighbors(path: str | Path) -> np.ndarray:
    """Load an (n, k) integer array of neighbor ids without going through float32.

    Text and JSON files are parsed as integers directly, so ids above 2**24 survive.
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".csv", ".tsv", ".txt"}:
        arr = np.loadtxt(p, delimiter="," if suf == ".csv" else None, dtype=np.int64,
                         encoding="utf-8-sig", ndmin=2)
    elif suf in {".jsonl", ".json"}:
        arr = load_vectors(p)  # validates structure; values re-read as ints below
        text = p.read_text(encoding="utf-8-sig")
        rows = json.loads(text) if suf == ".json" else [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        if rows and isinstance(rows[0], dict):
            key = _vector_key(rows[0])
            rows = [r[key] for r in rows]
        arr = np.asarray(rows, dtype=np.int64)
    else:
        arr = load_vectors(p)
        if not np.issubdtype(arr.dtype, np.integer):
            if not np.all(np.equal(np.mod(arr, 1), 0)):
                raise ValueError(f"{p}: neighbor ids must be integers")
            arr = arr.astype(np.int64)
    if arr.ndim != 2:
        raise ValueError(f"{p}: neighbor ids must be shaped (n, k), got {arr.shape}")
    return arr.astype(np.int64)
