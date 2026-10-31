"""Load embeddings from common on-disk formats."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np


class _File:
    """A file to parse, read from disk or from bytes already read by the caller (the MCP
    server reads a file once through a verified handle and passes the bytes, so a path
    swapped after the check cannot be what gets parsed)."""

    def __init__(self, path: str | Path, data: bytes | None, kind: str):
        self.path = Path(path)
        self.data = data
        if data is None and not self.path.exists():
            raise FileNotFoundError(f"{kind} file not found: {self.path}")

    def __str__(self) -> str:
        return str(self.path)

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    def binary(self):
        return io.BytesIO(self.data) if self.data is not None else self.path

    def text(self) -> str:
        if self.data is None:
            return self.path.read_text(encoding="utf-8-sig")
        try:
            return self.data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError(f"{self.path}: not a UTF-8 text file") from None


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
    if not all(isinstance(v, list) for v in vectors):
        raise ValueError(f"{what}: each row must be a list of numbers (or an object holding one)")
    if not all(_is_numeric_list(v) for v in vectors):
        raise ValueError(f"{what}: rows must hold numbers only (no booleans, strings or nulls)")
    widths = {len(v) for v in vectors}
    if len(widths) != 1:
        raise ValueError(f"{what}: vectors have different lengths {sorted(widths)}")
    if widths == {0}:
        raise ValueError(f"{what}: vectors are empty lists")
    try:
        arr = np.asarray(vectors, dtype=np.float64)  # float64: huge values are named, not Inf
    except (TypeError, ValueError) as e:
        raise ValueError(f"{what}: vectors must be flat lists of numbers ({e})") from None
    if arr.ndim != 2:
        raise ValueError(f"{what}: vectors must be flat lists of numbers, got shape {arr.shape}")
    return arr


def _np_load(p: _File):
    import zipfile

    try:
        loaded = np.load(p.binary())
    except (EOFError, zipfile.BadZipFile, OSError, ValueError) as e:
        raise ValueError(f"{p}: not a readable {p.suffix} file ({e})") from None
    # np.load picks the format from the file's bytes, whatever the suffix says
    want_archive = p.suffix == ".npz"
    if want_archive != isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError(f"{p}: the file is {'an archive' if not want_archive else 'a single array'}"
                         f", not what its {p.suffix} suffix says; rename it")
    return loaded


def _parquet_rows(p: _File, pq, pt) -> tuple[list, str]:
    """Rows of the first list-of-numbers column (preferring names with emb/vec) as Python lists,
    so integer columns keep their exact values."""
    t = pq.read_table(p.binary())

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
    return t.column(cols[0]).to_pylist(), f"{p} column {cols[0]!r}"


def _as_matrix(arr: np.ndarray, p) -> np.ndarray:
    if arr.dtype.kind not in "iufb":
        raise ValueError(f"{p}: embeddings must be real numbers, got dtype {arr.dtype}")
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError(
            f"{p}: expected a 2-D array with one non-empty vector per row, got shape {arr.shape}"
        )
    return arr


def load_vectors(path: str | Path, data: bytes | None = None) -> np.ndarray:
    """Return a 2-D float array with one vector per row.

    Supported: .npy, .npz (first array), .parquet/.pq (column containing "emb" or "vec"),
    .jsonl (one object per line), .json (array of objects or array of lists), .csv/.tsv/.txt.
    ``data`` is the file's content when the caller has already read it; ``path`` then only
    names the file and picks the format.
    """
    p = _File(path, data, "embedding")
    suf = p.suffix
    if suf == ".npy":
        return _as_matrix(_np_load(p), p)
    if suf == ".npz":
        z = _np_load(p)
        if not z.files:
            raise ValueError(f"{p}: the .npz archive is empty")
        return _as_matrix(z[z.files[0]], p)
    if suf in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:  # pragma: no cover
            raise ImportError("reading parquet needs pyarrow: pip install pyarrow") from e
        import pyarrow.types as pt

        return _rows_to_array(*_parquet_rows(p, pq, pt))
    if suf == ".jsonl":
        text = p.text()
        try:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}: not valid JSON lines ({e})") from None
        return _rows_to_array(rows, str(p))
    if suf == ".json":
        try:
            rows = json.loads(p.text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}: not valid JSON ({e})") from None
        if not isinstance(rows, list):
            raise ValueError(f"{p}: expected a JSON array of objects or of lists")
        return _rows_to_array(rows, str(p))
    if suf in {".csv", ".tsv", ".txt"}:
        text = p.text()
        if not text.strip():
            raise ValueError(f"{p} is empty")
        try:
            arr = np.loadtxt(
                io.StringIO(text), delimiter="," if suf == ".csv" else None, dtype=np.float64,
                ndmin=2,
            )
        except ValueError as e:
            raise ValueError(
                f"{p}: could not parse as numbers ({e}). Text formats take one vector per "
                "line, numbers only, no header or id column."
            ) from e
        return _as_matrix(arr, p)
    raise ValueError(f"unsupported embedding file: {p} (use .npy/.npz/.parquet/.jsonl/.json/.csv)")


def _whole_numbers(arr: np.ndarray) -> bool:
    """Finite whole numbers that fit an int64 (2**53 keeps them exact in float64)."""
    return bool(np.isfinite(arr).all() and (np.abs(arr) < 2**53).all()
                and np.all(np.equal(np.mod(arr, 1), 0)))


def load_neighbors(path: str | Path, data: bytes | None = None) -> np.ndarray:
    """Load an (n, k) integer array of neighbor ids without going through float32.

    Text, JSON and parquet files are parsed as float64 and required to be whole numbers, so
    ids above 2**24 (the float32 limit) survive; `12.0` is accepted, `12.5` is not. ``data``
    is the file's content when the caller has already read it.
    """
    p = _File(path, data, "neighbor")
    suf = p.suffix
    if suf in {".csv", ".tsv", ".txt"}:
        text = p.text()
        if not text.strip():
            raise ValueError(f"{p} is empty")
        delimiter = "," if suf == ".csv" else None
        try:  # float64 holds every id below 2**53 exactly; "12.0" is accepted, "nan" is not
            arr = np.loadtxt(io.StringIO(text), delimiter=delimiter, dtype=np.float64, ndmin=2)
        except ValueError as e:
            raise ValueError(f"{p}: could not parse neighbor ids ({e})") from None
        if not _whole_numbers(arr):
            raise ValueError(f"{p}: neighbor ids must be integers")
    elif suf in {".jsonl", ".json", ".parquet", ".pq"}:
        load_vectors(p.path, p.data)  # validates structure; values re-read below without float32
        if suf in {".parquet", ".pq"}:
            import pyarrow.parquet as pq
            import pyarrow.types as pt

            rows, _ = _parquet_rows(p, pq, pt)
        else:
            text = p.text()
            rows = (json.loads(text) if suf == ".json"
                    else [json.loads(ln) for ln in text.splitlines() if ln.strip()])
        if rows and isinstance(rows[0], dict):
            key = _vector_key(rows[0])
            rows = [r[key] for r in rows]
        try:
            arr = np.asarray(rows, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as e:
            raise ValueError(f"{p}: neighbor ids must be integers ({e})") from None
        if arr.ndim != 2 or not _whole_numbers(arr):
            raise ValueError(f"{p}: neighbor ids must be integers shaped (n, k)")
    else:
        arr = load_vectors(p.path, p.data)
        if arr.dtype.kind == "b":
            raise ValueError(f"{p}: neighbor ids must be integers, got a boolean array")
        if not np.issubdtype(arr.dtype, np.integer):
            if not _whole_numbers(arr):
                raise ValueError(f"{p}: neighbor ids must be integers")
        elif arr.size and arr.dtype.kind == "u" and arr.max() >= 2**63:
            raise ValueError(f"{p}: neighbor id {int(arr.max())} is beyond the int64 range")
        arr = arr.astype(np.int64)
    if arr.ndim != 2:
        raise ValueError(f"{p}: neighbor ids must be shaped (n, k), got {arr.shape}")
    return arr.astype(np.int64)
