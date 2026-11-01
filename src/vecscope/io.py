"""Load embeddings from common on-disk formats."""

from __future__ import annotations

import contextlib
import io
import json
import warnings
from pathlib import Path
from typing import BinaryIO

import numpy as np


class _File:
    """A file to parse, read from disk, from bytes already read by the caller, or from a
    binary handle the caller opened (the MCP server opens a file once, verifies the handle and
    passes it, so a path swapped after the check cannot be what gets parsed)."""

    def __init__(self, path: str | Path, data: bytes | BinaryIO | None, kind: str):
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
        if self.data is None:
            return self.path
        if isinstance(self.data, bytes):
            return io.BytesIO(self.data)
        self.data.seek(0)
        return self.data

    def open(self):
        if self.data is None:
            return open(self.path, "rb")
        return contextlib.nullcontext(self.binary())  # the caller's handle stays open

    def text(self) -> str:
        if self.data is None:
            return self.path.read_text(encoding="utf-8-sig")
        raw = self.data if isinstance(self.data, bytes) else self.binary().read()
        try:
            return raw.decode("utf-8-sig")
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


_BLOCK_ROWS = 4096


def _rows_to_array(rows, what: str) -> np.ndarray:
    """Stack rows (lists of numbers, or objects holding one) into a float64 matrix, converting
    a block of rows at a time, so a streamed file never holds every row as Python lists."""
    key = width = None
    blocks, block = [], []
    for i, r in enumerate(rows):
        if i == 0 and isinstance(r, dict):
            key = _vector_key(r)
        if key is not None:
            if not isinstance(r, dict) or not _is_numeric_list(r.get(key)):
                raise ValueError(f"{what}: row {i} has no numeric list under {key!r}")
            r = r[key]
        if not _is_numeric_list(r):
            raise ValueError(
                f"{what}: each row must be a list of numbers (or an object holding one)"
                if not isinstance(r, list) else
                f"{what}: rows must hold numbers only (no booleans, strings, nulls or nested "
                f"lists), row {i} does not")
        if width is None:
            width = len(r)
        elif len(r) != width:
            raise ValueError(f"{what}: vectors have different lengths (row 0 has {width}, "
                             f"row {i} has {len(r)})")
        block.append(r)
        if len(block) == _BLOCK_ROWS:
            blocks.append(_block(block, what))
            block = []
    if block:
        blocks.append(_block(block, what))
    if not blocks:
        raise ValueError(f"{what} is empty")
    return blocks[0] if len(blocks) == 1 else np.concatenate(blocks)


def _block(rows: list, what: str) -> np.ndarray:
    try:
        return np.asarray(rows, dtype=np.float64)  # float64: huge values are named, not Inf
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"{what}: vectors must be flat lists of numbers ({e})") from None


def _jsonl_rows(f, p: _File):
    """Parsed rows of a .jsonl file, read one line at a time."""
    for n, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}: line {n} is not valid JSON ({e})") from None
        except UnicodeDecodeError:
            raise ValueError(f"{p}: not a UTF-8 text file") from None


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


def _parquet_matrix(p: _File) -> tuple[np.ndarray, str]:
    """The first list-of-numbers column (preferring names with emb/vec) as an (n, dim) array of
    the column's own dtype, read alone and taken from Arrow's buffers without Python lists, so
    integer columns keep their exact values."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading parquet needs pyarrow: pip install pyarrow") from e
    pt = pa.types
    try:
        f = pq.ParquetFile(p.binary())
    except (pa.ArrowException, OSError) as e:
        raise ValueError(f"{p}: not a readable parquet file ({e})") from None

    def is_vector(field) -> bool:
        typ = field.type
        return (pt.is_list(typ) or pt.is_large_list(typ) or pt.is_fixed_size_list(typ)) and (
            pt.is_floating(typ.value_type) or pt.is_integer(typ.value_type)
        )

    schema = f.schema_arrow
    vector_cols = [x.name for x in schema if is_vector(x)]
    preferred = [c for c in vector_cols if "emb" in c.lower() or "vec" in c.lower()]
    cols = preferred or vector_cols
    if not cols:
        raise ValueError(
            f"{p}: no list-of-numbers column found (columns: {schema.names}); "
            "store vectors as a list<float> column, ideally named 'embedding'"
        )
    what = f"{p} column {cols[0]!r}"
    n = f.metadata.num_rows
    if n == 0:
        raise ValueError(f"{what} is empty")
    out, at = None, 0
    for batch in f.iter_batches(batch_size=_BLOCK_ROWS * 4, columns=[cols[0]]):
        arr = batch.column(0)
        if arr.null_count:
            raise ValueError(f"{what}: rows must hold numbers only, row {at + _first_null(arr)} "
                             "is null")
        if pt.is_fixed_size_list(arr.type):
            widths = np.array([arr.type.list_size])
        else:
            widths = np.unique(np.diff(arr.offsets.to_numpy()))
        if widths.size != 1 or (out is not None and widths[0] != out.shape[1]):
            seen = [] if out is None else [out.shape[1]]
            raise ValueError(f"{what}: vectors have different lengths "
                             f"{sorted({*seen, *widths.tolist()})[:10]}")
        values = arr.flatten()
        if values.null_count:
            raise ValueError(f"{what}: rows must hold numbers only, a vector holds a null")
        block = values.to_numpy(zero_copy_only=False).reshape(len(arr), int(widths[0]))
        if out is None:
            if block.shape[1] == 0:
                raise ValueError(f"{what}: vectors are empty lists")
            out = np.empty((n, block.shape[1]), block.dtype)
        out[at:at + len(block)] = block
        at += len(block)
    return out, what


def _first_null(arr) -> int:
    return int(np.flatnonzero(arr.is_null().to_numpy(zero_copy_only=False))[0])


def _as_matrix(arr: np.ndarray, p) -> np.ndarray:
    if arr.dtype.kind not in "iufb":
        raise ValueError(f"{p}: embeddings must be real numbers, got dtype {arr.dtype}")
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError(
            f"{p}: expected a 2-D array with one non-empty vector per row, got shape {arr.shape}"
        )
    return arr


def load_vectors(path: str | Path, data: bytes | BinaryIO | None = None) -> np.ndarray:
    """Return a 2-D float array with one vector per row.

    Supported: .npy, .npz (first array), .parquet/.pq (column containing "emb" or "vec"),
    .jsonl (one object per line), .json (array of objects or array of lists), .csv/.tsv/.txt.
    ``data`` is the file's content, or a seekable binary handle, when the caller has already
    opened it; ``path`` then only names the file and picks the format.
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
        arr, what = _parquet_matrix(p)
        return _as_matrix(arr, what)
    if suf == ".jsonl":
        with p.open() as f:
            return _rows_to_array(_jsonl_rows(f, p), str(p))
    if suf == ".json":
        try:
            rows = json.loads(p.text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}: not valid JSON ({e})") from None
        if not isinstance(rows, list):
            raise ValueError(f"{p}: expected a JSON array of objects or of lists")
        return _rows_to_array(rows, str(p))
    if suf in {".csv", ".tsv", ".txt"}:
        try:
            arr = _text_matrix(p)
        except ValueError as e:
            if str(e).endswith(" is empty") or "UTF-8" in str(e):
                raise
            raise ValueError(
                f"{p}: could not parse as numbers ({e}). Text formats take one vector per "
                "line, numbers only, no header or id column."
            ) from e
        return _as_matrix(arr, p)
    raise ValueError(f"unsupported embedding file: {p} (use .npy/.npz/.parquet/.jsonl/.json/.csv)")


def _text_matrix(p: _File) -> np.ndarray:
    """A .csv/.tsv/.txt file parsed as float64, read in chunks of lines rather than as one
    string (float64 holds every id below 2**53 exactly; "12.0" parses, "nan" is caught later)."""
    with p.open() as f:
        text = io.TextIOWrapper(f, encoding="utf-8-sig")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # an empty file is named below
                arr = np.loadtxt(text, delimiter="," if p.suffix == ".csv" else None,
                                 dtype=np.float64, ndmin=2)
        except UnicodeDecodeError:
            raise ValueError(f"{p}: not a UTF-8 text file") from None
        finally:
            text.detach()  # leaves a handle the caller passed open
    if arr.size == 0:
        raise ValueError(f"{p} is empty")
    return arr


def _whole_numbers(arr: np.ndarray) -> bool:
    """Finite whole numbers that fit an int64 (2**53 keeps them exact in float64)."""
    return bool(np.isfinite(arr).all() and (np.abs(arr) < 2**53).all()
                and np.all(np.equal(np.mod(arr, 1), 0)))


def load_neighbors(path: str | Path, data: bytes | BinaryIO | None = None) -> np.ndarray:
    """Load an (n, k) integer array of neighbor ids without going through float32.

    Text and JSON files are parsed as float64 and required to be whole numbers, and parquet
    keeps its column's integer type, so ids above 2**24 (the float32 limit) survive; `12.0` is
    accepted, `12.5` is not. ``data`` is the file's content or a binary handle, as in
    ``load_vectors``.
    """
    p = _File(path, data, "neighbor")
    suf = p.suffix
    if suf in {".csv", ".tsv", ".txt"}:
        try:
            arr = _text_matrix(p)
        except ValueError as e:
            if str(e).endswith(" is empty") or "UTF-8" in str(e):
                raise
            raise ValueError(f"{p}: could not parse neighbor ids ({e})") from None
        if not _whole_numbers(arr):
            raise ValueError(f"{p}: neighbor ids must be integers")
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
