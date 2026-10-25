"""Load embeddings from common on-disk formats."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _first_vector_key(row: dict) -> str:
    for key, value in row.items():
        if isinstance(value, list):
            return key
    raise ValueError("no list-valued field found in JSON rows")


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
        return np.load(p)
    if suf == ".npz":
        z = np.load(p)
        return z[z.files[0]]
    if suf in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:  # pragma: no cover
            raise ImportError("reading parquet needs pyarrow: pip install pyarrow") from e
        t = pq.read_table(p)
        cols = [c for c in t.column_names if "emb" in c.lower() or "vec" in c.lower()]
        if not cols:
            raise ValueError(f"no column containing 'emb' or 'vec' in {t.column_names}")
        return np.stack(t.column(cols[0]).to_pylist()).astype(np.float32)
    if suf == ".jsonl":
        text = p.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        key = _first_vector_key(rows[0])
        return np.asarray([r[key] for r in rows], dtype=np.float32)
    if suf == ".json":
        rows = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("expected a non-empty JSON array")
        if isinstance(rows[0], dict):
            key = _first_vector_key(rows[0])
            rows = [r[key] for r in rows]
        return np.asarray(rows, dtype=np.float32)
    if suf in {".csv", ".tsv", ".txt"}:
        return np.loadtxt(p, delimiter="," if suf == ".csv" else None, dtype=np.float32)
    raise ValueError(f"unsupported embedding file: {p} (use .npy/.npz/.parquet/.jsonl/.json/.csv)")
