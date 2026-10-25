"""Audit real, publicly available embeddings to calibrate thresholds.

Downloads one parquet shard per dataset from the Hugging Face Hub (pre-computed
embeddings, no model needed) and prints the vecscope findings side by side.

    pip install huggingface_hub pyarrow
    python examples/real_models.py [--rows 40000] [--k 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vecscope import audit

DATASETS = {
    "cohere-v3-multilingual-ko": (
        "Cohere/wikipedia-2023-11-embed-multilingual-v3", "ko/0000.parquet", "emb"
    ),
    "cohere-v3-multilingual-en": (
        "Cohere/wikipedia-2023-11-embed-multilingual-v3", "en/0000.parquet", "emb"
    ),
    "openai-ada-002-dbpedia": (
        "KShivendu/dbpedia-entities-openai-1M",
        "data/train-00000-of-00026-3c7b99d1c7eda36e.parquet", "openai",
    ),
    "openai-3-large-dbpedia": (
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
        "data/train-00000-of-00026.parquet", "text-embedding-3-large-1536-embedding",
    ),
    "all-MiniLM-L6-v2-wiki": (
        "maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2",
        "data/train-00000-of-00145-437d77a320cee9e2.parquet", "emb",
    ),
    "bge-m3-wiki-en": ("Upstash/wikipedia-2024-06-bge-m3", "data/en/000.parquet", "embedding"),
    "mxbai-embed-large-wiki": (
        "anton-l/wiki-embed-mxbai-embed-large-v1", "data/train-00000-of-00235.parquet", "embedding"
    ),
    "gte-base-arxiv": (
        "bluuebunny/arxiv_embeddings_Alibaba-NLP_gte-base-en-v1.5", "2024.parquet", "embedding"
    ),
}


def load(repo: str, file: str, col: str, rows: int) -> np.ndarray:
    path = hf_hub_download(repo, file, repo_type="dataset")
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    if col not in names:
        col = next(c for c in names if "emb" in c.lower() or "vec" in c.lower())
    chunks, total = [], 0
    for i in range(pf.num_row_groups):  # read only the row groups we need
        chunk = pf.read_row_group(i, columns=[col]).column(col).to_numpy(zero_copy_only=False)
        chunks.append(np.stack(chunk)); total += len(chunk)
        if total >= rows:
            break
    return np.concatenate(chunks)[:rows].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=40000)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    results = {}
    for name, (repo, file, col) in DATASETS.items():
        if args.only and name not in args.only:
            continue
        x = load(repo, file, col, args.rows)
        ev = audit(x, k=args.k)
        results[name] = ev
        print(f"\n== {name}  n={ev.n} dim={ev.dim}  verdict={ev.verdict.upper()}")
        for f in ev.findings:
            print(f"  [{f.verdict:4}] {f.name:<15} {f.value:.4f}  (warn {f.warn_at}, fail {f.fail_at})")

    names = list(results)
    metrics = [f.name for f in next(iter(results.values())).findings]
    print("\n| finding | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for m in metrics:
        row = []
        for n in names:
            f = next(f for f in results[n].findings if f.name == m)
            row.append(f"{f.value:.3f} {f.verdict}")
        print(f"| {m} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
