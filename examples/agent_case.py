"""Build the data of the agent case study in docs/agent_case.md: real SciFact embeddings
(bge-small-en-v1.5) with two ingestion faults injected, labeled queries, and the top-10 a
production inner-product index returns.

The faults are ones a pipeline actually makes:

- a batch ingested twice: documents 1000 to 1599 appended again at the end
- one ingestion worker that skipped normalization: documents 2600 onward stored with a
  per-row length between 0.6 and 1.6

qrels.tsv refers to row positions in corpus.npy, and prod_topk.npy holds the top-10 of an
inner-product index over corpus.npy for each query (what IndexFlatIP would return).

Run:   python examples/agent_case.py --out agent_case
Needs: pip install datasets sentence-transformers (the embeddings are cached after one run)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_quality import embed, load_beir

PROMPT = """Our SciFact search keeps returning the same paper twice and some queries look
worse since the last ingestion. The corpus embeddings are corpus.npy, query embeddings
queries.npy, relevance labels qrels.tsv (query row, corpus row, grade; rows are positions in
those files) and the production index's top-10 per query prod_topk.npy (an inner-product
index over corpus.npy). Use the vecscope MCP tools to find what is wrong, fix the corpus
into corpus_fixed.npy (and recompute the production top-10 into prod_topk_fixed.npy the
same way), and show with a re-audit that the fix worked. Do not change the queries."""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="agent_case")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    doc_ids, doc_text, q_ids, q_text, rel = load_beir("scifact")
    docs, queries = embed("BAAI/bge-small-en-v1.5", "scifact", doc_text, q_text,
                          "Represent this sentence for searching relevant passages: ", "")
    docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    corpus = docs.copy()
    corpus[2600:] *= rng.uniform(0.6, 1.6, (len(docs) - 2600, 1)).astype(np.float32)
    corpus = np.concatenate([corpus, corpus[1000:1600]]).astype(np.float32)

    row = {d: i for i, d in enumerate(doc_ids)}
    with open(out / "qrels.tsv", "w", encoding="utf-8") as f:
        f.write("query\tdoc\tgrade\n")
        f.writelines(f"{qi}\t{row[d]}\t{g}\n"
                     for qi, qid in enumerate(q_ids) for d, g in rel[qid].items())
    top = np.argsort(-(queries @ corpus.T), axis=1, kind="stable")[:, :10]
    np.save(out / "corpus.npy", corpus)
    np.save(out / "queries.npy", queries.astype(np.float32))
    np.save(out / "prod_topk.npy", top.astype(np.int64))
    (out / "PROMPT.txt").write_text(PROMPT + "\n", encoding="utf-8")
    print(f"wrote {out}/: corpus {corpus.shape}, queries {queries.shape}, "
          f"{sum(len(r) for r in rel.values())} judgments")


if __name__ == "__main__":
    main()
