"""Retrieval quality on labeled queries, next to the findings that might explain it.

The findings describe the vector space; whether the right documents come back needs labels.
With relevance judgments (qrels) this module scores exact cosine search, and optionally the
ranking a production index returned, so a change to the vectors or the index can be judged
by the finding it was meant to fix *and* by retrieval quality:

- ndcg_at_k, recall_at_k, mrr_at_k, hit_rate_at_k over the judged queries
- duplicate_result_share: share of top-k slots taken by a near-duplicate of a higher-ranked
  result, over every query (what deduplication should lower)
- for a supplied ranking, overlap_with_exact: share of its top-k that exact search also
  returns (what raising ef_search / nprobe should raise)

Everything is exact and deterministic; there are no verdicts, because no threshold on nDCG
holds across domains.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from vecscope._knn import normalize
from vecscope.diagnostics.queries import query_knn


def _row(value, what: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"qrels: {what} must be an integer row position, got {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"qrels: {what} must be an integer row position, got {value!r}") from None
    if not math.isfinite(f) or f != int(f) or f < 0:
        raise ValueError(f"qrels: {what} must be a non-negative integer row position, got {value!r}")
    return int(f)


def _grade(value) -> float:
    if isinstance(value, bool):
        raise TypeError(f"qrels: grade must be a number, got {value!r}")
    try:
        g = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"qrels: grade must be a number, got {value!r}") from None
    if not math.isfinite(g):
        raise ValueError(f"qrels: grade must be finite, got {value!r}")
    return g


def parse_qrels(obj) -> dict[int, dict[int, float]]:
    """Relevance judgments as {query row: {corpus row: grade}}; rows are positions in the query
    and corpus files. Accepts that mapping (keys may be strings), or a list of
    ``[query, doc, grade]`` / ``{"query": q, "doc": d, "grade": g}`` rows (BEIR names
    ``query-id``, ``corpus-id``, ``score`` are accepted too). Grades <= 0 mean not relevant."""
    out: dict[int, dict[int, float]] = {}

    def add(q, d, g=1.0):
        out.setdefault(_row(q, "query"), {})[_row(d, "doc")] = _grade(g)

    if isinstance(obj, dict):
        for q, docs in obj.items():
            if not isinstance(docs, dict):
                raise TypeError("qrels: a mapping must be {query: {doc: grade}}")
            for d, g in docs.items():
                add(q, d, g)
    elif isinstance(obj, list):
        for r in obj:
            if isinstance(r, dict):
                q = r.get("query", r.get("query-id", r.get("query_id")))
                d = r.get("doc", r.get("corpus-id", r.get("doc_id")))
                g = r.get("grade", r.get("score", 1.0))
                if q is None or d is None:
                    raise ValueError(f"qrels: row {r!r:.80} needs query and doc")
                add(q, d, g)
            elif isinstance(r, (list, tuple)) and len(r) in (2, 3):
                add(*r)
            else:
                raise ValueError(f"qrels: row {r!r:.80} must be [query, doc(, grade)] or an object")
    else:
        raise TypeError("qrels must be a mapping or a list of rows")
    if not any(g > 0 for docs in out.values() for g in docs.values()):
        raise ValueError("qrels has no relevant (grade > 0) judgment")
    return out


def load_qrels(path: str | Path, data: bytes | None = None) -> dict[int, dict[int, float]]:
    """Read qrels from .json (mapping or list), .jsonl (one row per line) or .tsv/.csv/.txt
    (``query doc [grade]`` per line; a non-numeric first line is a header). ``data`` is the
    file's content when the caller has already read it."""
    p = Path(path)
    if data is None and not p.exists():
        raise FileNotFoundError(f"qrels file not found: {p}")
    suf = p.suffix.lower()
    if suf not in {".json", ".jsonl", ".tsv", ".csv", ".txt"}:
        raise ValueError(f"unsupported qrels file: {p} (use .json/.jsonl/.tsv/.csv)")
    try:
        text = p.read_text(encoding="utf-8-sig") if data is None else data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(f"{p}: not a text qrels file (binary or not UTF-8)") from None
    if suf == ".json":
        try:
            return parse_qrels(json.loads(text))
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}: not valid JSON ({e})") from None
    if suf == ".jsonl":
        try:
            return parse_qrels([json.loads(ln) for ln in text.splitlines() if ln.strip()])
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}: not valid JSON lines ({e})") from None
    rows = []  # .tsv, .csv, .txt
    for i, ln in enumerate(text.splitlines()):
        if not ln.strip():
            continue
        parts = ln.replace(",", " ").split()
        if i == 0 and not all(_is_number(x) for x in parts):
            continue  # header
        rows.append(parts)
    return parse_qrels(rows)


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def qrels_fingerprint(qrels: dict[int, dict[int, float]]) -> str:
    """Stable hash of the judgments, so a baseline diff can tell whether both runs used the
    same labels."""
    canon = sorted((q, d, g) for q, docs in qrels.items() for d, g in docs.items())
    return hashlib.sha256(json.dumps(canon).encode()).hexdigest()[:16]


def _scores(ranked: np.ndarray, qrels: dict, k: int, judged: list[int]) -> dict:
    ndcg, recall, rr, hit = [], [], [], []
    for q in judged:
        rel = qrels[q]
        ids = [int(i) for i in ranked[q][:k] if i >= 0]
        dcg = sum((2.0 ** rel.get(d, 0.0) - 1.0) / math.log2(r + 2)
                  for r, d in enumerate(ids) if rel.get(d, 0.0) > 0)
        ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:k]
        idcg = sum((2.0 ** g - 1.0) / math.log2(r + 2) for r, g in enumerate(ideal))
        ndcg.append(dcg / idcg)
        n_rel = sum(1 for g in rel.values() if g > 0)
        found = [r for r, d in enumerate(ids) if rel.get(d, 0.0) > 0]
        recall.append(len(found) / n_rel)
        rr.append(1.0 / (found[0] + 1) if found else 0.0)
        hit.append(1.0 if found else 0.0)
    return {"ndcg_at_k": float(np.mean(ndcg)), "recall_at_k": float(np.mean(recall)),
            "mrr_at_k": float(np.mean(rr)), "hit_rate_at_k": float(np.mean(hit))}


def _duplicate_share(c: np.ndarray, ranked: np.ndarray, k: int, near_dup_cos: float) -> float:
    dup = slots = 0
    for row in ranked:
        ids = row[:k][row[:k] >= 0]
        if ids.size == 0:
            continue
        v = c[ids].astype(np.float64)
        sim = v @ v.T
        earlier = np.tril(sim >= near_dup_cos, k=-1)  # slot j duplicates a higher-ranked slot
        dup += int(earlier.any(axis=1).sum())
        slots += ids.size
    return dup / slots if slots else 0.0


def evaluate_retrieval(
    corpus: np.ndarray,
    queries: np.ndarray,
    qrels: dict[int, dict[int, float]],
    k: int = 10,
    ranked: np.ndarray | None = None,
    near_dup_cos: float = 0.99,
) -> dict:
    """Score exact cosine top-k (and ``ranked``, an (m x >= k) array of corpus row ids per
    query from the production index, -1 for fillers) against ``qrels``."""
    corpus = np.asarray(corpus)
    queries = np.asarray(queries)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    k = int(min(k, corpus.shape[0]))
    m, n = queries.shape[0], corpus.shape[0]
    for q, docs in qrels.items():
        if q >= m:
            raise ValueError(f"qrels names query row {q} but there are {m} queries")
        for d in docs:
            if d >= n:
                raise ValueError(f"qrels names corpus row {d} but the corpus has {n} rows")
    judged = sorted(q for q, docs in qrels.items() if any(g > 0 for g in docs.values()))
    c = normalize(corpus)
    exact, _ = query_knn(c, queries, k)
    out = {
        "k": k,
        "n_queries": int(m),
        "judged_queries": len(judged),
        "qrels_fingerprint": qrels_fingerprint(qrels),
        "near_dup_cos": float(near_dup_cos),
        "exact": {**_scores(exact, qrels, k, judged),
                  "duplicate_result_share": _duplicate_share(c, exact, k, near_dup_cos)},
        "ranked": None,
    }
    if ranked is not None:
        r = np.asarray(ranked)
        if r.ndim != 2 or r.shape[0] != m or not np.issubdtype(r.dtype, np.integer):
            raise ValueError(f"ranked must be integer corpus row ids shaped ({m}, k), got "
                             f"{r.shape} {r.dtype}")
        if r.shape[1] < k:
            raise ValueError(f"ranked has {r.shape[1]} columns; at least k={k} are needed")
        if r.size and r.max() >= n:
            raise ValueError(f"ranked contains row {int(r.max())} but the corpus has {n} rows")
        r = r[:, :k].astype(np.int64)
        overlap = np.mean([np.intersect1d(a[a >= 0], b).size / k for a, b in zip(r, exact)])
        out["ranked"] = {**_scores(r, qrels, k, judged),
                         "duplicate_result_share": _duplicate_share(c, r, k, near_dup_cos),
                         "overlap_with_exact": float(overlap)}
    e = out["exact"]
    out["explanation"] = (
        f"exact cosine top-{k} over {len(judged)} judged queries: nDCG@{k} {e['ndcg_at_k']:.4f}, "
        f"recall@{k} {e['recall_at_k']:.4f}, MRR@{k} {e['mrr_at_k']:.4f}; "
        f"{e['duplicate_result_share']:.1%} of result slots repeat a near-duplicate of a "
        "higher-ranked result."
        + ("" if out["ranked"] is None else
           f" The supplied ranking scores nDCG@{k} {out['ranked']['ndcg_at_k']:.4f} and shares "
           f"{out['ranked']['overlap_with_exact']:.1%} of its top-{k} with exact search.")
        + " These are the numbers a change should move; compare them against a baseline "
          "with the same qrels_fingerprint.")
    return out
