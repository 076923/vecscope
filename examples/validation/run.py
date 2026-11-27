"""Embed the held-out public datasets with public models and audit every pair.

For each (dataset, model) the corpus and the judged queries are embedded once (cached), then
``vecscope.audit`` runs with the queries and the qrels, so the record holds every finding's
value and verdict next to the exact-search nDCG@10 of the same vectors. One JSON file per
pair is written under ``<out>/results``; an existing file is skipped, so an interrupted run
resumes where it stopped, unless ``--reaudit`` asks to audit the cached embeddings again
(after a change to vecscope; the embedding time of the first run is kept). ``report.py`` turns the files into the tables in the docs.

Needs:  pip install sentence-transformers datasets scipy  (a GPU makes it hours, not days)
Run:    python examples/validation/run.py [--datasets xpqa-ja cosqa] [--models bge-m3] [--reaudit]
Output: $VECSCOPE_VALIDATION_DIR, default $HF_HOME/vecscope_validation

Only public data and public models are used; see registry.py.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from registry import DATASETS, MODELS, Dataset, Model

from vecscope import __version__, audit

SEED = 0
# ESCI gains as in the KDD Cup 2022 task 1 metric
ESCI_GAIN = {"Exact": 1.0, "Substitute": 0.1, "Complement": 0.01, "Irrelevant": 0.0}


def out_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    if os.environ.get("VECSCOPE_VALIDATION_DIR"):
        return Path(os.environ["VECSCOPE_VALIDATION_DIR"])
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf / "vecscope_validation"


def _id_col(cols: list[str]) -> str:
    return "_id" if "_id" in cols else "id"


def _text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


def load_mteb(ds: Dataset):
    from datasets import load_dataset

    corpus = load_dataset(ds.repo, ds.corpus[0], split=ds.corpus[1])
    queries = load_dataset(ds.repo, ds.queries[0], split=ds.queries[1])
    qrels = load_dataset(ds.repo, ds.qrels[0], split=ds.qrels[1])
    cid = _id_col(corpus.column_names)
    doc_ids = [str(i) for i in corpus[cid]]
    doc_text = [_text(r) for r in corpus]
    rel: dict[str, dict[str, float]] = {}
    for r in qrels:
        g = float(r["score"])
        if g > 0:
            rel.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = g
    qid = _id_col(queries.column_names)
    q_all = {str(i): t for i, t in zip(queries[qid], queries["text"])}
    q_ids = [q for q in q_all if q in rel]
    return doc_ids, doc_text, q_ids, [q_all[q] for q in q_ids], rel


def load_esci(ds: Dataset):
    from datasets import load_dataset

    data = load_dataset(ds.repo, split="test")
    data = data.filter(
        lambda r: r["product_locale"] == ds.locale and str(r["small_version"]) == "1",
        num_proc=8,
    )
    rows = data.select_columns(["query_id", "query", "product_id", "product_title", "esci_label"])
    qids = sorted({str(q) for q in rows["query_id"]}, key=lambda s: (len(s), s))
    rng = np.random.default_rng(SEED)
    pick = set(rng.choice(qids, min(ds.max_queries, len(qids)), replace=False).tolist())
    q_text: dict[str, str] = {}
    titles: dict[str, str] = {}
    rel: dict[str, dict[str, float]] = {}
    for r in rows:
        q = str(r["query_id"])
        if q not in pick:
            continue
        q_text[q] = r["query"]
        titles.setdefault(r["product_id"], (r["product_title"] or "").strip())
        g = ESCI_GAIN[r["esci_label"]]
        if g > 0:
            rel.setdefault(q, {})[r["product_id"]] = g
    doc_ids = sorted(titles)
    q_ids = sorted((q for q in q_text if q in rel), key=lambda s: (len(s), s))
    return doc_ids, [titles[d] for d in doc_ids], q_ids, [q_text[q] for q in q_ids], rel


def load(ds: Dataset, cache: Path):
    """(doc_ids, doc_text, q_ids, q_text, rel), cached as JSON so later models skip the Hub."""
    f = cache / "data" / f"{ds.name}.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        return d["doc_ids"], d["doc_text"], d["q_ids"], d["q_text"], d["rel"]
    doc_ids, doc_text, q_ids, q_text, rel = (load_esci if ds.kind == "esci" else load_mteb)(ds)
    if ds.max_queries and len(q_ids) > ds.max_queries and ds.kind != "esci":
        keep = sorted(np.random.default_rng(SEED).choice(len(q_ids), ds.max_queries, False))
        q_ids, q_text = [q_ids[i] for i in keep], [q_text[i] for i in keep]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(
        json.dumps(
            {
                "doc_ids": doc_ids,
                "doc_text": doc_text,
                "q_ids": q_ids,
                "q_text": q_text,
                "rel": rel,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return doc_ids, doc_text, q_ids, q_text, rel


_LOADED: dict[str, object] = {}


def _model(m: Model):
    if m.id not in _LOADED:
        import torch
        from sentence_transformers import SentenceTransformer

        _LOADED.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        st = SentenceTransformer(
            m.id,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={"torch_dtype": getattr(torch, m.dtype)},
            **m.kwargs,
        )
        st.max_seq_length = min(st.max_seq_length or 512, 512)
        _LOADED[m.id] = st
    return _LOADED[m.id]


def embed(m: Model, ds: Dataset, doc_text, q_text, cache: Path):
    f = cache / "emb" / f"{ds.name}__{m.short}.npz"
    if f.exists():
        z = np.load(f)
        return z["docs"], z["queries"], None
    st = _model(m)
    t = time.time()
    docs = st.encode(
        [m.doc_prefix + d for d in doc_text], batch_size=64, convert_to_numpy=True
    ).astype(np.float32)
    kw = {"prompt_name": m.query_prompt_name} if m.query_prompt_name else {}
    qs = st.encode(
        [m.query_prefix + q for q in q_text], batch_size=64, convert_to_numpy=True, **kw
    ).astype(np.float32)
    secs = time.time() - t
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, docs=docs, queries=qs)
    return docs, qs, secs


def qrels_rows(doc_ids, q_ids, rel) -> dict[int, dict[int, float]]:
    row = {d: i for i, d in enumerate(doc_ids)}
    out: dict[int, dict[int, float]] = {}
    for qi, q in enumerate(q_ids):
        judged = {row[d]: g for d, g in rel[q].items() if d in row}
        if judged:
            out[qi] = judged
    return out


def run_pair(ds: Dataset, m: Model, data, cache: Path, sample: int) -> dict:
    doc_ids, doc_text, q_ids, q_text, rel = data
    docs, qs, embed_secs = embed(m, ds, doc_text, q_text, cache)
    qrels = qrels_rows(doc_ids, q_ids, rel)
    t = time.time()
    ev = audit(docs, k=10, queries=qs, qrels=qrels, sample=sample)
    e = ev.retrieval["exact"]
    lengths = np.array([len(d) for d in doc_text])
    return {
        "dataset": {
            "name": ds.name,
            "repo": ds.repo,
            "language": ds.language,
            "domain": ds.domain,
            "license": ds.license,
            "role": ds.role,
            "n_docs": len(doc_ids),
            "n_queries": len(qrels),
            "median_doc_chars": int(np.median(lengths)),
        },
        "model": {"id": m.id, "short": m.short, "languages": m.languages, "dtype": m.dtype},
        "ndcg_at_10": e["ndcg_at_k"],
        "recall_at_10": e["recall_at_k"],
        "mrr_at_10": e["mrr_at_k"],
        "duplicate_result_share": e["duplicate_result_share"],
        "verdict": ev.verdict,
        "findings": {
            f.name: {"value": f.value, "verdict": f.verdict, "inconclusive": f.inconclusive}
            for f in ev.findings
        },
        "plan": [s["group"] for s in ev.plan],
        "sampled": ev.sampled,
        "n_audited": ev.n,
        "dim": ev.dim,
        "seconds": {"embed": embed_secs, "audit": time.time() - t},
        "vecscope": __version__,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--datasets", nargs="*", default=list(DATASETS))
    p.add_argument("--models", nargs="*", help="model ids or short names (default all)")
    p.add_argument("--out", help="cache and results directory")
    p.add_argument("--sample", type=int, default=100_000, help="vecscope audit --sample")
    p.add_argument(
        "--reaudit",
        action="store_true",
        help="audit pairs that already have a result again from the cached embeddings",
    )
    args = p.parse_args(argv)
    cache = out_dir(args.out)
    (cache / "results").mkdir(parents=True, exist_ok=True)
    models = [m for m in MODELS if not args.models or m.id in args.models or m.short in args.models]
    unknown = set(args.datasets) - set(DATASETS)
    if unknown:
        p.error(f"unknown datasets {sorted(unknown)}; known: {sorted(DATASETS)}")
    meta = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    (cache / "environment.json").write_text(json.dumps(meta, indent=1))
    # model-major order: each model is loaded once and visits every dataset
    data_cache: dict[str, tuple] = {}
    failed = 0
    for m in models:
        for name in args.datasets:
            ds = DATASETS[name]
            f = cache / "results" / f"{ds.name}__{m.short}.json"
            old = None
            if f.exists():
                if not args.reaudit:
                    continue
                old = json.loads(f.read_text(encoding="utf-8"))
            t = time.time()
            try:
                if name not in data_cache:
                    data_cache[name] = load(ds, cache)
                rec = run_pair(ds, m, data_cache[name], cache, args.sample)
                if old and rec["seconds"]["embed"] is None:
                    rec["seconds"]["embed"] = old["seconds"]["embed"]
            # keep a long run going (Hub errors are OSErrors, CUDA out of memory a
            # RuntimeError); the pair is retried on the next run
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                print(f"{ds.name:<22} {m.short:<40} ERROR {type(e).__name__}: {e}", flush=True)
                failed += 1
                continue
            f.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
            print(
                f"{ds.name:<22} {m.short:<40} nDCG@10 {rec['ndcg_at_10']:.3f}  "
                f"{rec['verdict']:<4}  {time.time() - t:6.0f}s",
                flush=True,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
