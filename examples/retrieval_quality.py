"""Do vecscope findings track retrieval quality? Measure on labeled BEIR data.

Two experiments, both with exact cosine search so the index cannot be the cause:

1. Across models: embed SciFact and NFCorpus with several public models, audit the corpus
   with the queries, and correlate each finding with nDCG@10 (Spearman over models).
2. Within a model: degrade the best model's vectors in controlled ways (rogue dimension,
   shared offset, low rank, noise, duplicates, wrong query prefix / wrong query model) and
   record how the finding and nDCG@10 move together.

Needs: pip install sentence-transformers datasets scipy
Run:   python examples/retrieval_quality.py [--out docs/retrieval_quality.md]
Embeddings are cached under $HF_HOME/vecscope_rq so re-runs only recompute the tables.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vecscope import __version__, audit

DATASETS = ["scifact", "nfcorpus"]
# (model id, query prefix, document prefix)
MODELS = [
    ("sentence-transformers/average_word_embeddings_glove.6B.300d", "", ""),
    ("sentence-transformers/paraphrase-MiniLM-L3-v2", "", ""),
    ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    ("sentence-transformers/all-mpnet-base-v2", "", ""),
    ("intfloat/e5-small-v2", "query: ", "passage: "),
    ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    ("thenlper/gte-base", "", ""),
]
FINDINGS = ["hubness", "antihubs", "anisotropy", "effective_rank", "collapse", "rogue_dimensions",
            "norms", "metric_mismatch", "distance_concentration", "outliers", "connectivity",
            "ood_queries", "query_hubness"]
CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "vecscope_rq"


def short(model: str) -> str:
    return model.split("/")[-1].replace("average_word_embeddings_", "")


def load_beir(name: str):
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels = load_dataset(f"BeIR/{name}-qrels", split="test")
    doc_ids = [str(d) for d in corpus["_id"]]
    doc_text = [(t + " " + x).strip() for t, x in zip(corpus["title"], corpus["text"])]
    rel: dict[str, dict[str, int]] = {}
    for r in qrels:
        if int(r["score"]) > 0:
            rel.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    q_ids = [str(q) for q in queries["_id"]]
    keep = [i for i, q in enumerate(q_ids) if q in rel]
    q_ids = [q_ids[i] for i in keep]
    q_text = [queries["text"][i] for i in keep]
    return doc_ids, doc_text, q_ids, q_text, rel


def embed(model: str, dataset: str, doc_text, q_text, qp: str, dp: str):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{dataset}_{short(model)}.npz"
    if f.exists():
        z = np.load(f)
        return z["docs"], z["queries"]
    import torch
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model, device="cpu", model_kwargs={"dtype": torch.float32})
    m.max_seq_length = min(m.max_seq_length or 512, 512)
    t = time.time()
    docs = m.encode([dp + d for d in doc_text], batch_size=32, convert_to_numpy=True,
                    show_progress_bar=False).astype(np.float32)
    qs = m.encode([qp + q for q in q_text], batch_size=64, convert_to_numpy=True,
                  show_progress_bar=False).astype(np.float32)
    print(f"  embedded {dataset} with {short(model)} in {time.time() - t:.0f}s", flush=True)
    np.savez(f, docs=docs, queries=qs)
    return docs, qs


def ndcg_at_k(docs, qs, doc_ids, q_ids, rel, k=10) -> tuple[float, float]:
    """nDCG@k and recall@100 with exact cosine search."""
    d = docs / np.maximum(np.linalg.norm(docs, axis=1, keepdims=True), 1e-12)
    q = qs / np.maximum(np.linalg.norm(qs, axis=1, keepdims=True), 1e-12)
    scores = q @ d.T
    top = np.argpartition(-scores, 100, axis=1)[:, :100]
    ndcg, recall = [], []
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for i, qid in enumerate(q_ids):
        order = top[i][np.argsort(-scores[i, top[i]])]
        gains = rel[qid]
        got = [gains.get(doc_ids[j], 0) for j in order[:k]]
        dcg = float(np.sum(np.array(got) * disc))
        ideal = sorted(gains.values(), reverse=True)[:k]
        idcg = float(np.sum(np.array(ideal) * disc[: len(ideal)]))
        ndcg.append(dcg / idcg if idcg else 0.0)
        recall.append(sum(doc_ids[j] in gains for j in order) / len(gains))
    return float(np.mean(ndcg)), float(np.mean(recall))


def findings_of(docs, qs) -> dict[str, float]:
    ev = audit(docs, k=10, sample=None, queries=qs)
    return {f.name: f.value for f in ev.findings}


def _rho(v: float) -> str:
    return "n/a" if math.isnan(v) else f"{v:+.2f}"  # constant columns have no rank correlation


def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else float("nan")


def degrade(name: str, level: float, docs, qs, rng):
    """Return (docs, queries) with one controlled defect at the given level."""
    scale = float(np.median(np.linalg.norm(docs, axis=1)))
    if name == "rogue_dimension":  # one high-variance dimension unrelated to content
        v = rng.normal(0, level * scale, (docs.shape[0], 1)).astype(np.float32)
        vq = rng.normal(0, level * scale, (qs.shape[0], 1)).astype(np.float32)
        return np.hstack([docs, v]), np.hstack([qs, vq])
    if name == "shared_offset":  # anisotropy: every vector shifted along one direction
        u = rng.normal(size=docs.shape[1]).astype(np.float32)
        u /= np.linalg.norm(u)
        return docs + level * scale * u, qs + level * scale * u
    if name == "low_rank":  # keep only the top `level` principal components
        r = int(level)
        mu = docs.mean(0)
        _, _, vt = np.linalg.svd(docs - mu, full_matrices=False)
        p = vt[:r].T
        return ((docs - mu) @ p) @ p.T + mu, ((qs - mu) @ p) @ p.T + mu
    if name == "noise":  # isotropic noise whose total norm is `level` x the median vector norm
        sigma = level * scale / np.sqrt(docs.shape[1])
        return (docs + rng.normal(0, sigma, docs.shape).astype(np.float32), qs)
    if name == "duplicates":  # a share of the corpus copied with tiny jitter
        m = int(level * docs.shape[0])
        pick = rng.choice(docs.shape[0], m, replace=False)
        extra = docs[pick] + rng.normal(0, 1e-3 * scale, (m, docs.shape[1])).astype(np.float32)
        return np.vstack([docs, extra]), qs
    raise ValueError(name)


def run(out: Path) -> None:
    results: dict = {"across_models": {}, "degradations": {}}
    lines = ["# Do the findings track retrieval quality?", "",
             ("Measured on BEIR SciFact and NFCorpus with exact cosine search, so index recall is "
              f"not a factor. Generated by `examples/retrieval_quality.py` with vecscope {__version__}\n"
              "defaults; verdict columns use those thresholds.\n"
              'See the README section "Do the findings predict retrieval quality?" for the '
              "conclusions."), ""]

    data = {ds: load_beir(ds) for ds in DATASETS}
    emb: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    # ---- 1. across models -------------------------------------------------------------
    for ds in DATASETS:
        doc_ids, doc_text, q_ids, q_text, rel = data[ds]
        rows = []
        for model, qp, dp in MODELS:
            docs, qs = embed(model, ds, doc_text, q_text, qp, dp)
            emb[(ds, model)] = (docs, qs)
            nd, rc = ndcg_at_k(docs, qs, doc_ids, q_ids, rel)
            fv = findings_of(docs, qs)
            rows.append({"model": short(model), "dim": int(docs.shape[1]), "ndcg10": nd,
                         "recall100": rc, **fv})
            print(f"{ds:9s} {short(model):28s} nDCG@10={nd:.3f} R@100={rc:.3f}", flush=True)
        results["across_models"][ds] = rows
        lines += [f"## Across models: {ds} ({len(doc_ids)} docs, {len(q_ids)} queries)", "",
                  "| model | dim | nDCG@10 | " + " | ".join(FINDINGS) + " |",
                  "|---|---|---|" + "---|" * len(FINDINGS)]
        for r in sorted(rows, key=lambda r: -r["ndcg10"]):
            lines.append(f"| {r['model']} | {r['dim']} | {r['ndcg10']:.3f} | "
                         + " | ".join(f"{r[f]:.3g}" for f in FINDINGS) + " |")
        nd = [r["ndcg10"] for r in rows]
        lines += ["", ("Spearman correlation with nDCG@10 over the models (negative means a "
                       "higher value goes with worse retrieval):"), "",
                  "| finding | " + " | ".join(FINDINGS) + " |", "|---|" + "---|" * len(FINDINGS),
                  "| rho | " + " | ".join(_rho(spearman([r[f] for r in rows], nd)) for f in FINDINGS)
                  + " |", ""]

    # ---- 2. degradations of the best model ------------------------------------------
    plan = {
        "rogue_dimension": [0.0, 0.25, 0.5, 1.0, 2.0],
        "shared_offset": [0.0, 0.5, 1.0, 2.0, 4.0],
        "low_rank": [768, 128, 32, 8, 2],
        "noise": [0.0, 0.5, 1.0, 2.0, 4.0],
        "duplicates": [0.0, 0.1, 0.3, 0.6],
    }
    watch = {"rogue_dimension": "rogue_dimensions", "shared_offset": "anisotropy",
             "low_rank": "effective_rank", "noise": "distance_concentration",
             "duplicates": "collapse"}
    for ds in DATASETS:
        doc_ids, doc_text, q_ids, q_text, rel = data[ds]
        best = max(results["across_models"][ds], key=lambda r: r["ndcg10"])["model"]
        model = next(m for m, _, _ in MODELS if short(m) == best)
        docs0, qs0 = emb[(ds, model)]
        base_nd, _ = ndcg_at_k(docs0, qs0, doc_ids, q_ids, rel)
        lines += [f"## Controlled defects: {ds}, {best} (baseline nDCG@10 {base_nd:.3f})", "",
                  "| defect | level | watched finding | value | verdict | nDCG@10 | other findings that moved |",
                  "|---|---|---|---|---|---|---|"]
        base_f = findings_of(docs0, qs0)
        ev0 = audit(docs0, k=10, sample=None, queries=qs0)
        base_verdict = {f.name: f.verdict for f in ev0.findings}
        dres = []
        for defect, levels in plan.items():
            for lv in levels:
                if defect == "low_rank" and lv > docs0.shape[1]:
                    lv = docs0.shape[1]
                rng = np.random.default_rng(0)
                d, q = degrade(defect, lv, docs0, qs0, rng)
                if defect == "duplicates":  # extra copies are never relevant: ids beyond the corpus
                    ids = doc_ids + [f"dup{i}" for i in range(d.shape[0] - len(doc_ids))]
                else:
                    ids = doc_ids
                nd, _ = ndcg_at_k(d, q, ids, q_ids, rel)
                ev = audit(d, k=10, sample=None, queries=q)
                fv = {f.name: f.value for f in ev.findings}
                vd = {f.name: f.verdict for f in ev.findings}
                w = watch[defect]
                moved = [f"{n} {base_verdict[n]}->{vd[n]}" for n in FINDINGS
                         if n != w and vd[n] != base_verdict[n]]
                dres.append({"defect": defect, "level": lv, "ndcg10": nd, **fv, "verdicts": vd})
                lines.append(f"| {defect} | {lv:g} | {w} | {fv[w]:.3g} | {vd[w]} | {nd:.3f} | "
                             f"{', '.join(moved) or '-'} |")
                print(f"{ds:9s} {defect:16s} {lv:<5g} {w}={fv[w]:.3g} {vd[w]:4s} nDCG={nd:.3f}", flush=True)
        # query pipeline bugs
        for model2, qp2, dp2 in MODELS:
            if short(model2) == best:
                continue
            docs2, qs2 = emb[(ds, model2)]
            if docs2.shape[1] != docs0.shape[1]:
                continue
            nd, _ = ndcg_at_k(docs0, qs2, doc_ids, q_ids, rel)
            ev = audit(docs0, k=10, sample=None, queries=qs2)
            fv = {f.name: f.value for f in ev.findings}
            vd = {f.name: f.verdict for f in ev.findings}
            lines.append(f"| queries from {short(model2)} | - | ood_queries | {fv['ood_queries']:.3g} | "
                         f"{vd['ood_queries']} | {nd:.3f} | query_hubness {vd['query_hubness']} |")
            dres.append({"defect": f"queries_from_{short(model2)}", "ndcg10": nd, **fv, "verdicts": vd})
        qp = next(p for m, p, _ in MODELS if short(m) == best)
        if qp:
            import torch
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer(model, device="cpu", model_kwargs={"dtype": torch.float32})
            qs_np = m.encode(q_text, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
            nd, _ = ndcg_at_k(docs0, qs_np, doc_ids, q_ids, rel)
            ev = audit(docs0, k=10, sample=None, queries=qs_np.astype(np.float32))
            fv = {f.name: f.value for f in ev.findings}
            vd = {f.name: f.verdict for f in ev.findings}
            lines.append(f"| query prefix dropped | - | ood_queries | {fv['ood_queries']:.3g} | "
                         f"{vd['ood_queries']} | {nd:.3f} | query_hubness {vd['query_hubness']} |")
            dres.append({"defect": "query_prefix_dropped", "ndcg10": nd, **fv, "verdicts": vd})
        lines.append("")
        results["degradations"][ds] = {"model": best, "baseline_ndcg10": base_nd,
                                       "baseline": base_f, "rows": dres}

    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/retrieval_quality.md")
    run(Path(ap.parse_args().out))
