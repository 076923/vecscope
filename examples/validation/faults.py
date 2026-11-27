"""Inject realistic pipeline faults into the good held-out pipelines and see what catches them.

The bad pipelines of run.py are mostly English-only models on other languages, an easy fault.
This script takes every good pipeline (nDCG@10 at least 90% of the best on its dataset) on a
corpus of at most ``--max-docs`` documents, applies one fault at a time to its cached
embeddings, and audits the result with the queries and qrels. Each faulted run is judged
three ways, as a CI job would see it:

- absolute: the verdict reaches ``fail`` (``vecscope audit`` with the default ``--fail-on``)
- findings diff: against the clean run as baseline, a verdict got worse (``--fail-on none
  --baseline``)
- retrieval gate: against the same baseline, exact nDCG@10 fell by more than 0.01
  (``--retrieval-tolerance ndcg_at_k=0.01``)

Records written since the label-free result change was added also hold
``result_agreement``: the share of the sampled top-k results (200 queries, 200 rows) that
the fault changed (``--result-sample 200``).

Faults (seeded):

- query_model: the queries come from another model with the same dimension
- truncate: both sides keep the first half of the dimensions (an index configured with the
  wrong dimension, a model that was not trained for truncation)
- zero_rows: 5% of the documents are zero vectors (failed encodes stored as zeros)
- duplicates: 20% of the documents are stored twice (re-ingestion without dedup)
- id_shift: 10% of the rows point at another document's vector (a lost id mapping)
- quantize: every value rounded to 4 bits per dimension (int4 without per-vector scaling)

Run:   python examples/validation/faults.py [--max-docs 25000] [--role heldout|confirm]
Needs: the run.py cache (embeddings and results)
Writes <cache>/faults/<dataset>__<model>__<fault>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from registry import DATASETS, MODELS
from run import SEED, load, out_dir, qrels_rows

from vecscope import audit
from vecscope.evidence import WORSE, diff_evidence

GOOD = 0.9
TOLERANCE = {"ndcg_at_k": 0.01}
RESULT_SAMPLE = 200


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.where(n > 0, n, 1)).astype(np.float32)


def fault_query_model(docs, queries, other, rng):
    return docs, other


def fault_truncate(docs, queries, other, rng):
    h = docs.shape[1] // 2
    return np.ascontiguousarray(docs[:, :h]), np.ascontiguousarray(queries[:, :h])


def fault_zero_rows(docs, queries, other, rng):
    d = docs.copy()
    d[rng.choice(len(d), max(1, len(d) // 20), replace=False)] = 0
    return d, queries


def fault_duplicates(docs, queries, other, rng):
    extra = rng.choice(len(docs), max(1, len(docs) // 5), replace=False)
    # copies go after the originals, so the qrels still point at the original rows
    return np.concatenate([docs, docs[extra]]), queries


def fault_id_shift(docs, queries, other, rng):
    d = docs.copy()
    rows = rng.choice(len(d), max(2, len(d) // 10), replace=False)
    d[rows] = docs[np.roll(rows, 1)]
    return d, queries


def fault_quantize(docs, queries, other, rng):
    def q(x):
        x = _unit(x)
        lo, hi = float(x.min()), float(x.max())
        step = (hi - lo) / 15
        return (np.round((x - lo) / step) * step + lo).astype(np.float32)

    return q(docs), q(queries)


FAULTS = {
    "query_model": fault_query_model,
    "truncate": fault_truncate,
    "zero_rows": fault_zero_rows,
    "duplicates": fault_duplicates,
    "id_shift": fault_id_shift,
    "quantize": fault_quantize,
}


def _other_model(ds: str, model: str, dim: int, cache: Path, results: dict) -> np.ndarray | None:
    """Queries of the best other model on this dataset with the same dimension."""
    cands = [
        (results[(ds, m.short)]["ndcg_at_10"], m.short)
        for m in MODELS
        if m.short != model and (ds, m.short) in results and results[(ds, m.short)]["dim"] == dim
    ]
    if not cands:
        return None
    _, short = max(cands)
    return np.load(cache / "emb" / f"{ds}__{short}.npz")["queries"]


def _summary(ev: dict) -> dict:
    return {
        "verdict": ev["verdict"],
        "ndcg_at_10": ev["retrieval"]["exact"]["ndcg_at_k"],
        "flagged": sorted(f["name"] for f in ev["findings"] if f["verdict"] != "ok"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--max-docs", type=int, default=25_000)
    p.add_argument("--faults", nargs="*", default=list(FAULTS), choices=list(FAULTS))
    p.add_argument("--out", help="run.py cache directory")
    p.add_argument("--role", default="heldout", help="dataset role in registry.py")
    args = p.parse_args(argv)
    cache = out_dir(args.out)
    results = {}
    for f in (cache / "results").glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        results[(r["dataset"]["name"], r["model"]["short"])] = r
    best: dict[str, float] = {}
    for (ds, _), r in results.items():
        best[ds] = max(best.get(ds, 0.0), r["ndcg_at_10"])
    work = cache / "faults"
    work.mkdir(parents=True, exist_ok=True)
    for name, ds in DATASETS.items():
        if ds.role != args.role:
            continue
        good = [
            m
            for (d, m), r in sorted(results.items())
            if d == name
            and r["ndcg_at_10"] >= GOOD * best[name]
            and r["dataset"]["n_docs"] <= args.max_docs
        ]
        if not good:
            continue
        print(f"{name}: {len(good)} good pipelines", flush=True)
        doc_ids, _, q_ids, _, rel = load(ds, cache)
        qrels = qrels_rows(doc_ids, q_ids, rel)
        for model in good:
            z = np.load(cache / "emb" / f"{name}__{model}.npz")
            docs, queries = z["docs"], z["queries"]
            clean = None
            for fault in args.faults:
                out = work / f"{name}__{model}__{fault}.json"
                if out.exists():
                    continue
                if clean is None:
                    clean = audit(
                        docs, k=10, queries=queries, qrels=qrels, result_sample=RESULT_SAMPLE
                    ).to_dict()
                rng = np.random.default_rng(SEED)
                other = (
                    _other_model(name, model, docs.shape[1], cache, results)
                    if fault == "query_model"
                    else None
                )
                if fault == "query_model" and other is None:
                    continue
                t = time.time()
                fd, fq = FAULTS[fault](docs, queries, other, rng)
                ev = audit(fd, k=10, queries=fq, qrels=qrels, result_sample=RESULT_SAMPLE).to_dict()
                diff = diff_evidence(ev, clean, retrieval_tolerances=TOLERANCE)
                rec = {
                    "dataset": name,
                    "model": model,
                    "fault": fault,
                    "clean": _summary(clean),
                    "faulted": _summary(ev),
                    "ndcg_drop": clean["retrieval"]["exact"]["ndcg_at_k"]
                    - ev["retrieval"]["exact"]["ndcg_at_k"],
                    "absolute_fail": ev["verdict"] == "fail",
                    "absolute_fail_clean": clean["verdict"] == "fail",
                    "verdict_regressions": diff["regressions"],
                    "verdict_worse": WORSE[ev["verdict"]] > WORSE[clean["verdict"]],
                    "retrieval_regressions": diff["retrieval_regressions"],
                    "retrieval_unchecked": diff["retrieval_unchecked"],
                    "comparable": diff["comparable"],
                    "change_decision": diff["change_decision"]["decision"],
                    "result_agreement": diff["result_agreement"],
                    "seconds": time.time() - t,
                }
                out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
                print(
                    f"{name:<22} {model:<38} {fault:<12} drop {rec['ndcg_drop']:+.3f} "
                    f"abs {'F' if rec['absolute_fail'] else '-'} "
                    f"diff {','.join(rec['verdict_regressions']) or '-'} "
                    f"ret {'R' if rec['retrieval_regressions'] else '-'}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
