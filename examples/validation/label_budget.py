"""How many labeled queries does a retrieval tolerance need to catch a harmful change?

For every faulted pipeline of faults.py, the per-query exact nDCG@10 of the clean and the
faulted run is computed on all judged queries. The full set says whether the fault is
harmful (nDCG drop above the tolerance). Then ``--draws`` seeded subsets of N judged
queries play the labeled set a team could afford, and the gate ``drop on the subset >
tolerance`` is scored against the full-set answer:

- caught: share of harmful (dataset, model, fault) cases the subset gate fails
- false alarm: share of harmless cases (full drop at most the tolerance) it fails

Run:   python examples/validation/label_budget.py [--role heldout|confirm]
Needs: the run.py cache and the faults.py records
Writes <cache>/label_budget_<role>.json and prints the table
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from faults import FAULTS, _other_model
from registry import DATASETS, role
from run import SEED, load, out_dir, qrels_rows

from vecscope._knn import normalize

SIZES = (25, 50, 100, 200, 400, 800)
TOLERANCE = 0.01


def per_query_ndcg(docs: np.ndarray, queries: np.ndarray, qrels: dict, k: int = 10) -> dict:
    d, q = normalize(docs), normalize(queries)
    judged = sorted(qrels)
    out = {}
    for s in range(0, len(judged), 512):
        rows = judged[s : s + 512]
        sim = q[rows] @ d.T
        part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sim, part, axis=1), axis=1, kind="stable")
        top = np.take_along_axis(part, order, axis=1)
        for qi, ids in zip(rows, top):
            rel = qrels[qi]
            dcg = sum(
                (2.0 ** rel.get(int(x), 0.0) - 1.0) / math.log2(r + 2) for r, x in enumerate(ids)
            )
            ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:k]
            idcg = sum((2.0**g - 1.0) / math.log2(r + 2) for r, g in enumerate(ideal))
            out[qi] = dcg / idcg
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", help="run.py cache directory")
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--role", default="heldout", help="dataset role in registry.py")
    args = p.parse_args(argv)
    cache = out_dir(args.out)
    results = {}
    for f in (cache / "results").glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        results[(r["dataset"]["name"], r["model"]["short"])] = r
    recs = [
        json.loads(f.read_text(encoding="utf-8")) for f in sorted((cache / "faults").glob("*.json"))
    ]
    recs = [r for r in recs if role(r["dataset"]) == args.role]
    by_pipe: dict = {}
    for r in recs:
        by_pipe.setdefault((r["dataset"], r["model"]), []).append(r["fault"])
    cases = []
    loaded: dict = {}
    for (name, model), faults in sorted(by_pipe.items()):
        if name not in loaded:
            data = load(DATASETS[name], cache)
            loaded = {name: qrels_rows(data[0], data[2], data[4])}
        qrels = loaded[name]
        z = np.load(cache / "emb" / f"{name}__{model}.npz")
        docs, queries = z["docs"], z["queries"]
        clean = per_query_ndcg(docs, queries, qrels)
        for fault in faults:
            other = (
                _other_model(name, model, docs.shape[1], cache, results)
                if fault == "query_model"
                else None
            )
            fd, fq = FAULTS[fault](docs, queries, other, np.random.default_rng(SEED))
            bad = per_query_ndcg(fd, fq, qrels)
            drops = np.array([clean[q] - bad[q] for q in sorted(clean)])
            cases.append({"dataset": name, "model": model, "fault": fault, "drops": drops})
        print(f"{name:<22} {model:<38} {len(clean)} judged queries", flush=True)
    rng = np.random.default_rng(SEED)
    table = []
    for n in SIZES:
        caught = alarms = harmful = harmless = 0
        bands = {"0.01-0.02": [0, 0], "0.02-0.05": [0, 0], ">0.05": [0, 0]}
        for c in cases:
            drops = c["drops"]
            if len(drops) < n:
                continue
            full = float(drops.mean())
            hits = (
                sum(
                    float(drops[rng.choice(len(drops), n, replace=False)].mean()) > TOLERANCE
                    for _ in range(args.draws)
                )
                / args.draws
            )
            if full > TOLERANCE:
                harmful += 1
                caught += hits
                band = "0.01-0.02" if full <= 0.02 else "0.02-0.05" if full <= 0.05 else ">0.05"
                bands[band][0] += hits
                bands[band][1] += 1
            else:
                harmless += 1
                alarms += hits
        table.append(
            {
                "labeled_queries": n,
                "harmful_cases": harmful,
                "caught": caught / harmful if harmful else None,
                "harmless_cases": harmless,
                "false_alarm": alarms / harmless if harmless else None,
                "caught_by_drop": {b: (h / m if m else None, m) for b, (h, m) in bands.items()},
            }
        )
    for row in table:
        cb = "  ".join(
            f"{b} {v[0]:.0%} of {v[1]}" if v[0] is not None else f"{b} -"
            for b, v in row["caught_by_drop"].items()
        )
        print(
            f"N={row['labeled_queries']:<4} caught {row['caught']:.0%} of {row['harmful_cases']} "
            f"harmful, false alarm {row['false_alarm']:.0%} of {row['harmless_cases']} harmless; "
            f"by full drop: {cb}"
        )
    (cache / f"label_budget_{args.role}.json").write_text(
        json.dumps({"tolerance": TOLERANCE, "draws": args.draws, "table": table}, indent=1),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
