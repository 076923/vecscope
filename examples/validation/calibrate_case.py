"""A public `vecscope calibrate` case: own thresholds for a domain the defaults do not fit.

On very short documents (product answers, product titles) several defaults flag good models,
because short near-paraphrases sit close together in every model. This script takes the
embeddings that run.py cached for one dataset, treats each model as a pipeline state, splits
the judged queries in two seeded halves, audits every state on each half, and runs
``vecscope calibrate`` with the first half for fitting and the second as the held-out check.

Run:   python examples/validation/calibrate_case.py --dataset xpqa-ja [--min-quality 0.5]
Needs: the run.py cache for that dataset (embeddings of at least ten models)
Writes <cache>/calibrate/<dataset>/: fit_*.json, hold_*.json, report.json, thresholds.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from registry import DATASETS, MODELS
from run import SEED, load, out_dir, qrels_rows

from vecscope import audit
from vecscope.calibrate import calibrate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", default="xpqa-ja", choices=sorted(DATASETS))
    p.add_argument(
        "--min-quality",
        type=float,
        help="nDCG@10 an acceptable state reaches (default: 80%% of the best state on the fit half)",
    )
    p.add_argument("--out", help="run.py cache directory")
    args = p.parse_args(argv)
    cache = out_dir(args.out)
    ds = DATASETS[args.dataset]
    doc_ids, _, q_ids, _, rel = load(ds, cache)
    order = np.random.default_rng(SEED).permutation(len(q_ids))
    halves = {"fit": np.sort(order[: len(order) // 2]), "hold": np.sort(order[len(order) // 2 :])}
    work = cache / "calibrate" / ds.name
    work.mkdir(parents=True, exist_ok=True)
    states: dict[str, dict[str, dict]] = {"fit": {}, "hold": {}}
    for m in MODELS:
        f = cache / "emb" / f"{ds.name}__{m.short}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        for half, rows in halves.items():
            q = [q_ids[i] for i in rows]
            ev = audit(
                z["docs"], k=10, queries=z["queries"][rows], qrels=qrels_rows(doc_ids, q, rel)
            )
            d = ev.to_dict()
            states[half][m.short] = d
            (work / f"{half}_{m.short}.json").write_text(json.dumps(d), encoding="utf-8")
    if len(states["fit"]) < 10:
        p.error(f"{len(states['fit'])} models embedded for {ds.name}; run run.py for at least 10")
    best = max(d["retrieval"]["exact"]["ndcg_at_k"] for d in states["fit"].values())
    min_q = args.min_quality if args.min_quality is not None else round(0.8 * best, 3)
    names = sorted(states["fit"])
    rep = calibrate(
        [states["fit"][n] for n in names],
        min_quality=min_q,
        names=names,
        holdout=[states["hold"][n] for n in names],
    )
    (work / "report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (work / "thresholds.json").write_text(json.dumps(rep.get("thresholds", {}), indent=1))
    print(json.dumps({k: rep[k] for k in rep if k != "findings"}, indent=1, default=str)[:3000])
    print(f"min_quality {min_q}; wrote {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
