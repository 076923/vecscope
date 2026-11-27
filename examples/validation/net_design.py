"""Design data for a net rule on agreed top hits: lost and gained, on earlier sets only.

The sixth set showed that the agreed top hits flag model changes that improve retrieval:
the diff counts the agreed top hits a new run loses, never the reference's first results it
finds that the baseline missed. This script measures both, on data that is no longer blind:

- the injected faults of the fourth, fifth and sixth sets (duplicates left out), rebuilt
  from the cached embeddings with the same seed;
- model changes on every dataset run so far (sets 1 to 6): multilingual-e5-base, and then
  paraphrase-multilingual-MiniLM-L12-v2 (a weaker baseline, so most changes improve), replaced
  by each other model except the reference (gte-multilingual-base on the fifth set, bge-m3
  elsewhere), judged by the full-label nDCG@10 of run.py.

Per change, on the sampled queries (200, k=10), with a = the baseline's top-k holds the
reference's first result and b = the new top-k does:

- ``agreed`` (a), ``lost`` (a and not b), ``gained`` (not a and b), ``compared``
- the lost share and its z, as shipped
- the net drop: the mean of a - b over the compared queries, its standard error and z

Only the result sample is computed (vecscope.results.result_sample, as the audit does), so
no full audit runs. Writes <cache>/net_design/*.json and examples/validation/net_design.json
with each candidate rule's rates. This is design data: the rule chosen here is tested on a
seventh set (preregistration_seventh.md).

    python examples/validation/net_design.py
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
from registry import MODELS, role
from run import SEED, out_dir

from vecscope.results import REFERENCE_Z, result_sample

M, K = 200, 10
BASELINES = ("multilingual-e5-base", "paraphrase-multilingual-MiniLM-L12-v2")
TOLERANCE = 0.01


def counts(base: dict, new: dict) -> dict:
    """Agreed, lost and gained reference first results on the sampled queries."""
    first = base["reference_first"]["queries"]
    a = np.array([f in r for f, r in zip(first, base["query_results"])], dtype=float)
    b = np.array([f in r for f, r in zip(first, new["query_results"])], dtype=float)
    n, agreed = len(a), int(a.sum())
    lost, gained = int((a * (1 - b)).sum()), int(((1 - a) * b).sum())
    lost_v = 1 - b[a == 1]
    ls = float(lost_v.mean()) if agreed else 0.0
    lse = float(lost_v.std(ddof=1) / math.sqrt(agreed)) if agreed > 1 else 0.0
    d = a - b
    nm = float(d.mean())
    nse = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0

    def z(m, s):
        return m / s if s > 0 else (0.0 if m == 0 else math.copysign(1e6, m))

    return {
        "compared": n,
        "agreed": agreed,
        "lost": lost,
        "gained": gained,
        "lost_share": ls,
        "lost_z": z(ls, lse),
        "net_drop": nm,
        "net_z": z(nm, nse),
    }


def _sample(x, q, ref=None):
    if ref is None:
        return result_sample(x, q, M, K)
    return result_sample(x, q, M, K, ref["docs"], ref["queries"])


def fault_rows(cache: Path, results: dict, work: Path) -> list[dict]:
    out = []
    for folder in ("fourth_faults", "fifth_faults", "sixth_faults"):
        recs = [json.loads(p.read_text()) for p in sorted((cache / folder).glob("*.json"))]
        recs = [r for r in recs if r["fault"] != "duplicates"]
        by_pipe: dict = {}
        for r in recs:
            by_pipe.setdefault((r["dataset"], r["model"], r["reference"]), []).append(r)
        for (name, model, ref), group in sorted(by_pipe.items()):
            paths = {r["fault"]: work / f"fault__{name}__{model}__{r['fault']}.json" for r in group}
            if all(p.exists() for p in paths.values()):
                out += [json.loads(p.read_text()) for p in paths.values()]
                continue
            z = np.load(cache / "emb" / f"{name}__{model}.npz")
            rf = np.load(cache / "emb" / f"{name}__{ref}.npz")
            base = _sample(z["docs"], z["queries"], rf)
            for r in group:
                other = (
                    _other_model(name, model, z["docs"].shape[1], cache, results)
                    if r["fault"] == "query_model"
                    else None
                )
                fd, fq = FAULTS[r["fault"]](
                    z["docs"], z["queries"], other, np.random.default_rng(SEED)
                )
                c = counts(base, _sample(fd, fq))
                rec = {
                    "kind": "fault",
                    "set": folder.split("_")[0],
                    "dataset": name,
                    "model": model,
                    "reference": ref,
                    "fault": r["fault"],
                    "ndcg_drop": r["ndcg_drop"],
                    "stored_lost": (r["top_hit"].get("queries") or {}).get("lost"),
                    **c,
                }
                paths[r["fault"]].write_text(json.dumps(rec))
                out.append(rec)
            print(f"{folder:<14} {name:<22} {model:<34} done", flush=True)
    return out


def change_rows(cache: Path, results: dict, work: Path, baseline: str) -> list[dict]:
    tag = "" if baseline == BASELINES[0] else f"{baseline}__"
    out = []
    for name in sorted({k[0] for k in results}):
        ref = "gte-multilingual-base" if role(name) == "fifth" else "bge-m3"
        if (name, baseline) not in results or (name, ref) not in results:
            continue
        cands = [
            m.short for m in MODELS if m.short not in (baseline, ref) and (name, m.short) in results
        ]
        paths = {m: work / f"change__{tag}{name}__{m}.json" for m in cands}
        if all(p.exists() for p in paths.values()):
            out += [json.loads(p.read_text()) for p in paths.values()]
            continue
        rf = np.load(cache / "emb" / f"{name}__{ref}.npz")
        z = np.load(cache / "emb" / f"{name}__{baseline}.npz")
        base = _sample(z["docs"], z["queries"], rf)
        for m, p in paths.items():
            if p.exists():
                out.append(json.loads(p.read_text()))
                continue
            c = np.load(cache / "emb" / f"{name}__{m}.npz")
            rec = {
                "kind": "change",
                "baseline": baseline,
                "set": role(name),
                "dataset": name,
                "model": m,
                "reference": ref,
                "languages": next(x.languages for x in MODELS if x.short == m),
                "ndcg_drop": results[(name, baseline)]["ndcg_at_10"]
                - results[(name, m)]["ndcg_at_10"],
                **counts(base, _sample(c["docs"], c["queries"])),
            }
            p.write_text(json.dumps(rec))
            out.append(rec)
        print(f"changes        {name:<22} {baseline} done", flush=True)
    return out


RULES = {
    "shipped: lost share > 0.05, z > 2": lambda c: (
        c["lost_share"] > 0.05 and c["lost_z"] > REFERENCE_Z
    ),
    "shipped and net drop > 0": lambda c: (
        c["lost_share"] > 0.05 and c["lost_z"] > REFERENCE_Z and c["net_drop"] > 0
    ),
    "shipped and net z > 1": lambda c: (
        c["lost_share"] > 0.05 and c["lost_z"] > REFERENCE_Z and c["net_z"] > 1
    ),
    "shipped and net z > 2": lambda c: (
        c["lost_share"] > 0.05 and c["lost_z"] > REFERENCE_Z and c["net_z"] > REFERENCE_Z
    ),
    "net drop > 0.03, net z > 2": lambda c: c["net_drop"] > 0.03 and c["net_z"] > REFERENCE_Z,
}


def rates(cases: list[dict]) -> dict:
    big = [c for c in cases if c["ndcg_drop"] > 0.02]
    quiet = [c for c in cases if c["ndcg_drop"] <= TOLERANCE]
    return {
        "n": len(cases),
        "drop_over_002": len(big),
        "harmless": len(quiet),
        **{
            name: {
                "flagged_of_drop_over_002": sum(map(rule, big)),
                "flagged_of_harmless": sum(map(rule, quiet)),
            }
            for name, rule in RULES.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", help="run.py cache directory")
    args = p.parse_args(argv)
    cache = out_dir(args.out)
    results = {}
    for f in (cache / "results").glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        results[(r["dataset"]["name"], r["model"]["short"])] = r
    work = cache / "net_design"
    work.mkdir(exist_ok=True)
    faults = fault_rows(cache, results, work)
    changes = [c for b in BASELINES for c in change_rows(cache, results, work, b)]
    for c in changes:
        c.setdefault("baseline", BASELINES[0])
    mismatch = sum(c["stored_lost"] is not None and c["stored_lost"] != c["lost"] for c in faults)
    summary = {
        "check_lost_matches_stored": {"faults": len(faults), "mismatch": mismatch},
        "faults": {
            s: rates([c for c in faults if c["set"] == s]) for s in ("fourth", "fifth", "sixth")
        },
        "faults_all": rates(faults),
        "changes": {
            "all": rates(changes),
            "multilingual": rates([c for c in changes if c["languages"] == "multi"]),
            "by_baseline": {
                b: rates([c for c in changes if c["baseline"] == b]) for b in BASELINES
            },
            "by_set": {
                s: rates([c for c in changes if c["set"] == s])
                for s in sorted({c["set"] for c in changes})
            },
        },
    }
    (HERE / "net_design.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
