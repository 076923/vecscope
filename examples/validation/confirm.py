"""Check the preregistered predictions (preregistration.md) on the second held-out set.

Reads the run.py results of the datasets with ``role="confirm"``, audits every pair again with
a reference model (bge-m3; multilingual-e5-base for bge-m3 itself), reads the faults.py and
label_budget.py output for the same datasets, and scores P1 to P4 exactly as written in
preregistration.md. Nothing here is tuned: the thresholds, the reference rule and the
criteria were committed before any of this data was embedded.

Run (after run.py, faults.py --role confirm, label_budget.py --role confirm):
    python examples/validation/confirm.py
Writes <cache>/confirm/<dataset>__<model>.json and examples/validation/confirm_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from registry import DATASETS, role
from run import load, out_dir, qrels_rows

from vecscope import audit
from vecscope.evidence import THRESHOLDS, WORSE, diff_evidence

GOOD, BAD = 0.9, 0.5
REFERENCE, REFERENCE_ALT = "bge-m3", "multilingual-e5-base"
SAMPLE = 100_000  # run.py --sample default


def _plain_verdict(value: float, key: str) -> str:
    warn, fail = THRESHOLDS[key]
    return "fail" if value >= fail else "warn" if value >= warn else "ok"


def audit_pair(cache: Path, name: str, model: str) -> dict:
    z = np.load(cache / "emb" / f"{name}__{model}.npz")
    ref = REFERENCE if model != REFERENCE else REFERENCE_ALT
    r = np.load(cache / "emb" / f"{name}__{ref}.npz")
    ev = audit(
        z["docs"],
        k=10,
        queries=z["queries"],
        sample=SAMPLE,
        reference_embeddings=r["docs"],
        reference_queries=r["queries"],
    )
    out = {"reference": ref, "explained": ev.reference["explained"], "findings": {}}
    for f in ev.findings:
        before = (f.details.get("reference") or {}).get("verdict_without_reference", f.verdict)
        out["findings"][f.name] = {"value": f.value, "with_reference": f.verdict, "without": before}
        if f.name == "query_hubness":
            out["findings"][f.name]["threshold_only"] = _plain_verdict(
                f.value, "queries.query_hubness_skewness"
            )
            out["findings"][f.name]["null_skewness"] = f.details["null_skewness"]
    return out


def _worst(fs: dict, key: str) -> str:
    return max((v[key] for v in fs.values()), key=WORSE.get, default="ok")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", help="run.py cache directory")
    args = p.parse_args(argv)
    cache = out_dir(args.out)
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in (cache / "results").glob("*.json")]
    recs = [r for r in recs if role(r["dataset"]["name"]) == "confirm"]
    best: dict[str, float] = defaultdict(float)
    for r in recs:
        best[r["dataset"]["name"]] = max(best[r["dataset"]["name"]], r["ndcg_at_10"])
    work = cache / "confirm"
    work.mkdir(exist_ok=True)
    rows = []
    for r in sorted(recs, key=lambda r: (r["dataset"]["name"], r["model"]["short"])):
        name, model = r["dataset"]["name"], r["model"]["short"]
        share = r["ndcg_at_10"] / best[name] if best[name] else 0.0
        label = "good" if share >= GOOD else "bad" if share <= BAD else "middle"
        f = work / f"{name}__{model}.json"
        if f.exists():
            a = json.loads(f.read_text(encoding="utf-8"))
        else:
            a = audit_pair(cache, name, model)
            f.write_text(json.dumps(a, indent=1), encoding="utf-8")
            print(f"{name:<24} {model:<38} {label:<6} explained {a['explained']}", flush=True)
        rows.append({"dataset": name, "model": model, "label": label, "ndcg": r["ndcg_at_10"], **a})

    def count(label, key, level):
        return sum(
            1 for x in rows if x["label"] == label and WORSE[_worst(x["findings"], key)] >= level
        )

    good = [x for x in rows if x["label"] == "good"]
    bad = [x for x in rows if x["label"] == "bad"]
    summary: dict = {"pipelines": len(rows), "good": len(good), "bad": len(bad)}

    # P1 query_hubness against chance
    def qh(x, key):
        return WORSE[x["findings"]["query_hubness"][key]]

    p1 = {
        "good_flagged_rule": sum(qh(x, "without") >= 1 for x in good),
        "good_flagged_threshold_only": sum(qh(x, "threshold_only") >= 1 for x in good),
        "bad_fail_threshold_only": sum(qh(x, "threshold_only") == 2 for x in bad),
        "bad_fail_kept_by_rule": sum(
            qh(x, "threshold_only") == 2 and qh(x, "without") == 2 for x in bad
        ),
    }
    p1["pass"] = p1["good_flagged_rule"] <= p1["good_flagged_threshold_only"] and (
        p1["bad_fail_kept_by_rule"] >= 0.9 * p1["bad_fail_threshold_only"]
    )
    summary["P1_query_hubness_null"] = p1

    # P2 reference model
    p2 = {
        "good_fail_without": count("good", "without", 2),
        "good_fail_with": count("good", "with_reference", 2),
        "good_flagged_without": count("good", "without", 1),
        "good_flagged_with": count("good", "with_reference", 1),
        "good_all_ok_without": len(good) - count("good", "without", 1),
        "good_all_ok_with": len(good) - count("good", "with_reference", 1),
        "bad_fail_without": count("bad", "without", 2),
        "bad_fail_with": count("bad", "with_reference", 2),
        "bad_flagged_without": count("bad", "without", 1),
        "bad_flagged_with": count("bad", "with_reference", 1),
    }
    p2["pass_good"] = p2["good_fail_with"] <= 0.75 * p2["good_fail_without"]
    p2["pass_bad"] = p2["bad_fail_with"] >= 0.95 * p2["bad_fail_without"]
    explained = defaultdict(lambda: defaultdict(int))
    for x in rows:
        for n in x["explained"]:
            explained[x["label"]][n] += 1
    p2["explained"] = {k: dict(v) for k, v in explained.items()}
    p2["good_fail_left"] = sorted(
        f"{x['dataset']} {x['model']}: "
        + ", ".join(n for n, v in x["findings"].items() if v["with_reference"] == "fail")
        for x in good
        if _worst(x["findings"], "with_reference") == "fail"
    )
    p2["bad_not_flagged_with"] = sorted(
        f"{x['dataset']} {x['model']}"
        for x in bad
        if _worst(x["findings"], "with_reference") == "ok"
    )
    summary["P2_reference_model"] = p2

    # P3 labeled-query budget
    lb = cache / "label_budget_confirm.json"
    if lb.exists():
        t = {r["labeled_queries"]: r for r in json.loads(lb.read_text())["table"]}
        row = t.get(100)
        p3 = {"table": list(t.values())}
        if row and row["caught"] is not None:
            big = [row["caught_by_drop"][b] for b in ("0.02-0.05", ">0.05")]
            n_big = sum(m for _, m in big)
            caught_big = sum((h or 0) * m for h, m in big) / n_big if n_big else None
            p3.update(
                caught_at_100=row["caught"],
                caught_over_002_at_100=caught_big,
                false_alarm_at_100=row["false_alarm"],
                harmful_cases=row["harmful_cases"],
            )
            p3["pass"] = row["caught"] >= 0.85 and (caught_big is None or caught_big >= 0.9)
        summary["P3_label_budget"] = p3

    # P4 label-free result change
    recs_f = [
        json.loads(f.read_text(encoding="utf-8")) for f in sorted((cache / "faults").glob("*.json"))
    ]
    recs_f = [r for r in recs_f if role(r["dataset"]) == "confirm" and "result_agreement" in r]
    changed = [
        r
        for r in recs_f
        if any(s["changed_share"] > 0 for s in (r["result_agreement"] or {}).values())
    ]
    by_fault = defaultdict(lambda: [0, 0])
    for r in recs_f:
        by_fault[r["fault"]][1] += 1
        by_fault[r["fault"]][0] += r in changed
    rerun = []
    for name in sorted({x["dataset"] for x in good}):
        model = next(x["model"] for x in good if x["dataset"] == name)
        d = load(DATASETS[name], cache)
        if len(d[0]) > 25_000:  # faults.py --max-docs default
            continue
        z = np.load(cache / "emb" / f"{name}__{model}.npz")
        qrels = qrels_rows(d[0], d[2], d[4])
        a = audit(z["docs"], k=10, queries=z["queries"], qrels=qrels, result_sample=200).to_dict()
        b = audit(z["docs"], k=10, queries=z["queries"], qrels=qrels, result_sample=200).to_dict()
        agree = diff_evidence(a, b, result_tolerance=0.0)["result_agreement"]
        rerun.append(
            {"dataset": name, "model": model, **{s: v["changed_share"] for s, v in agree.items()}}
        )
    p4 = {
        "faults": len(recs_f),
        "changed": len(changed),
        "by_fault": {k: {"changed": v[0], "of": v[1]} for k, v in by_fault.items()},
        "reruns": rerun,
    }
    p4["pass"] = (
        bool(recs_f)
        and len(changed) == len(recs_f)
        and all(v == 0 for r in rerun for k, v in r.items() if k in ("queries", "rows"))
    )
    summary["P4_result_change"] = p4

    (HERE / "confirm_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items()}, indent=1, ensure_ascii=False)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
