"""Check the predictions of preregistration_third.md on the third held-out set.

Reads the run.py results of the datasets with ``role="third"`` and scores:

- P1, query_hubness against chance, on the pipelines where chance alone reaches warn
- P5, the overlap with a reference model's results as a label-free sign of harm, on six
  faults injected one at a time into the good pipelines with at most 25k documents
- P6, the labeled gate judged against the noise of 100 labeled queries
  (``retrieval_confidence``) against the mean alone
- P7, determinism: the same audit twice gives the same JSON

and reports the reference-model rule (``--reference``) without a prediction. Every number
comes from the vecscope library as shipped at the freeze: ``audit`` with
``reference_embeddings`` and ``result_sample``, ``diff_evidence`` with
``reference_tolerance`` and ``retrieval_confidence``. Nothing here is tuned.

The summary in third_summary.json was scored at commit 0d554ad. After it,
``reference_regressions`` changed from the drop of the whole overlap to the share of agreed
top hits lost (preregistration_fourth.md), so a re-run scores P5b with the new rule.

Run (after run.py on the third set):
    python examples/validation/third.py
Writes <cache>/third/*.json, <cache>/third_faults/*.json and
examples/validation/third_summary.json
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
from confirm import BAD, GOOD, REFERENCE, REFERENCE_ALT, _worst, audit_pair
from faults import FAULTS, _other_model
from registry import DATASETS, role
from run import SEED, load, out_dir, qrels_rows

from vecscope import audit
from vecscope.evidence import THRESHOLDS, WORSE, _confident_drop, diff_evidence

ROLE = "third"
MAX_DOCS = 25_000  # faults.py default
RESULT_SAMPLE = 200
TOLERANCE = 0.01  # nDCG@10 drop that counts as harmful
REFERENCE_TOLERANCE = 0.02
CONFIDENCE = 0.95
LABELED = 100
DRAWS = 200
ROW_FAULTS = {"duplicates"}  # faults in the rows themselves: the reference embeds them too


def _auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    p, n = np.array(pos)[:, None], np.array(neg)[None, :]
    return float(((p > n) + 0.5 * (p == n)).mean())


def fault_cases(cache: Path, results: dict, good: list[tuple[str, str]]) -> list[dict]:
    work = cache / "third_faults"
    work.mkdir(exist_ok=True)
    out = []
    loaded: dict = {}
    for name, model in good:
        if results[(name, model)]["dataset"]["n_docs"] > MAX_DOCS:
            continue
        files = {f: work / f"{name}__{model}__{f}.json" for f in FAULTS}
        if all(p.exists() for p in files.values()):
            out += [json.loads(p.read_text(encoding="utf-8")) for p in files.values()]
            continue
        if name not in loaded:
            d = load(DATASETS[name], cache)
            loaded = {name: qrels_rows(d[0], d[2], d[4])}
        qrels = loaded[name]
        ref = REFERENCE if model != REFERENCE else REFERENCE_ALT
        z = np.load(cache / "emb" / f"{name}__{model}.npz")
        r = np.load(cache / "emb" / f"{name}__{ref}.npz")
        kw = {"k": 10, "qrels": qrels, "result_sample": RESULT_SAMPLE}
        clean = audit(
            z["docs"], queries=z["queries"], reference_embeddings=r["docs"],
            reference_queries=r["queries"], **kw,
        ).to_dict()  # fmt: skip
        for fault, path in files.items():
            if path.exists():
                out.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            other = (
                _other_model(name, model, z["docs"].shape[1], cache, results)
                if fault == "query_model"
                else None
            )
            if fault == "query_model" and other is None:
                continue
            fd, fq = FAULTS[fault](z["docs"], z["queries"], other, np.random.default_rng(SEED))
            rd = (
                FAULTS[fault](r["docs"], r["queries"], None, np.random.default_rng(SEED))[0]
                if fault in ROW_FAULTS
                else r["docs"]
            )
            ev = audit(
                fd, queries=fq, reference_embeddings=rd, reference_queries=r["queries"], **kw
            ).to_dict()
            d = diff_evidence(
                ev,
                clean,
                retrieval_tolerances={"ndcg_at_k": TOLERANCE},
                retrieval_confidence=CONFIDENCE,
                reference_tolerance=REFERENCE_TOLERANCE,
            )
            ag = d["result_agreement"]
            rec = {
                "dataset": name,
                "model": model,
                "reference": ref,
                "fault": fault,
                "ndcg_drop": clean["retrieval"]["exact"]["ndcg_at_k"]
                - ev["retrieval"]["exact"]["ndcg_at_k"],
                "changed_share": {s: v["changed_share"] for s, v in ag.items()},
                "reference_overlap": {s: v.get("reference") for s, v in ag.items()},
                "reference_regressions": d["reference_regressions"],
                "query_rows": clean["retrieval"]["per_query"]["query_rows"],
                "clean_ndcg": clean["retrieval"]["per_query"]["exact"]["ndcg_at_k"],
                "faulted_ndcg": ev["retrieval"]["per_query"]["exact"]["ndcg_at_k"],
            }
            path.write_text(json.dumps(rec), encoding="utf-8")
            out.append(rec)
        print(f"{name:<22} {model:<38} faults done", flush=True)
    return out


def _block(rows: list[int], vals: dict) -> dict:
    return {"per_query": {"query_rows": rows, "exact": {"ndcg_at_k": [vals[q] for q in rows]}}}


def label_gate(cases: list[dict]) -> dict:
    """P6: the mean rule against the confidence rule on seeded subsets of 100 judged queries."""
    rng = np.random.default_rng(SEED)
    tally = {
        rule: {k: {"fail": 0.0, "inconclusive": 0.0, "pass": 0.0} for k in ("harmful", "harmless")}
        for rule in ("mean", "confidence")
    }
    n = {"harmful": 0, "harmless": 0}
    for c in cases:
        qs = c["query_rows"]
        if len(qs) < LABELED:
            continue
        clean = dict(zip(qs, c["clean_ndcg"]))
        bad = dict(zip(qs, c["faulted_ndcg"]))
        kind = "harmful" if c["ndcg_drop"] > TOLERANCE else "harmless"
        n[kind] += 1
        for _ in range(DRAWS):
            sub = sorted(rng.choice(qs, LABELED, replace=False).tolist())
            mean_drop = float(np.mean([clean[q] - bad[q] for q in sub]))
            tally["mean"][kind]["fail" if mean_drop > TOLERANCE else "pass"] += 1 / DRAWS
            cd = _confident_drop(
                "exact.ndcg_at_k", TOLERANCE, _block(sub, bad), _block(sub, clean), CONFIDENCE
            )
            tally["confidence"][kind][cd["decision"]] += 1 / DRAWS
    share = {
        rule: {k: {d: (v / n[k] if n[k] else None) for d, v in t[k].items()} for k in t}
        for rule, t in tally.items()
    }
    return {"cases": n, "share": share}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", help="run.py cache directory")
    args = p.parse_args(argv)
    cache = out_dir(args.out)
    results = {}
    for f in (cache / "results").glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        results[(r["dataset"]["name"], r["model"]["short"])] = r
    recs = {k: r for k, r in results.items() if role(k[0]) == ROLE}
    best: dict[str, float] = defaultdict(float)
    for (name, _), r in recs.items():
        best[name] = max(best[name], r["ndcg_at_10"])
    work = cache / "third"
    work.mkdir(exist_ok=True)
    rows = []
    for (name, model), r in sorted(recs.items()):
        share = r["ndcg_at_10"] / best[name] if best[name] else 0.0
        label = "good" if share >= GOOD else "bad" if share <= BAD else "middle"
        f = work / f"{name}__{model}.json"
        if f.exists():
            a = json.loads(f.read_text(encoding="utf-8"))
        else:
            a = audit_pair(cache, name, model)
            f.write_text(json.dumps(a, indent=1), encoding="utf-8")
            print(f"{name:<22} {model:<38} {label:<6} audited", flush=True)
        rows.append({"dataset": name, "model": model, "label": label, **a})
    good = [x for x in rows if x["label"] == "good"]
    bad = [x for x in rows if x["label"] == "bad"]
    summary: dict = {"pipelines": len(rows), "good": len(good), "bad": len(bad)}

    # P1: query_hubness against chance, where chance alone reaches warn
    warn_at = THRESHOLDS["queries.query_hubness_skewness"][0]

    def qh(x, key):
        return WORSE[x["findings"]["query_hubness"][key]]

    chance = [x for x in rows if x["findings"]["query_hubness"]["null_skewness"] >= warn_at]
    cg = [x for x in chance if x["label"] == "good"]
    cb = [x for x in chance if x["label"] == "bad"]
    p1 = {
        "datasets": sorted(
            {
                (x["dataset"], round(x["findings"]["query_hubness"]["null_skewness"], 2))
                for x in chance
            }
        ),
        "good": len(cg),
        "good_flagged_threshold_only": sum(qh(x, "threshold_only") >= 1 for x in cg),
        "good_flagged_rule": sum(qh(x, "without") >= 1 for x in cg),
        "bad": len(cb),
        "bad_fail_threshold_only": sum(qh(x, "threshold_only") == 2 for x in cb),
        "bad_fail_kept_by_rule": sum(
            qh(x, "threshold_only") == 2 and qh(x, "without") == 2 for x in cb
        ),
    }
    p1["testable_a"] = p1["good_flagged_threshold_only"] >= 5
    p1["testable_b"] = p1["bad_fail_threshold_only"] >= 5
    p1["pass_a"] = (
        p1["good_flagged_rule"] <= 0.5 * p1["good_flagged_threshold_only"]
        if p1["testable_a"]
        else None
    )
    p1["pass_b"] = (
        p1["bad_fail_kept_by_rule"] >= 0.9 * p1["bad_fail_threshold_only"]
        if p1["testable_b"]
        else None
    )
    summary["P1_query_hubness_null"] = p1

    # reported without a prediction: the reference rule on this set
    def count(group, key, level):
        return sum(WORSE[_worst(x["findings"], key)] >= level for x in group)

    summary["reference_rule"] = {
        "good_fail_without": count(good, "without", 2),
        "good_fail_with": count(good, "with_reference", 2),
        "good_flagged_without": count(good, "without", 1),
        "good_flagged_with": count(good, "with_reference", 1),
        "bad_fail_without": count(bad, "without", 2),
        "bad_fail_with": count(bad, "with_reference", 2),
        "bad_flagged_with": count(bad, "with_reference", 1),
        "good_fail_left": sorted(
            f"{x['dataset']} {x['model']}: "
            + ", ".join(n for n, v in x["findings"].items() if v["with_reference"] == "fail")
            for x in good
            if _worst(x["findings"], "with_reference") == "fail"
        ),
    }

    # P5, P6 on injected faults
    good_pairs = sorted((x["dataset"], x["model"]) for x in good)
    cases = fault_cases(cache, results, good_pairs)
    nodup = [c for c in cases if c["fault"] not in ROW_FAULTS]

    def z(c):
        r = c["reference_overlap"].get("queries")
        return r["z"] if r else None

    harm = [c for c in nodup if c["ndcg_drop"] > TOLERANCE and z(c) is not None]
    harmless = [c for c in nodup if c["ndcg_drop"] <= TOLERANCE and z(c) is not None]
    big = [c for c in nodup if c["ndcg_drop"] > 0.02]
    auc_ref = _auc([z(c) for c in harm], [z(c) for c in harmless])
    auc_changed = _auc(
        [c["changed_share"]["queries"] for c in harm],
        [c["changed_share"]["queries"] for c in harmless],
    )
    flagged = [c for c in nodup if c["reference_regressions"]]
    p5 = {
        "faults": len(cases),
        "without_duplicates": len(nodup),
        "harmful": len(harm),
        "harmless": len(harmless),
        "auc_reference_z": auc_ref,
        "auc_changed_share": auc_changed,
        "flagged_of_drop_over_002": (sum(c in flagged for c in big), len(big)),
        "flagged_of_harmless": (sum(c in flagged for c in harmless), len(harmless)),
        "by_fault": {
            f: {
                "harmful": sum(c["fault"] == f and c["ndcg_drop"] > TOLERANCE for c in cases),
                "flagged": sum(c["fault"] == f and bool(c["reference_regressions"]) for c in cases),
                "of": sum(c["fault"] == f for c in cases),
            }
            for f in FAULTS
        },
    }
    p5["testable"] = len(harmless) >= 10 and len(big) >= 10
    if p5["testable"]:
        caught = p5["flagged_of_drop_over_002"][0] / len(big)
        alarm = p5["flagged_of_harmless"][0] / len(harmless)
        p5["pass_a"] = (
            auc_ref is not None and auc_ref >= 0.80 and auc_ref >= (auc_changed or 0) + 0.10
        )
        p5["pass_b"] = caught >= 0.70 and alarm <= 0.35
    summary["P5_reference_overlap"] = p5

    p6 = label_gate(cases)
    s = p6["share"]
    p6["testable"] = p6["cases"]["harmful"] >= 10 and p6["cases"]["harmless"] >= 10
    if p6["testable"]:
        miss_mean = s["mean"]["harmful"]["pass"]
        miss_conf = s["confidence"]["harmful"]["pass"]
        alarm_mean = s["mean"]["harmless"]["fail"]
        alarm_conf = s["confidence"]["harmless"]["fail"]
        p6["pass_a"] = miss_conf <= 0.03 and miss_conf <= 0.5 * miss_mean
        p6["pass_b"] = alarm_conf <= 0.75 * alarm_mean
    summary["P6_labeled_confidence"] = p6

    # P7: the same audit twice gives the same JSON
    reruns = []
    for name in sorted({c["dataset"] for c in cases}):
        model = next(c["model"] for c in cases if c["dataset"] == name)
        ref = REFERENCE if model != REFERENCE else REFERENCE_ALT
        d = load(DATASETS[name], cache)
        qrels = qrels_rows(d[0], d[2], d[4])
        z_ = np.load(cache / "emb" / f"{name}__{model}.npz")
        r_ = np.load(cache / "emb" / f"{name}__{ref}.npz")

        def once(z_=z_, r_=r_, qrels=qrels):
            return json.dumps(
                audit(
                    z_["docs"], k=10, queries=z_["queries"], qrels=qrels,
                    reference_embeddings=r_["docs"], reference_queries=r_["queries"],
                    result_sample=RESULT_SAMPLE,
                ).to_dict(),
                sort_keys=True,
            )  # fmt: skip

        reruns.append({"dataset": name, "model": model, "identical": once() == once()})
    summary["P7_determinism"] = {"reruns": reruns, "pass": all(r["identical"] for r in reruns)}

    (HERE / "third_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False)[:8000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
