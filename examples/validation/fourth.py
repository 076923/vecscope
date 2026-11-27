"""Check the predictions of preregistration_fourth.md on the fourth held-out set.

Reads the run.py results of the datasets with ``role="fourth"`` and scores:

- P8, the agreed top hits as a label-free sign of harm (``reference_top_hit``,
  ``reference_regressions`` at 0.05), on six faults injected one at a time into the good
  pipelines with at most 25k documents, against the overlap-drop rule the third set tested
- P9, lowering hub findings at fail when the results agree with the reference
  (``reference.agreement``): on good and bad pipelines, and on the injected faults
- P10, determinism: the same audit twice gives the same JSON

and reports without a prediction: the query_hubness chance rule, the labeled gate judged
against noise, and the reference rule for findings. Every number comes from the vecscope
library as shipped at the freeze. Nothing here is tuned.

Run (after run.py on the fourth set):
    python examples/validation/fourth.py
Writes <cache>/fourth/*.json, <cache>/fourth_faults/*.json and
examples/validation/fourth_summary.json
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
from confirm import BAD, GOOD, REFERENCE, REFERENCE_ALT, SAMPLE, _plain_verdict
from faults import FAULTS, _other_model
from registry import DATASETS, role
from run import SEED, load, out_dir, qrels_rows
from third import _auc, label_gate

from vecscope import audit
from vecscope.evidence import THRESHOLDS, WORSE, diff_evidence
from vecscope.results import REFERENCE_Z

ROLE = "fourth"
MAX_DOCS = 25_000  # faults.py default
RESULT_SAMPLE = 200
TOLERANCE = 0.01  # nDCG@10 drop that counts as harmful
REFERENCE_TOLERANCE = 0.05  # the shipped default for agreed top hits
OLD_TOLERANCE = 0.02  # the overlap-drop rule the third set tested (P5b)
CONFIDENCE = 0.95
ROW_FAULTS = {"duplicates"}  # faults in the rows themselves: the reference embeds them too


def _verdicts(ev) -> dict:
    """Per finding: the verdict without the reference, with the reference before the
    agreement rule, and the final verdict."""
    out = {}
    for f in ev.findings:
        ref = f.details.get("reference") or {}
        out[f.name] = {
            "value": f.value,
            "without": ref.get("verdict_without_reference", f.verdict),
            "before_agreement": ref.get("verdict_before_agreement", f.verdict),
            "final": f.verdict,
        }
    return out


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
    out = {
        "reference": ref,
        "explained": ev.reference["explained"],
        "agreement": ev.reference.get("agreement"),
        "findings": _verdicts(ev),
    }
    qh = next(f for f in ev.findings if f.name == "query_hubness")
    out["findings"]["query_hubness"]["threshold_only"] = _plain_verdict(
        qh.value, "queries.query_hubness_skewness"
    )
    out["findings"]["query_hubness"]["null_skewness"] = qh.details["null_skewness"]
    return out


def _worst(fs: dict, key: str) -> str:
    return max((v[key] for v in fs.values()), key=WORSE.get, default="ok")


def _caught(clean: dict, faulted: dict, key: str) -> bool:
    """The findings catch a fault: the run is at fail where the clean one was not, or a
    finding is worse than on the clean run (the baseline diff)."""
    if _worst(faulted, key) == "fail" and _worst(clean, key) != "fail":
        return True
    return any(WORSE[faulted[n][key]] > WORSE[clean[n][key]] for n in faulted if n in clean)


def fault_cases(cache: Path, results: dict, good: list[tuple[str, str]]) -> list[dict]:
    work = cache / "fourth_faults"
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
        clean_ev = audit(
            z["docs"], queries=z["queries"], reference_embeddings=r["docs"],
            reference_queries=r["queries"], **kw,
        )  # fmt: skip
        clean, clean_v = clean_ev.to_dict(), _verdicts(clean_ev)
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
            ev_ = audit(
                fd, queries=fq, reference_embeddings=rd, reference_queries=r["queries"], **kw
            )
            ev, ev_v = ev_.to_dict(), _verdicts(ev_)
            d = diff_evidence(
                ev,
                clean,
                retrieval_tolerances={"ndcg_at_k": TOLERANCE},
                retrieval_confidence=CONFIDENCE,
                reference_tolerance=REFERENCE_TOLERANCE,
            )
            ag = d["result_agreement"]
            old = (ag.get("queries") or {}).get("reference")
            rec = {
                "dataset": name,
                "model": model,
                "reference": ref,
                "fault": fault,
                "ndcg_drop": clean["retrieval"]["exact"]["ndcg_at_k"]
                - ev["retrieval"]["exact"]["ndcg_at_k"],
                "changed_share": {s: v["changed_share"] for s, v in ag.items()},
                "top_hit": {s: v.get("reference_top_hit") for s, v in ag.items()},
                "overlap_drop": {s: v.get("reference") for s, v in ag.items()},
                "reference_regressions": d["reference_regressions"],
                "old_rule": bool(old and old["drop"] > OLD_TOLERANCE and old["z"] > REFERENCE_Z),
                "agreement": ev_.reference.get("agreement"),
                "caught_before_agreement": _caught(clean_v, ev_v, "before_agreement"),
                "caught_with_agreement": _caught(clean_v, ev_v, "final"),
                "query_rows": clean["retrieval"]["per_query"]["query_rows"],
                "clean_ndcg": clean["retrieval"]["per_query"]["exact"]["ndcg_at_k"],
                "faulted_ndcg": ev["retrieval"]["per_query"]["exact"]["ndcg_at_k"],
            }
            path.write_text(json.dumps(rec), encoding="utf-8")
            out.append(rec)
        print(f"{name:<22} {model:<38} faults done", flush=True)
    return out


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
    work = cache / "fourth"
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

    # P9 (a, b): the agreement rule on good and bad pipelines at fail with the reference
    gf = [x for x in good if _worst(x["findings"], "before_agreement") == "fail"]
    bf = [x for x in bad if _worst(x["findings"], "before_agreement") == "fail"]
    p9 = {
        "good_fail_with_reference": len(gf),
        "good_cleared": sum(_worst(x["findings"], "final") != "fail" for x in gf),
        "bad_fail_with_reference": len(bf),
        "bad_kept": sum(_worst(x["findings"], "final") == "fail" for x in bf),
        "good_fail_left": sorted(
            f"{x['dataset']} {x['model']} (agreement "
            f"{(x['agreement'] or {}).get('queries', float('nan')):.2f}): "
            + ", ".join(n for n, v in x["findings"].items() if v["final"] == "fail")
            for x in gf
            if _worst(x["findings"], "final") == "fail"
        ),
        "agreement_good": sorted(
            round(x["agreement"]["queries"], 3) for x in good if x.get("agreement")
        ),
        "agreement_bad": sorted(
            round(x["agreement"]["queries"], 3) for x in bad if x.get("agreement")
        ),
    }
    ag_g = [x["agreement"]["queries"] for x in good if x.get("agreement")]
    ag_b = [x["agreement"]["queries"] for x in bad if x.get("agreement")]
    p9["auc_agreement_good_vs_bad"] = _auc(ag_g, ag_b)
    p9["testable_a"] = len(gf) >= 4
    p9["testable_b"] = len(bf) >= 10
    p9["pass_a"] = p9["good_cleared"] >= 0.5 * len(gf) if p9["testable_a"] else None
    p9["pass_b"] = p9["bad_kept"] >= 0.95 * len(bf) if p9["testable_b"] else None

    # reported without a prediction: the chance rule and the reference rule
    warn_at = THRESHOLDS["queries.query_hubness_skewness"][0]
    chance = [x for x in rows if x["findings"]["query_hubness"]["null_skewness"] >= warn_at]
    summary["query_hubness_chance"] = {
        "datasets": sorted(
            {
                (x["dataset"], round(x["findings"]["query_hubness"]["null_skewness"], 2))
                for x in chance
            }
        ),
        "good_flagged_threshold_only": sum(
            WORSE[x["findings"]["query_hubness"]["threshold_only"]] >= 1
            for x in chance
            if x["label"] == "good"
        ),
        "good_flagged_rule": sum(
            WORSE[x["findings"]["query_hubness"]["without"]] >= 1
            for x in chance
            if x["label"] == "good"
        ),
        "bad_fail_threshold_only": sum(
            x["findings"]["query_hubness"]["threshold_only"] == "fail"
            for x in chance
            if x["label"] == "bad"
        ),
        "bad_fail_rule": sum(
            x["findings"]["query_hubness"]["without"] == "fail"
            for x in chance
            if x["label"] == "bad"
        ),
    }

    def count(group, key, level):
        return sum(WORSE[_worst(x["findings"], key)] >= level for x in group)

    summary["reference_rule"] = {
        k: {
            "good_fail": count(good, k, 2),
            "good_flagged": count(good, k, 1),
            "bad_fail": count(bad, k, 2),
            "bad_flagged": count(bad, k, 1),
        }
        for k in ("without", "before_agreement", "final")
    }

    # P8 and P9 (c) on injected faults
    good_pairs = sorted((x["dataset"], x["model"]) for x in good)
    cases = fault_cases(cache, results, good_pairs)
    nodup = [c for c in cases if c["fault"] not in ROW_FAULTS]

    def hit(c):
        return (c["top_hit"].get("queries") or {}).get("z")

    def drop(c):
        return (c["overlap_drop"].get("queries") or {}).get("z")

    harm = [c for c in nodup if c["ndcg_drop"] > TOLERANCE and hit(c) is not None]
    harmless = [c for c in nodup if c["ndcg_drop"] <= TOLERANCE and hit(c) is not None]
    big = [c for c in nodup if c["ndcg_drop"] > 0.02 and hit(c) is not None]
    p8 = {
        "faults": len(cases),
        "without_duplicates": len(nodup),
        "harmful": len(harm),
        "harmless": len(harmless),
        "drop_over_002": len(big),
        "auc_top_hit_z": _auc([hit(c) for c in harm], [hit(c) for c in harmless]),
        "auc_overlap_drop_z": _auc(
            [drop(c) for c in harm if drop(c) is not None],
            [drop(c) for c in harmless if drop(c) is not None],
        ),
        "auc_changed_share": _auc(
            [c["changed_share"]["queries"] for c in harm],
            [c["changed_share"]["queries"] for c in harmless],
        ),
        "flagged_of_drop_over_002": sum(bool(c["reference_regressions"]) for c in big),
        "flagged_of_harmless": sum(bool(c["reference_regressions"]) for c in harmless),
        "old_rule_flagged_of_drop_over_002": sum(c["old_rule"] for c in big),
        "old_rule_flagged_of_harmless": sum(c["old_rule"] for c in harmless),
        "by_fault": {
            f: {
                "harmful": sum(c["fault"] == f and c["ndcg_drop"] > TOLERANCE for c in cases),
                "flagged": sum(c["fault"] == f and bool(c["reference_regressions"]) for c in cases),
                "harmless_flagged": sum(
                    c["fault"] == f
                    and c["ndcg_drop"] <= TOLERANCE
                    and bool(c["reference_regressions"])
                    for c in cases
                ),
                "of": sum(c["fault"] == f for c in cases),
            }
            for f in FAULTS
        },
    }
    p8["testable"] = len(harmless) >= 10 and len(big) >= 10
    if p8["testable"]:
        caught = p8["flagged_of_drop_over_002"] / len(big)
        alarm = p8["flagged_of_harmless"] / len(harmless)
        old_alarm = p8["old_rule_flagged_of_harmless"] / len(harmless)
        p8["pass_a"] = p8["auc_top_hit_z"] is not None and p8["auc_top_hit_z"] >= 0.85
        p8["pass_b"] = caught >= 0.65 and alarm <= 0.15
        p8["pass_c"] = alarm <= 0.5 * old_alarm if old_alarm > 0 else alarm == 0
    summary["P8_agreed_top_hits"] = p8

    harmful_all = [c for c in cases if c["ndcg_drop"] > TOLERANCE]
    before = [c for c in harmful_all if c["caught_before_agreement"]]
    p9["faults_harmful"] = len(harmful_all)
    p9["faults_caught_before_agreement"] = len(before)
    p9["faults_still_caught"] = sum(c["caught_with_agreement"] for c in before)
    p9["testable_c"] = len(before) >= 10
    p9["pass_c"] = p9["faults_still_caught"] >= 0.95 * len(before) if p9["testable_c"] else None
    summary["P9_agreement_rule"] = p9

    summary["labeled_confidence"] = label_gate(cases)

    # P10: the same audit twice gives the same JSON
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
    summary["P10_determinism"] = {"reruns": reruns, "pass": all(r["identical"] for r in reruns)}

    (HERE / "fourth_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False)[:8000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
