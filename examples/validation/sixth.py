"""Check the predictions of preregistration_sixth.md on the sixth held-out set.

No rule changed after the fifth set. Reads the run.py results of the datasets with
``role="sixth"`` and scores, with bge-m3 as the reference model (gte-multilingual-base for
bge-m3 itself):

- P15, the agreed top hits and the review tier on the faults injected into the corpora with
  at most 25k documents (the criteria of P11b and P12, and P11a)
- P16, the same criteria on the faults injected into the two corpora over 25k documents
- P17, the two lowering rules on good and bad pipelines and on the faults (P13)
- P18, a model change: multilingual-e5-base, the baseline, replaced by each other model that
  is not the reference, judged by the agreed top hits and by 100 labeled queries against the
  full labels
- P19, determinism

with a 95% interval on each count (intervals.py). Every number comes from the vecscope
library as shipped at the freeze. Nothing here is tuned.

Run (after run.py on the sixth set):
    python examples/validation/sixth.py
Writes <cache>/sixth/*.json, <cache>/sixth_faults/*.json, <cache>/sixth_changes/*.json and
examples/validation/sixth_summary.json
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
from confirm import BAD, GOOD, SAMPLE, _plain_verdict
from faults import FAULTS, _other_model
from intervals import count_interval, fault_intervals
from registry import DATASETS, MODELS, role
from run import SEED, load, out_dir, qrels_rows
from third import _auc, label_gate

from vecscope import audit
from vecscope.evidence import THRESHOLDS, WORSE, diff_evidence
from vecscope.reference import PARTLY_EXPLAINED
from vecscope.results import REFERENCE_Z

ROLE = "sixth"
# the reference of sets 2 to 4; gte-multilingual-base for the reference model itself
REFERENCE, REFERENCE_ALT = "bge-m3", "gte-multilingual-base"
SMALL = 25_000  # the limit of the faults of the earlier sets
MAX_DOCS = 100_000  # faults go into every corpus of this set
BASELINE = "multilingual-e5-base"  # P18: the pipeline a team runs before a model change
RESULT_SAMPLE = 200
TOLERANCE = 0.01  # nDCG@10 drop that counts as harmful
REFERENCE_TOLERANCE = 0.05  # the shipped default for agreed top hits
OLD_TOLERANCE = 0.02  # the overlap-drop rule the third set tested (P5b)
CONFIDENCE = 0.95
ROW_FAULTS = {"duplicates"}  # faults in the rows themselves: the reference embeds them too


def _verdicts(ev) -> dict:
    """Per finding: the verdict without the reference, with the reference before the two
    lowering rules (agreement for hub findings, one step for effective_rank), and the final
    verdict."""
    out = {}
    for f in ev.findings:
        ref = f.details.get("reference") or {}
        without = ref.get("verdict_without_reference", f.verdict)
        before = ref.get("verdict_before_agreement", f.verdict)
        one_step = (
            f.name in PARTLY_EXPLAINED
            and without == "fail"
            and ref.get("verdict") == "warn"
            and f.verdict == "warn"
        )
        out[f.name] = {
            "value": f.value,
            "without": without,
            "before_rules": "fail" if one_step else before,
            "lowered_by": "agreement"
            if "verdict_before_agreement" in ref
            else "one_step"
            if one_step
            else None,
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
    work = cache / "sixth_faults"
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
                "n_docs": results[(name, model)]["dataset"]["n_docs"],
                "reference": ref,
                "fault": fault,
                "ndcg_drop": clean["retrieval"]["exact"]["ndcg_at_k"]
                - ev["retrieval"]["exact"]["ndcg_at_k"],
                "changed_share": {s: v["changed_share"] for s, v in ag.items()},
                "top_hit": {s: v.get("reference_top_hit") for s, v in ag.items()},
                "overlap_drop": {s: v.get("reference") for s, v in ag.items()},
                "reference_regressions": d["reference_regressions"],
                "reference_review": d["reference_review"],
                "old_rule": bool(old and old["drop"] > OLD_TOLERANCE and old["z"] > REFERENCE_Z),
                "agreement": ev_.reference.get("agreement"),
                "caught_before_rules": _caught(clean_v, ev_v, "before_rules"),
                "caught_with_rules": _caught(clean_v, ev_v, "final"),
                "query_rows": clean["retrieval"]["per_query"]["query_rows"],
                "clean_ndcg": clean["retrieval"]["per_query"]["exact"]["ndcg_at_k"],
                "faulted_ndcg": ev["retrieval"]["per_query"]["exact"]["ndcg_at_k"],
            }
            path.write_text(json.dumps(rec), encoding="utf-8")
            out.append(rec)
        print(f"{name:<22} {model:<38} faults done", flush=True)
    return out


def model_changes(cache: Path, results: dict, names: list[str]) -> list[dict]:
    """P18: replace the baseline model by each other model that is not the reference, on the
    same corpus and queries, and diff the two audits as a team would before a migration."""
    work = cache / "sixth_changes"
    work.mkdir(exist_ok=True)
    out = []
    for name in names:
        if (name, BASELINE) not in results or (name, REFERENCE) not in results:
            continue
        cands = [
            m.short
            for m in MODELS
            if m.short not in (BASELINE, REFERENCE) and (name, m.short) in results
        ]
        files = {m: work / f"{name}__{m}.json" for m in cands}
        if all(p.exists() for p in files.values()):
            out += [json.loads(p.read_text(encoding="utf-8")) for p in files.values()]
            continue
        d = load(DATASETS[name], cache)
        qrels = qrels_rows(d[0], d[2], d[4])
        r = np.load(cache / "emb" / f"{name}__{REFERENCE}.npz")
        kw = {
            "k": 10, "qrels": qrels, "result_sample": RESULT_SAMPLE,
            "reference_embeddings": r["docs"], "reference_queries": r["queries"],
        }  # fmt: skip
        z = np.load(cache / "emb" / f"{name}__{BASELINE}.npz")
        base = audit(z["docs"], queries=z["queries"], **kw).to_dict()
        for model, path in files.items():
            if path.exists():
                out.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            c = np.load(cache / "emb" / f"{name}__{model}.npz")
            ev = audit(c["docs"], queries=c["queries"], **kw).to_dict()
            dd = diff_evidence(
                ev,
                base,
                retrieval_tolerances={"ndcg_at_k": TOLERANCE},
                retrieval_confidence=CONFIDENCE,
                reference_tolerance=REFERENCE_TOLERANCE,
            )
            ag = dd["result_agreement"]
            rec = {
                "dataset": name,
                "model": model,
                "languages": next(m.languages for m in MODELS if m.short == model),
                "n_docs": results[(name, model)]["dataset"]["n_docs"],
                "baseline_ndcg": base["retrieval"]["exact"]["ndcg_at_k"],
                "candidate_ndcg": ev["retrieval"]["exact"]["ndcg_at_k"],
                "ndcg_drop": base["retrieval"]["exact"]["ndcg_at_k"]
                - ev["retrieval"]["exact"]["ndcg_at_k"],
                "changed_share": {s: v["changed_share"] for s, v in ag.items()},
                "top_hit": {s: v.get("reference_top_hit") for s, v in ag.items()},
                "reference_regressions": dd["reference_regressions"],
                "reference_review": dd["reference_review"],
                "query_rows": base["retrieval"]["per_query"]["query_rows"],
                "clean_ndcg": base["retrieval"]["per_query"]["exact"]["ndcg_at_k"],
                "faulted_ndcg": ev["retrieval"]["per_query"]["exact"]["ndcg_at_k"],
            }
            path.write_text(json.dumps(rec), encoding="utf-8")
            out.append(rec)
        print(f"{name:<22} model changes done", flush=True)
    return out


def _fault_criteria(cases: list[dict]) -> dict:
    """The criteria of P11 and P12 on a group of faults (duplicates left out)."""
    nodup = [c for c in cases if c["fault"] not in ROW_FAULTS]

    def hit(c):
        return (c["top_hit"].get("queries") or {}).get("z")

    harm = [c for c in nodup if c["ndcg_drop"] > TOLERANCE and hit(c) is not None]
    harmless = [c for c in nodup if c["ndcg_drop"] <= TOLERANCE and hit(c) is not None]
    big = [c for c in nodup if c["ndcg_drop"] > 0.02 and hit(c) is not None]
    missed = [c for c in big if not c["reference_regressions"]]
    quiet = [c for c in harmless if not c["reference_regressions"]]
    out = {
        "faults": len(cases),
        "without_duplicates": len(nodup),
        "harmful": len(harm),
        "harmless": len(harmless),
        "drop_over_002": len(big),
        "auc_top_hit_z": _auc([hit(c) for c in harm], [hit(c) for c in harmless]),
        "flagged_of_drop_over_002": sum(bool(c["reference_regressions"]) for c in big),
        "flagged_of_harmless": sum(bool(c["reference_regressions"]) for c in harmless),
        "drop_over_002_not_flagged": len(missed),
        "reviewed_of_those": sum(bool(c["reference_review"]) for c in missed),
        "harmless_reviewed": sum(bool(c["reference_review"]) for c in quiet),
        "by_fault": {
            f: {
                "of": sum(c["fault"] == f for c in cases),
                "harmful": sum(c["fault"] == f and c["ndcg_drop"] > TOLERANCE for c in cases),
                "flagged": sum(
                    c["fault"] == f
                    and c["ndcg_drop"] > TOLERANCE
                    and bool(c["reference_regressions"])
                    for c in cases
                ),
                "harmless_flagged": sum(
                    c["fault"] == f
                    and c["ndcg_drop"] <= TOLERANCE
                    and bool(c["reference_regressions"])
                    for c in cases
                ),
            }
            for f in FAULTS
        },
        "intervals": fault_intervals(cases, TOLERANCE),
    }
    out["testable_a"] = out["harmless"] >= 10 and out["harmful"] >= 10
    out["testable_b"] = out["harmless"] >= 10 and out["drop_over_002"] >= 10
    out["testable_review_a"] = len(missed) >= 10
    out["testable_review_b"] = out["harmless"] >= 10
    out["pass_a"] = (
        out["auc_top_hit_z"] is not None and out["auc_top_hit_z"] >= 0.85
        if out["testable_a"]
        else None
    )
    out["pass_b"] = (
        out["flagged_of_drop_over_002"] >= 0.65 * len(big)
        and out["flagged_of_harmless"] <= 0.15 * len(harmless)
        if out["testable_b"]
        else None
    )
    out["pass_review_a"] = (
        out["reviewed_of_those"] >= 0.5 * len(missed) if out["testable_review_a"] else None
    )
    out["pass_review_b"] = (
        out["harmless_reviewed"] <= 0.25 * len(harmless) if out["testable_review_b"] else None
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", help="run.py cache directory")
    p.add_argument("--role", default=ROLE, help="dataset role (another role only for a dry run)")
    p.add_argument("--summary", default=str(HERE / "sixth_summary.json"))
    args = p.parse_args(argv)
    cache = out_dir(args.out)
    results = {}
    for f in (cache / "results").glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        results[(r["dataset"]["name"], r["model"]["short"])] = r
    recs = {k: r for k, r in results.items() if role(k[0]) == args.role}
    best: dict[str, float] = defaultdict(float)
    for (name, _), r in recs.items():
        best[name] = max(best[name], r["ndcg_at_10"])
    work = cache / "sixth"
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

    # P13 (a, b): the two lowering rules on good and bad pipelines at fail with the reference
    gf = [x for x in good if _worst(x["findings"], "before_rules") == "fail"]
    bf = [x for x in bad if _worst(x["findings"], "before_rules") == "fail"]
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
    p9["lowered"] = {
        rule: {
            lab: sum(any(v["lowered_by"] == rule for v in x["findings"].values()) for x in group)
            for lab, group in (("good", good), ("bad", bad))
        }
        for rule in ("agreement", "one_step")
    }
    p9["larger_corpora"] = {
        name: {
            "n_docs": results[(name, REFERENCE)]["dataset"]["n_docs"],
            **{
                f"{lab}_{k}": sum(
                    _worst(x["findings"], k) == "fail"
                    for x in rows
                    if x["dataset"] == name and x["label"] == lab
                )
                for lab in ("good", "bad")
                for k in ("without", "before_rules", "final")
            },
        }
        for name in sorted({x["dataset"] for x in rows})
        if results[(name, REFERENCE)]["dataset"]["n_docs"] > SMALL
    }
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
        for k in ("without", "before_rules", "final")
    }

    # P15 and P16: the agreed top hits and the review tier on the faults, by corpus size
    good_pairs = sorted((x["dataset"], x["model"]) for x in good)
    cases = fault_cases(cache, results, good_pairs)
    small = [c for c in cases if c["n_docs"] <= SMALL]
    large = [c for c in cases if c["n_docs"] > SMALL]
    summary["P15_small_corpora"] = _fault_criteria(small)
    summary["P16_large_corpora"] = _fault_criteria(large)
    summary["P16_large_corpora"]["datasets"] = sorted({c["dataset"] for c in large})

    # P17 (c): the faults the findings catch stay caught with the two lowering rules
    harmful_all = [c for c in cases if c["ndcg_drop"] > TOLERANCE]
    before = [c for c in harmful_all if c["caught_before_rules"]]
    p9["faults_harmful"] = len(harmful_all)
    p9["faults_caught_before_rules"] = len(before)
    p9["faults_still_caught"] = sum(c["caught_with_rules"] for c in before)
    p9["faults_caught_large"] = sum(c["n_docs"] > SMALL for c in before)
    p9["faults_still_caught_large"] = sum(
        c["caught_with_rules"] for c in before if c["n_docs"] > SMALL
    )
    p9["testable_c"] = len(before) >= 10
    p9["pass_c"] = p9["faults_still_caught"] >= 0.95 * len(before) if p9["testable_c"] else None
    p9["intervals"] = {
        "good_cleared": count_interval(p9["good_cleared"], len(gf)),
        "bad_kept": count_interval(p9["bad_kept"], len(bf)),
        "faults_still_caught": count_interval(p9["faults_still_caught"], len(before)),
    }
    summary["P17_lowering_rules"] = p9
    summary["labeled_confidence_faults"] = label_gate(cases)

    # P18: a model change judged by the agreed top hits and by 100 labeled queries
    changes = model_changes(cache, results, sorted({x["dataset"] for x in rows}))
    big = [c for c in changes if c["ndcg_drop"] > 0.02]
    harm = [c for c in changes if c["ndcg_drop"] > TOLERANCE]
    harmless = [c for c in changes if c["ndcg_drop"] <= TOLERANCE]
    gate = label_gate(changes)
    sh = gate["share"]
    p18 = {
        "changes": len(changes),
        "baseline": BASELINE,
        "drop_over_002": len(big),
        "harmful": len(harm),
        "harmless": len(harmless),
        "flagged_of_drop_over_002": sum(bool(c["reference_regressions"]) for c in big),
        "flagged_of_harmless": sum(bool(c["reference_regressions"]) for c in harmless),
        "reviewed_of_harmless": sum(
            bool(c["reference_review"]) and not c["reference_regressions"] for c in harmless
        ),
        "flagged_or_reviewed_of_drop_over_002": sum(
            bool(c["reference_regressions"] or c["reference_review"]) for c in big
        ),
        "labeled_gate": gate,
        "by_languages": {
            lang: {
                "changes": sum(c["languages"] == lang for c in changes),
                "drop_over_002": sum(c["languages"] == lang for c in big),
                "flagged_of_drop_over_002": sum(
                    c["languages"] == lang and bool(c["reference_regressions"]) for c in big
                ),
                "harmless": sum(c["languages"] == lang for c in harmless),
                "flagged_of_harmless": sum(
                    c["languages"] == lang and bool(c["reference_regressions"]) for c in harmless
                ),
            }
            for lang in ("en", "multi")
        },
        "rows": [
            {
                "dataset": c["dataset"],
                "model": c["model"],
                "baseline_ndcg": round(c["baseline_ndcg"], 4),
                "candidate_ndcg": round(c["candidate_ndcg"], 4),
                "flagged": bool(c["reference_regressions"]),
                "reviewed": bool(c["reference_review"]),
                "lost_share": ((c["top_hit"].get("queries") or {}).get("lost_share")),
            }
            for c in sorted(changes, key=lambda c: (c["dataset"], c["model"]))
        ],
        "intervals": {
            "flagged_of_drop_over_002": count_interval(
                sum(bool(c["reference_regressions"]) for c in big), len(big)
            ),
            "flagged_of_harmless": count_interval(
                sum(bool(c["reference_regressions"]) for c in harmless), len(harmless)
            ),
        },
    }
    p18["testable_a"] = len(big) >= 10 and len(harmless) >= 10
    p18["pass_a"] = (
        p18["flagged_of_drop_over_002"] >= 0.65 * len(big)
        and p18["flagged_of_harmless"] <= 0.15 * len(harmless)
        if p18["testable_a"]
        else None
    )
    mbig = [c for c in big if c["languages"] == "multi"]
    mquiet = [c for c in harmless if c["languages"] == "multi"]
    p18["testable_multi"] = len(mbig) >= 10 and len(mquiet) >= 10
    p18["pass_multi"] = (
        sum(bool(c["reference_regressions"]) for c in mbig) >= 0.65 * len(mbig)
        and sum(bool(c["reference_regressions"]) for c in mquiet) <= 0.15 * len(mquiet)
        if p18["testable_multi"]
        else None
    )
    n = gate["cases"]
    p18["testable_b"] = n["harmful"] >= 10 and n["harmless"] >= 10
    if p18["testable_b"]:
        conf_pass = sh["confidence"]["harmful"]["pass"]
        mean_pass = sh["mean"]["harmful"]["pass"]
        p18["pass_b"] = conf_pass <= 0.03 and conf_pass <= 0.5 * mean_pass + 1e-12
        p18["pass_c"] = (
            sh["confidence"]["harmless"]["fail"] <= 0.75 * sh["mean"]["harmless"]["fail"] + 1e-12
        )
    else:
        p18["pass_b"] = p18["pass_c"] = None
    summary["P18_model_change"] = p18

    # P19: the same audit twice gives the same JSON
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
    summary["P19_determinism"] = {"reruns": reruns, "pass": all(r["identical"] for r in reruns)}

    Path(args.summary).write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "P18_model_change"}, indent=1)[:6000]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
