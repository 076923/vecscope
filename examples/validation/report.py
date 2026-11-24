"""Turn the held-out validation results (run.py) into tables of false positives and misses.

A pipeline is a (dataset, model) pair. Its quality is read from labels, not from vecscope:

- good: exact-search nDCG@10 at least ``GOOD`` (0.9) of the best model on that dataset
- bad: nDCG@10 at most ``BAD`` (0.5) of the best
- the rest are in between and appear only in the correlations

A false positive is a finding that is not ``ok`` on a good pipeline; a miss is a bad pipeline
on which no finding is ``ok``-violating. Both are broken down by language, document length,
corpus size and domain, the axes on which the defaults are suspected not to generalize.
``collapse`` on a corpus whose texts are themselves duplicated (share at least its warn_at)
describes the data correctly and is not counted as a false positive. The gate view counts
pipelines whose verdict is ``fail``, what the default ``--fail-on fail`` blocks.

If faults.py has run, the page also has its table: realistic faults injected into the good
pipelines, and whether the absolute verdict, the findings diff against the clean run, or the
retrieval tolerance catches each.

Run:  python examples/validation/report.py [--results DIR] [--out docs/validation.md]
Also writes the Korean page (docs/ko/validation.md) and examples/validation/summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import out_dir

from vecscope.evidence import THRESHOLDS

GOOD, BAD = 0.9, 0.5
MIN_MODELS = 5  # fewer models on a dataset and "the best" may itself be a poor pipeline
# ood_queries is deprecated and kept in the output for comparison only
SKIP = {"ood_queries"}
CORR = ["query_hubness", "hubness", "antihubs", "query_gap", "effective_rank", "anisotropy"]
DUP_WARN = THRESHOLDS["collapse.near_duplicate_ratio"][0]
QH_WARN = THRESHOLDS["queries.query_hubness_skewness"][0]
# an injected fault is harmful when exact nDCG@10 drops by more than this, the retrieval
# tolerance faults.py checks; the retrieval column then catches every harmful fault by
# construction, and the question is what the findings catch without labels
HARMFUL = 0.01


def text_duplicate_share(cache: Path, dataset: str) -> float | None:
    """Share of documents whose text (case-folded, trimmed) repeats another's."""
    f = cache / "data" / f"{dataset}.json"
    if not f.exists():
        return None
    texts = [t.strip().lower() for t in json.loads(f.read_text(encoding="utf-8"))["doc_text"]]
    return 1.0 - len(set(texts)) / len(texts) if texts else 0.0


def length_bucket(chars: int) -> str:
    return "short (< 200 chars)" if chars < 200 else "medium" if chars < 2000 else "long (>= 2,000)"


def size_bucket(n: int) -> str:
    return "< 5k docs" if n < 5_000 else "5k to 50k" if n < 50_000 else ">= 50k"


def lang_group(lang: str) -> str:
    return "code" if lang == "code" else "English" if lang == "en" else "other languages"


AXES = {
    "all": lambda d: "all",
    "language": lambda d: lang_group(d["language"]),
    "document length": lambda d: length_bucket(d["median_doc_chars"]),
    "corpus size": lambda d: size_bucket(d["n_docs"]),
    "domain": lambda d: d["domain"],
}


def load(results: Path, cache: Path) -> list[dict]:
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(results.glob("*.json"))]
    # the second held-out set (role "confirm") is analysed by confirm.py
    recs = [r for r in recs if r["dataset"].get("role", "heldout") == "heldout"]
    dup = {n: text_duplicate_share(cache, n) for n in {r["dataset"]["name"] for r in recs}}
    best: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for r in recs:
        best[r["dataset"]["name"]] = max(best[r["dataset"]["name"]], r["ndcg_at_10"])
        count[r["dataset"]["name"]] += 1
    recs = [r for r in recs if count[r["dataset"]["name"]] >= MIN_MODELS]
    for r in recs:
        b = best[r["dataset"]["name"]]
        share = r["ndcg_at_10"] / b if b else 0.0
        r["share_of_best"] = share
        r["label"] = "good" if share >= GOOD else "bad" if share <= BAD else "middle"
        r["text_duplicates"] = dup[r["dataset"]["name"]]
        explained = (
            {"collapse"}
            if r["text_duplicates"] is not None and r["text_duplicates"] >= DUP_WARN
            else set()
        )
        r["explained"] = sorted(
            n for n in explained if r["findings"].get(n, {}).get("verdict", "ok") != "ok"
        )
        r["flagged"] = sorted(
            n
            for n, f in r["findings"].items()
            if n not in SKIP and n not in explained and f["verdict"] != "ok"
        )
        r["gate_fail"] = any(
            f["verdict"] == "fail" for n, f in r["findings"].items() if n not in explained
        )
        qh = r["findings"].get("query_hubness")
        r["qh_not_judged"] = bool(qh and qh["verdict"] == "ok" and qh["value"] >= QH_WARN)
    return recs


def spearman(a, b) -> float:
    if len(a) < 4:
        return float("nan")
    from scipy.stats import spearmanr

    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else float("nan")


def summarize(recs: list[dict]) -> dict:
    findings = sorted({n for r in recs for n in r["findings"] if n not in SKIP})
    datasets = sorted({r["dataset"]["name"] for r in recs})
    out: dict = {
        "pipelines": len(recs),
        "datasets": len(datasets),
        "models": len({r["model"]["id"] for r in recs}),
        "good": sum(r["label"] == "good" for r in recs),
        "bad": sum(r["label"] == "bad" for r in recs),
        "good_gate_fail": sum(r["label"] == "good" and r["gate_fail"] for r in recs),
        "good_ok": sum(r["label"] == "good" and not r["flagged"] for r in recs),
        "bad_gate_fail": sum(r["label"] == "bad" and r["gate_fail"] for r in recs),
        "bad_model_language": sum(
            r["label"] == "bad"
            and r["model"]["languages"] == "en"
            and r["dataset"]["language"] not in ("en", "code")
            for r in recs
        ),
        "qh_not_judged": sum(r["qh_not_judged"] for r in recs),
        "explained": sum(bool(r["explained"]) for r in recs),
        "per_dataset": [],
        "false_positives": {},
        "misses": {},
        "spearman": {},
    }
    for name in datasets:
        rs = [r for r in recs if r["dataset"]["name"] == name]
        d = rs[0]["dataset"]
        top = max(rs, key=lambda r: r["ndcg_at_10"])
        out["per_dataset"].append(
            {
                **{k: d[k] for k in ("name", "language", "domain", "n_docs", "n_queries")},
                "median_doc_chars": d["median_doc_chars"],
                "text_duplicates": rs[0]["text_duplicates"],
                "models": len(rs),
                "best_ndcg": round(top["ndcg_at_10"], 3),
                "best_model": top["model"]["short"],
                "good": sum(r["label"] == "good" for r in rs),
                "bad": sum(r["label"] == "bad" for r in rs),
                "good_flagged": sum(r["label"] == "good" and bool(r["flagged"]) for r in rs),
                "bad_unflagged": sum(r["label"] == "bad" and not r["flagged"] for r in rs),
                "good_gate_fail": sum(r["label"] == "good" and r["gate_fail"] for r in rs),
            }
        )
        nd = [r["ndcg_at_10"] for r in rs]
        out["spearman"][name] = {
            f: spearman([r["findings"][f]["value"] for r in rs if f in r["findings"]], nd)
            if all(f in r["findings"] for r in rs)
            else float("nan")
            for f in CORR
        }
    for axis, key in AXES.items():
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            groups[key(r["dataset"])].append(r)
        fp, miss = {}, {}
        for g, rs in sorted(groups.items()):
            good = [r for r in rs if r["label"] == "good"]
            bad = [r for r in rs if r["label"] == "bad"]
            fp[g] = {
                "good": len(good),
                "any": sum(bool(r["flagged"]) for r in good),
                "gate_fail": sum(r["gate_fail"] for r in good),
                "per_finding": {
                    f: sum(r["findings"].get(f, {}).get("verdict", "ok") != "ok" for r in good)
                    for f in findings
                },
            }
            miss[g] = {
                "bad": len(bad),
                "unflagged": sum(not r["flagged"] for r in bad),
                "hub_flagged": sum(
                    any(f in r["flagged"] for f in ("hubness", "query_hubness")) for r in bad
                ),
                "median_flags": float(np.median([len(r["flagged"]) for r in bad])) if bad else 0,
            }
        out["false_positives"][axis] = fp
        out["misses"][axis] = miss
    return out


def summarize_faults(folder: Path) -> dict:
    """Per fault: how often it hurt retrieval and what caught it (faults.py records)."""
    from faults import FAULTS
    from registry import role

    recs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(folder.glob("*.json"))]
    recs = [r for r in recs if role(r["dataset"]) == "heldout"]
    out: dict = {}

    for fault in [f for f in FAULTS if any(r["fault"] == f for r in recs)]:
        rs = [r for r in recs if r["fault"] == fault]
        harmful = [r for r in rs if r["ndcg_drop"] > HARMFUL]
        harmless = [r for r in rs if r["ndcg_drop"] <= HARMFUL]

        def caught(r: dict) -> dict:
            return {
                "absolute": r["absolute_fail"] and not r["absolute_fail_clean"],
                "diff": bool(r["verdict_regressions"]),
                "retrieval": bool(r["retrieval_regressions"]),
            }

        def count(group: list[dict], how: str) -> int:
            if how == "any":
                return sum(any(caught(r).values()) for r in group)
            if how == "findings":
                return sum(caught(r)["absolute"] or caught(r)["diff"] for r in group)
            return sum(caught(r)[how] for r in group)

        hows = ("absolute", "diff", "findings", "retrieval", "any")
        out[fault] = {
            "runs": len(rs),
            "median_drop": float(np.median([r["ndcg_drop"] for r in rs])),
            "harmful": len(harmful),
            "caught": {h: count(harmful, h) for h in hows},
            "harmless": len(harmless),
            "alarms": {h: count(harmless, h) for h in hows},
        }
    return out


def _pct(a: int, b: int) -> str:
    return "n/a" if not b else f"{a}/{b} ({a / b:.0%})"


def _rho(v: float) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:+.2f}"


TEXT = {
    "en": {
        "lang_line": "**English** · [한국어](ko/validation.md)",
        "title": "# Held-out validation on public data",
        "intro": (
            "Every dataset and model on this page is public, and none of the datasets was used to "
            "set a threshold: the defaults were fitted on SciFact, NFCorpus and ko-strategyqa "
            "([verification](verification.md)). Generated by `examples/validation/report.py` from "
            "the results of `examples/validation/run.py`; the thresholds were not changed to fit "
            "these numbers."
        ),
        "labels": (
            f"A pipeline is a (dataset, model) pair. It is **good** when its exact-search nDCG@10 "
            f"is at least {GOOD:.0%} of the best model on that dataset and **bad** at "
            f"{BAD:.0%} or less; the rest are left out of the counts. Datasets with fewer than "
            f"{MIN_MODELS} models are left out. A **false positive** is a "
            "finding that is not `ok` on a good pipeline (the deprecated `ood_queries` is "
            "ignored). A **miss** is a bad pipeline with every finding `ok`."
        ),
        "count": "{pipelines} pipelines, {datasets} datasets, {models} models; {good} good, {bad} bad.",
        "reading": (
            "What these numbers mean for the defaults, and what to do about the false positives, "
            "is in [verification](verification.md#held-out-public-data)."
        ),
        "datasets": "## Datasets",
        "ds_head": (
            "| dataset | language | domain | documents | queries | median length | text "
            "duplicates | models | best nDCG@10 (model) | good | bad | good with a flag | good at "
            "fail | bad without a flag |"
        ),
        "fp": "## False positives on good pipelines",
        "fp_note": (
            "Share of good pipelines on which a finding is `warn` or `fail`, by the axes on which "
            "the defaults were expected not to generalize."
        ),
        "miss": "## Misses on bad pipelines",
        "miss_head": "| group | bad pipelines | no finding flagged | flagged by hubness or query_hubness | median findings flagged |",
        "corr": "## Spearman correlation with nDCG@10 per dataset",
        "corr_note": "Negative means the finding rises as retrieval gets worse.",
        "gate": (
            "At the default gate (`--fail-on fail`) {good_gate_fail} of the {good} good pipelines "
            "fail and {good_ok} have every finding `ok`; {bad_gate_fail} of the {bad} bad "
            "pipelines fail. {bad_model_language} of the bad pipelines are English-only models on "
            "other languages. `collapse` on a corpus whose texts are themselves duplicated "
            "({explained} pipelines) is counted as correct, not as a false positive. "
            "`query_hubness` was not judged at warn on {qh_not_judged} pipelines: chance alone "
            "reaches warn_at with that few queries over that many documents."
        ),
        "fail_col": "verdict fail",
        "faults": "## Injected faults",
        "faults_note": (
            "`examples/validation/faults.py` injects one fault at a time into every good "
            "pipeline on a corpus of at most 25,000 documents and audits it with its queries and "
            "qrels. A fault is **harmful** when exact nDCG@10 drops by more than {harmful}, the "
            "retrieval tolerance below, so the retrieval column catches every harmful fault by "
            "construction; it needs labeled queries, and the other columns show what the "
            "findings catch without them. "
            "Caught by: **absolute**, the verdict becomes `fail` (it was not on the clean run); "
            "**diff**, a verdict is worse than on the clean run as baseline (`--fail-on none "
            "--baseline`); **findings**, either of the two; **retrieval**, nDCG@10 fell by more "
            "than 0.01 against the baseline (`--retrieval-tolerance ndcg_at_k=0.01`). On harmless "
            "faults the same columns count alarms."
        ),
        "faults_head": (
            "| fault | runs | median nDCG drop | harmful | caught: absolute | diff | findings | "
            "retrieval | any | harmless | alarms: findings | retrieval |"
        ),
        "fault_names": {
            "query_model": "queries from another model of the same dimension",
            "truncate": "first half of the dimensions only",
            "zero_rows": "5% of documents are zero vectors",
            "duplicates": "20% of documents stored twice",
            "id_shift": "10% of rows hold another document's vector",
            "quantize": "4-bit rounding without per-vector scale",
        },
        "dup_col": "text duplicates",
        "axis": "By",
        "group": "group",
        "dataset": "dataset",
        "any": "any finding",
    },
    "ko": {
        "lang_line": "[English](../validation.md) · **한국어**",
        "title": "# 공개 데이터 held-out 검증",
        "intro": (
            "이 페이지의 데이터세트와 모델은 모두 공개된 것이고, 어느 데이터세트도 임곗값을 정하는 데 "
            "쓰지 않았습니다. 기본 임곗값은 SciFact, NFCorpus, ko-strategyqa로 맞췄습니다"
            "([검증 결과](verification.md)). `examples/validation/run.py`의 결과로 "
            "`examples/validation/report.py`가 만든 페이지이며, 이 숫자에 맞춰 임곗값을 바꾸지 "
            "않았습니다."
        ),
        "labels": (
            f"파이프라인은 (데이터세트, 모델) 쌍입니다. 정확 검색 nDCG@10이 그 데이터세트 최고 모델의 "
            f"{GOOD:.0%} 이상이면 **좋음**, {BAD:.0%} 이하이면 **나쁨**이고, 그 사이는 집계에서 "
            f"뺍니다. 모델이 {MIN_MODELS}개 미만인 데이터세트도 뺍니다. **오탐**은 좋은 파이프라인에서 `ok`가 아닌 항목이고(사용 중단된 `ood_queries`는 "
            "제외), **놓침**은 모든 항목이 `ok`인 나쁜 파이프라인입니다."
        ),
        "count": "파이프라인 {pipelines}개, 데이터세트 {datasets}개, 모델 {models}개; 좋음 {good}개, 나쁨 {bad}개.",
        "reading": (
            "이 숫자가 기본 임곗값에 대해 말하는 것과 오탐에 대한 대처는 "
            "[검증 결과](verification.md#held-out-공개-데이터)에 있습니다."
        ),
        "datasets": "## 데이터세트",
        "ds_head": (
            "| 데이터세트 | 언어 | 도메인 | 문서 | 질의 | 길이 중앙값 | 텍스트 중복 | 모델 | "
            "최고 nDCG@10 (모델) | 좋음 | 나쁨 | 플래그 있는 좋음 | fail인 좋음 | 플래그 없는 나쁨 |"
        ),
        "fp": "## 좋은 파이프라인의 오탐",
        "fp_note": (
            "좋은 파이프라인 중 해당 항목이 `warn`이나 `fail`인 비율을, 기본값이 일반화되지 않을 "
            "것으로 의심한 축별로 나눈 것입니다."
        ),
        "miss": "## 나쁜 파이프라인의 놓침",
        "miss_head": "| 그룹 | 나쁜 파이프라인 | 플래그 없음 | hubness나 query_hubness가 플래그 | 플래그된 항목 수 중앙값 |",
        "corr": "## 데이터세트별 nDCG@10과의 스피어만 상관",
        "corr_note": "음수는 검색 품질이 나빠질수록 값이 커진다는 뜻입니다.",
        "gate": (
            "기본 게이트(`--fail-on fail`)에서 좋은 파이프라인 {good}개 중 {good_gate_fail}개가 "
            "실패하고, 모든 항목이 `ok`인 것은 {good_ok}개입니다. 나쁜 파이프라인은 {bad}개 중 "
            "{bad_gate_fail}개가 실패합니다. 나쁜 파이프라인 중 {bad_model_language}개는 영어 전용 "
            "모델을 다른 언어에 쓴 경우입니다. 텍스트 자체가 중복된 코퍼스의 `collapse`"
            "({explained}개 파이프라인)는 오탐이 아니라 맞는 판정으로 셉니다. `query_hubness`는 "
            "{qh_not_judged}개 파이프라인에서 warn 단계를 판정하지 않았습니다. 그 질의 수와 문서 "
            "수에서는 우연만으로도 warn_at에 닿기 때문입니다."
        ),
        "fail_col": "판정 fail",
        "faults": "## 결함 주입",
        "faults_note": (
            "`examples/validation/faults.py`는 문서 2만 5천 개 이하 코퍼스의 좋은 파이프라인마다 "
            "결함을 하나씩 넣고 질의·qrels와 함께 감사합니다. 정확 검색 nDCG@10이 {harmful}보다 "
            "더 떨어지면 **해로운** 결함입니다. 아래 검색 허용량과 같은 기준이라 retrieval 열은 "
            "해로운 결함을 정의상 모두 잡습니다. 그 열에는 라벨 질의가 필요하고, 나머지 열은 라벨 "
            "없이 진단 항목이 잡는 것을 보여 줍니다. 잡은 쪽: **absolute**는 판정이 `fail`이 된 "
            "경우(깨끗한 "
            "실행은 아니었음), **diff**는 깨끗한 실행을 기준선으로 어떤 판정이 나빠진 경우"
            "(`--fail-on none --baseline`), **findings**는 둘 중 하나, **retrieval**은 기준선 대비 "
            "nDCG@10이 0.01보다 더 떨어진 경우(`--retrieval-tolerance ndcg_at_k=0.01`)입니다. 해롭지 "
            "않은 결함에서는 같은 열이 경보 수입니다."
        ),
        "faults_head": (
            "| 결함 | 실행 | nDCG 하락 중앙값 | 해로움 | 잡음: absolute | diff | findings | "
            "retrieval | 아무거나 | 해롭지 않음 | 경보: findings | retrieval |"
        ),
        "fault_names": {
            "query_model": "같은 차원의 다른 모델로 만든 질의",
            "truncate": "앞쪽 절반 차원만 사용",
            "zero_rows": "문서 5%가 영벡터",
            "duplicates": "문서 20%가 두 번 저장됨",
            "id_shift": "행 10%에 다른 문서의 벡터",
            "quantize": "벡터별 스케일 없는 4비트 반올림",
        },
        "dup_col": "텍스트 중복",
        "axis": "축:",
        "group": "그룹",
        "dataset": "데이터세트",
        "any": "아무 항목",
    },
}


def render(s: dict, lang: str) -> str:
    t = TEXT[lang]
    lines = [t["lang_line"], "", t["title"], "", t["intro"], "", t["labels"], ""]
    lines += [
        t["count"].format(**s),
        "",
        t["gate"].format(**s),
        "",
        t["reading"],
        "",
        t["datasets"],
        "",
        t["ds_head"],
        "|" + "---|" * 14,
    ]
    for d in s["per_dataset"]:
        dup = "n/a" if d["text_duplicates"] is None else f"{d['text_duplicates']:.1%}"
        lines.append(
            f"| {d['name']} | {d['language']} | {d['domain']} | {d['n_docs']:,} | "
            f"{d['n_queries']:,} | {d['median_doc_chars']:,} | {dup} | {d['models']} | "
            f"{d['best_ndcg']:.3f} ({d['best_model']}) | {d['good']} | {d['bad']} | "
            f"{d['good_flagged']} | {d['good_gate_fail']} | {d['bad_unflagged']} |"
        )
    lines += ["", t["fp"], "", t["fp_note"], ""]
    for axis, groups in s["false_positives"].items():
        flagged = sorted(
            {f for g in groups.values() for f, n in g["per_finding"].items() if n},
            key=lambda f: -sum(g["per_finding"][f] for g in groups.values()),
        )
        if axis != "all":
            lines += [f"### {t['axis']} {axis}", ""]
        lines.append(
            f"| {t['group']} | {t['fail_col']} | {t['any']} | " + " | ".join(flagged) + " |"
        )
        lines.append("|" + "---|" * (3 + len(flagged)))
        for g, v in groups.items():
            lines.append(
                f"| {g} | {_pct(v['gate_fail'], v['good'])} | {_pct(v['any'], v['good'])} | "
                + " | ".join(_pct(v["per_finding"][f], v["good"]) for f in flagged)
                + " |"
            )
        lines.append("")
    lines += [t["miss"], "", t["miss_head"], "|---|---|---|---|---|"]
    for axis, groups in s["misses"].items():
        for g, v in groups.items():
            if not v["bad"]:
                continue
            label = g if axis == "all" else f"{axis}: {g}"
            lines.append(
                f"| {label} | {v['bad']} | {_pct(v['unflagged'], v['bad'])} | "
                f"{_pct(v['hub_flagged'], v['bad'])} | {v['median_flags']:g} |"
            )
    lines += ["", t["corr"], "", t["corr_note"], ""]
    lines.append(f"| {t['dataset']} | " + " | ".join(CORR) + " |")
    lines.append("|" + "---|" * (1 + len(CORR)))
    for name, row in s["spearman"].items():
        lines.append(f"| {name} | " + " | ".join(_rho(row[f]) for f in CORR) + " |")
    if s.get("faults"):
        lines += ["", t["faults"], "", t["faults_note"].format(harmful=HARMFUL), ""]
        lines += [t["faults_head"], "|" + "---|" * 12]
        for fault, v in s["faults"].items():
            c, a = v["caught"], v["alarms"]
            lines.append(
                f"| {fault}: {t['fault_names'].get(fault, '')} | {v['runs']} | "
                f"{v['median_drop']:+.3f} | {v['harmful']} | "
                + " | ".join(_pct(c[h], v["harmful"]) for h in ("absolute", "diff", "findings"))
                + f" | {_pct(c['retrieval'], v['harmful'])} | {_pct(c['any'], v['harmful'])} | "
                f"{v['harmless']} | {_pct(a['findings'], v['harmless'])} | "
                f"{_pct(a['retrieval'], v['harmless'])} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", help="directory of run.py results (default: run.py's cache)")
    p.add_argument("--out", default=str(HERE.parents[1] / "docs" / "validation.md"))
    args = p.parse_args(argv)
    cache = out_dir(None)
    results = Path(args.results) if args.results else cache / "results"
    recs = load(results, cache)
    if not recs:
        p.error(f"no results in {results}; run examples/validation/run.py first")
    s = summarize(recs)
    if (cache / "faults").is_dir():
        s["faults"] = summarize_faults(cache / "faults")
    out = Path(args.out)
    out.write_text(render(s, "en"), encoding="utf-8")
    (out.parent / "ko" / out.name).write_text(render(s, "ko"), encoding="utf-8")
    (HERE / "summary.json").write_text(
        json.dumps(s, indent=1, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {out}, {out.parent / 'ko' / out.name} and {HERE / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
