"""vecscope CLI: `vecscope audit embeddings.npy`, `vecscope sweep`, `vecscope compare`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vecscope import __version__
from vecscope.adapters import annotate, is_source, load_source
from vecscope.compare import compare, render_compare
from vecscope.evidence import WORSE, audit, diff_evidence
from vecscope.incremental import cached_knn_for_audit
from vecscope.io import load_neighbors, load_vectors
from vecscope.retrieval import load_qrels
from vecscope.sweep import render_sweep, sweep


def _render_text(ev) -> str:
    mark = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}
    head = f"vecscope {ev.version}  n={ev.n} dim={ev.dim} k={ev.k}  verdict={ev.verdict.upper()}"
    if ev.sampled:
        head += f"  (sampled from {ev.n_total} rows"
        head += f", {ev.sample_repeats} repeats)" if ev.sample_repeats > 1 else ")"
    if ev.n_queries:
        head += f"  queries={ev.n_queries}"
    lines = [head]
    for f in ev.findings:
        line = f"  [{mark[f.verdict]}] {f.name:<22} {f.metric}={f.value:.4f}  ({f.healthy_range})"
        if f.value_range is not None:
            line += f"  repeats {f.value_range[0]:.4f}~{f.value_range[1]:.4f}"
            if not f.verdict_stable:
                line += "  verdict UNSTABLE"
        if f.inconclusive:
            line += "  INCONCLUSIVE (sampled)"
        lines.append(line)
        lines.append(f"{'':31}{f.explanation}")
        if f.verdict != "ok":
            lines.append(f"{'':31}try: {f.suggested_action}")
    if ev.retrieval:
        r = ev.retrieval
        lines.append(f"  retrieval ({r['judged_queries']} judged queries, k={r['k']}):")
        for part in ("exact", "ranked"):
            if r.get(part):
                m = r[part]
                extra = (f"  overlap_with_exact={m['overlap_with_exact']:.4f}"
                         if "overlap_with_exact" in m else "")
                lines.append(f"    {part:<7} nDCG={m['ndcg_at_k']:.4f} recall={m['recall_at_k']:.4f}"
                             f" MRR={m['mrr_at_k']:.4f} dup_share={m['duplicate_result_share']:.4f}"
                             + extra)
    if ev.plan:
        lines.append("  plan (fix in this order; each step is a hypothesis to test):")
        for st in ev.plan:
            names = ", ".join(f"{e['name']}={e['value']:.3g} {e['verdict']}"
                              + (" (inconclusive)" if e["inconclusive"] else "")
                              for e in st["findings"])
            lines.append(f"    {st['order']}. {st['group']}: {names}")
            lines.append(f"{'':7}cause: {st['common_cause']}")
            lines.append(f"{'':7}first: {st['first_change']}")
            lines.append(f"{'':7}check: {st['re_measure']}")
    if ev.source:
        lines.append(f"  source: {ev.source['backend']} {ev.source['uri']} (rows in ascending id "
                     f"order, fingerprint {ev.source['fingerprint'][:12]})")
    if ev.knn_cache:
        c = ev.knn_cache
        changed = f", {c['changed_rows']} changed" if c.get("changed_rows") else ""
        why = f" ({c['reason']})" if c.get("reason") else ""
        lines.append(f"  knn cache {c['path']}: {c['status']}{changed}, "
                     f"{c.get('recomputed_rows', c['new_rows'])} rows computed{why}")
    if ev.clusters:
        lines.append(f"  clusters ({len(ev.clusters)}):")
        lines.append(f"{'':4}{'id':>3}{'size':>7}{'share':>7}{'cos2ctr':>9}{'neardup':>9}"
                     f"{'hubshare':>10}  sample_ids")
        for c in ev.clusters:
            lines.append(f"{'':4}{c['cluster']:>3}{c['size']:>7}{c['share']:>7.1%}"
                         f"{c['mean_cosine_to_center']:>9.3f}{c['near_duplicate_ratio']:>9.1%}"
                         f"{c['hub_share']:>10.1%}  {c['sample_ids']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vecscope", description="Embedding space evidence for agents.")
    p.add_argument("--version", action="version", version=f"vecscope {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="audit an embedding matrix")
    a.add_argument("path", help=".npy/.npz/.parquet/.jsonl/.csv with one vector per row, a faiss "
                                "index file, or a vector store URI (qdrant://, chroma://, "
                                "milvus://, weaviate://; see the README)")
    a.add_argument("-k", type=int, default=10, help="neighbors per item (default 10)")
    a.add_argument("--ann", help="optional (n x k) neighbor ids from an ANN index, one row per "
                                 "embedding row; every row must be audited (--sample 0 above "
                                 "the sample size); self ids are ignored")
    a.add_argument("--queries", help="optional query embeddings (m x dim) for query-side findings")
    a.add_argument("--qrels", help="relevance judgments for the queries (.json/.jsonl/.tsv: query "
                                   "row, corpus row, grade); adds nDCG/recall/MRR of exact search")
    a.add_argument("--ranked", help="(m x k) corpus row ids per query from the production index; "
                                    "scored with --qrels next to exact search")
    a.add_argument("--ann-from-source", action="store_true",
                   help="when the path is a faiss index or vector store, query its index for "
                        "every stored vector and audit that as --ann")
    a.add_argument("--save-ids", help="write the source's ids in row order to this JSON file")
    a.add_argument("--knn-cache", help="exact k-NN cache file (.npz): reused for the same rows, "
                                       "extended when rows were appended, created otherwise; "
                                       "needs every row audited")
    a.add_argument("--clusters", type=int, default=0,
                   help="attach a per-cluster report with this many clusters, at most one per "
                        "10 rows (0 = off)")
    a.add_argument("--thresholds", help='JSON file of overrides, e.g. {"hubness.skewness": [2.5, 5]}')
    a.add_argument("--near-dup", type=float, default=0.99, help="cosine threshold for near-dups")
    a.add_argument("--sample", type=int, default=100_000,
                   help="audit a seeded random subset when n exceeds this (0 = all rows)")
    a.add_argument("--sample-repeats", type=int, default=1,
                   help="when sampling applies, re-audit this many independent samples and "
                        "report each finding's value range and verdict stability (default 1)")
    a.add_argument("--json", action="store_true", help="emit JSON evidence instead of text")
    a.add_argument("--fail-on", choices=["warn", "fail", "none"], default="fail",
                   help="exit 1 when the verdict is at least this level against the absolute "
                        "thresholds (default fail); 'none' never fails on the thresholds alone, "
                        "so with --baseline only regressions against the baseline fail the run")
    a.add_argument("--baseline", help="earlier --json output to diff against; prints each "
                                      "finding's change and exits 1 when any verdict got worse. "
                                      "With --queries, a baseline audited with queries also "
                                      "adds query_shift against its query distances")
    a.add_argument("--tolerances", help='JSON file of allowed worsening per finding against the '
                                        'baseline, e.g. {"collapse": 0.01, "ann_recall": 0.02}; '
                                        'a finding that worsens by more exits 1 even when its '
                                        'verdict did not change (needs --baseline)')
    a.add_argument("--require-comparable", action="store_true",
                   help="with --baseline: exit 2 instead of printing a note when n, dim, k, "
                        "sampling, thresholds or the near-dup cutoff differ from the baseline")
    s = sub.add_parser("sweep", help="sweep HNSW/IVF parameters (needs faiss-cpu)")
    s.add_argument("path", help="embedding file, one vector per row")
    s.add_argument("-k", type=int, default=10, help="neighbors per query (default 10)")
    s.add_argument("--queries", help="optional query embeddings; default samples the corpus")
    s.add_argument("--n-queries", type=int, default=1000, help="queries to time (default 1000)")
    s.add_argument("--target-recall", type=float, default=0.95,
                   help="recall@k a configuration must reach to be recommended")
    s.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                   help="run IVF and the exact baseline on the GPU when faiss has GPU support "
                        "(default auto); HNSW is CPU-only in faiss")
    s.add_argument("--timing-repeats", type=int, default=3,
                   help="time each setting this many times and keep the median qps (default 3)")
    s.add_argument("--margin", type=float, default=0.02,
                   help="recall headroom over the target below which the recommendation is "
                        "marked tight and a safer setting is named (default 0.02)")
    s.add_argument("--latency-queries", type=int, default=200,
                   help="queries searched one at a time per setting for the latency "
                        "percentiles p50/p95/p99 (default 200, 0 = skip)")
    s.add_argument("--json", action="store_true", help="emit JSON instead of text")
    c = sub.add_parser("compare", help="compare two embeddings of the same rows (no verdicts)")
    c.add_argument("path_a", help="embedding file A, one vector per row")
    c.add_argument("path_b", help="embedding file B, same rows in the same order")
    c.add_argument("-k", type=int, default=10, help="neighbors per item (default 10)")
    c.add_argument("--near-dup", type=float, default=0.99, help="cosine threshold for near-dups")
    c.add_argument("--sample", type=int, default=100_000,
                   help="compare a seeded random subset when n exceeds this (0 = all rows); "
                        "without --neighbors-a/-b the neighbors are recomputed inside the "
                        "sample and the result says neighbor_scope=sample")
    c.add_argument("--neighbors-a", help="(n x >= k) neighbor ids per row computed on the full "
                                         "corpus in space A (exact or from the index); with "
                                         "--neighbors-b, a sampled compare keeps full-corpus "
                                         "neighbors")
    c.add_argument("--neighbors-b", help="same for space B; both or neither")
    c.add_argument("--json", action="store_true", help="emit JSON instead of text")
    b = sub.add_parser("calibrate", help="propose thresholds from audits of your own pipeline "
                                         "states with labeled queries")
    b.add_argument("states", nargs="+", help="--json audits of pipeline states, each run with the "
                                             "same --queries and --qrels (3 or more; 10 or more to "
                                             "support a threshold)")
    b.add_argument("--min-quality", type=float, required=True,
                   help="a state is acceptable when its retrieval metric reaches this value")
    b.add_argument("--metric", default="ndcg_at_k",
                   help="retrieval metric (ndcg_at_k, recall_at_k, mrr_at_k, hit_rate_at_k)")
    b.add_argument("--ranking", choices=["exact", "ranked"], default="exact",
                   help="score exact search or the --ranked results (default exact)")
    b.add_argument("--holdout", nargs="+", default=[],
                   help="--json audits not used for fitting (other states, or the same states "
                        "scored on held-out queries) to check the proposal on")
    b.add_argument("--out", help="write the gate thresholds here, for audit --thresholds")
    b.add_argument("--include-low-confidence", action="store_true",
                   help="also write exploratory pairs (tracking quality, but not significant "
                        "after the multiple-comparison correction) to --out")
    b.add_argument("--allow-mixed-qrels", action="store_true",
                   help="accept states scored on different qrels or k (their quality values "
                        "then do not compare)")
    b.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = p.parse_args(argv)
    try:
        return _run(args)
    except (ValueError, TypeError, FileNotFoundError, ImportError, OSError, RuntimeError,
            json.JSONDecodeError) as e:
        print(f"vecscope: error: {e}", file=sys.stderr)
        return 2


def _run(args) -> int:
    if args.cmd == "calibrate":
        from vecscope.calibrate import calibrate, render_calibration

        res = calibrate([_load_json(f, "states") for f in args.states], args.min_quality,
                        metric=args.metric, ranking=args.ranking,
                        holdout=[_load_json(f, "--holdout") for f in args.holdout] or None,
                        names=[Path(f).name for f in args.states],
                        include_low_confidence=args.include_low_confidence,
                        allow_mixed_qrels=args.allow_mixed_qrels)
        if args.out:
            Path(args.out).write_text(json.dumps(res["thresholds"], indent=2) + "\n",
                                      encoding="utf-8")
        print(json.dumps(res, indent=2) if args.json else render_calibration(res))
        return 0
    if args.cmd == "audit" and args.sample < 0:
        raise ValueError("--sample must be 0 (all rows) or a positive number")
    if args.cmd == "audit" and args.clusters < 0:
        raise ValueError("--clusters must be 0 or a positive number")
    if args.cmd == "audit" and args.sample_repeats < 1:
        raise ValueError("--sample-repeats must be a positive number")
    if args.cmd == "compare":
        if args.sample < 0:
            raise ValueError("--sample must be 0 (all rows) or a positive number")
        if (args.neighbors_a is None) != (args.neighbors_b is None):
            raise ValueError("--neighbors-a and --neighbors-b go together")
        na = load_neighbors(args.neighbors_a) if args.neighbors_a else None
        nb = load_neighbors(args.neighbors_b) if args.neighbors_b else None
        res = compare(load_vectors(args.path_a), load_vectors(args.path_b), k=args.k,
                      near_dup_cos=args.near_dup, sample=args.sample, neighbors_a=na,
                      neighbors_b=nb)
        print(json.dumps(res.to_dict(), indent=2) if args.json else render_compare(res))
        return 0
    src = None
    if args.cmd == "audit" and is_source(args.path):
        src = load_source(args.path)
        x = src.vectors
    else:
        x = load_vectors(args.path)
    if args.cmd == "sweep":
        q = load_vectors(args.queries) if args.queries else None
        res = sweep(x, k=args.k, queries=q, n_queries=args.n_queries,
                    target_recall=args.target_recall, device=args.device,
                    timing_repeats=args.timing_repeats, margin=args.margin,
                    latency_queries=args.latency_queries)
        print(json.dumps(res.to_dict(), indent=2) if args.json else render_sweep(res))
        return 0

    if (args.ann_from_source or args.save_ids) and src is None:
        raise ValueError("--ann-from-source and --save-ids need a faiss index or store URI as path")
    if args.ann_from_source and args.ann:
        raise ValueError("pass --ann or --ann-from-source, not both")
    ann = load_neighbors(args.ann) if args.ann else None
    if args.ann_from_source:
        ann = src.neighbors(args.k)
    if args.save_ids:
        with open(args.save_ids, "w", encoding="utf-8") as fh:
            json.dump(src.ids, fh)
    if src is not None:
        src.close()
    q = load_vectors(args.queries) if args.queries else None
    qrels = load_qrels(args.qrels) if args.qrels else None
    ranked = load_neighbors(args.ranked) if args.ranked else None
    knn, cache_info = None, None
    if args.knn_cache:
        if args.sample and x.shape[0] > args.sample:
            raise ValueError("--knn-cache needs every row audited; pass --sample 0")
        knn, cache_info = cached_knn_for_audit(x, args.k, args.knn_cache)
    th = _load_json(args.thresholds, "--thresholds") if args.thresholds else None
    if (args.tolerances or args.require_comparable) and not args.baseline:
        raise ValueError("--tolerances and --require-comparable need --baseline")
    base = _load_json(args.baseline, "--baseline") if args.baseline else None
    ev = audit(x, k=args.k, ann_neighbors=ann, near_dup_cos=args.near_dup,
               sample=args.sample, queries=q, clusters=args.clusters, thresholds=th,
               sample_repeats=args.sample_repeats,
               query_baseline=base if q is not None and _query_ref_ok(base, x) else None,
               qrels=qrels, ranked=ranked, knn=knn)
    if src is not None:
        annotate(ev, src)
    if cache_info is not None:
        ev.knn_cache = cache_info
    d = None
    if base is not None:
        tol = _load_json(args.tolerances, "--tolerances") if args.tolerances else None
        d = diff_evidence(ev.to_dict(), base, tolerances=tol)
        if args.require_comparable and not d["comparable"]:
            raise ValueError(
                f"--require-comparable: baseline {args.baseline} was made with different n, "
                "dim, k, sampling, thresholds or near-dup cutoff; re-create it with the same "
                "settings or drop the flag"
            )
    gate = gate_decision(ev.verdict, args.fail_on, d, ev.inconclusive_findings)
    note = _query_ref_note(base, x, q)
    if args.json:
        out = ev.to_dict()
        if d is not None:
            out["baseline_diff"] = d
        if note:
            out["query_reference_note"] = note
        out["gate"] = gate
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(_render_text(ev))
        if d is not None:
            print(_render_diff(d, args.baseline))
        if note:
            print(f"  note: {note}")
        print(_render_gate(gate))
    return gate["exit_code"]


def gate_decision(verdict: str, fail_on: str, diff: dict | None,
                  inconclusive: list[str] | None = None) -> dict:
    """Why the run passes or fails, with the absolute-threshold rule and the baseline rules
    kept apart: ``absolute`` is the verdict against the thresholds (``fail_on`` = 'none'
    disables it), ``verdict_regressions`` and ``value_regressions`` come from the baseline.
    A gate that blocks on the baseline alone (``--fail-on none --baseline``) judges the
    change, not whether the corpus meets the default thresholds. ``inconclusive`` names the
    sampled findings whose verdict cannot stand for the full corpus; they never fail the
    gate, but a pass is reported as not covering them."""
    absolute = fail_on != "none" and WORSE[verdict] >= WORSE[fail_on]
    verdict_reg = list(diff["regressions"]) if diff else []
    value_reg = list(diff["value_regressions"]) if diff else []
    reasons = []
    if absolute:
        reasons.append(f"verdict {verdict} reaches --fail-on {fail_on}")
    if verdict_reg:
        reasons.append(f"verdict worse than baseline: {', '.join(verdict_reg)}")
    if value_reg:
        reasons.append(f"value worse than baseline beyond tolerance: {', '.join(value_reg)}")
    return {"fail_on": fail_on, "absolute": absolute, "verdict_regressions": verdict_reg,
            "value_regressions": value_reg, "baseline": diff is not None,
            "inconclusive": list(inconclusive or []),
            "exit_code": 1 if reasons else 0, "reasons": reasons}


def _render_gate(gate: dict) -> str:
    tail = (f"; not covered, inconclusive on this sample: {', '.join(gate['inconclusive'])}"
            if gate.get("inconclusive") else "")
    if gate["exit_code"] == 0:
        rule = ("absolute thresholds" if not gate["baseline"] else
                "baseline" if gate["fail_on"] == "none" else "absolute thresholds and baseline")
        return f"  gate: pass ({rule}){tail}"
    return "  gate: FAIL; " + "; ".join(gate["reasons"]) + tail


def _query_ref_ok(base, x) -> bool:
    """A baseline is the query_shift reference only when it holds query distances and was
    audited on the same corpus (or an unchanged prefix): after a transform its distances
    are in another space."""
    from vecscope.evidence import same_corpus

    return _has_query_gap(base) and same_corpus(base, x)


def _query_ref_note(base, x, q) -> str | None:
    if base is None or q is None or not _has_query_gap(base) or _query_ref_ok(base, x):
        return None
    return ("query_shift skipped: the baseline was audited on a different corpus (rows "
            "changed or transformed), so its query distances are not a reference here")


def _has_query_gap(ev) -> bool:
    """Whether earlier evidence carries the query distances query_shift compares against."""
    return isinstance(ev, dict) and isinstance(ev.get("findings"), list) and any(
        isinstance(f, dict) and f.get("name") == "query_gap" for f in ev["findings"])


def _load_json(path: str, option: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"{option} {path} is not a JSON file ({e})") from None


def _render_diff(d: dict, path: str) -> str:
    lines = [f"  baseline {path}: verdict {d['baseline_verdict']} -> {d['verdict']}"
             + ("" if d["comparable"] else "  (n, dim, k, sampling, thresholds or near-dup "
                                           "cutoff differ: values and verdicts are not directly "
                                           "comparable)")]
    for c in d["changes"]:
        mark = ("WORSE" if c["name"] in d["regressions"] else
                "better" if c["name"] in d["improvements"] else "")
        if c["name"] in d["value_regressions"]:
            mark = (mark + " " if mark else "") + f"VALUE WORSE by {c['worsening']:.4f} > {c['tolerance']:g}"
        lines.append(f"    {c['name']:<22} {c['baseline_value']:.4f} -> {c['value']:.4f}  "
                     f"({c['delta']:+.4f})  {c['baseline_verdict']} -> {c['verdict']}  {mark}")
    for name in d["only_in_new"]:
        lines.append(f"    {name:<22} new finding, not in baseline")
    for name in d["only_in_baseline"]:
        lines.append(f"    {name:<22} in baseline only")
    if d.get("retrieval_changes"):
        note = "" if d["retrieval_comparable"] else "  (different qrels or k: not comparable)"
        lines.append(f"  retrieval vs baseline{note}:")
        for r in d["retrieval_changes"]:
            lines.append(f"    {r['ranking']:<7} {r['metric']:<24} {r['baseline_value']:.4f} -> "
                         f"{r['value']:.4f}  ({r['delta']:+.4f})")
    if d["regressions"]:
        lines.append(f"  regressions: {', '.join(d['regressions'])}")
    if d["value_regressions"]:
        lines.append(f"  value regressions (beyond tolerance): {', '.join(d['value_regressions'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
