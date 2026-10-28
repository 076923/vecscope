"""vecscope CLI: `vecscope audit embeddings.npy`, `vecscope sweep`, `vecscope compare`."""

from __future__ import annotations

import argparse
import json
import sys

from vecscope import __version__
from vecscope.compare import compare, render_compare
from vecscope.evidence import audit
from vecscope.io import load_neighbors, load_vectors
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
        line = f"  [{mark[f.verdict]}] {f.name:<16} {f.metric}={f.value:.4f}  ({f.healthy_range})"
        if f.value_range is not None:
            line += f"  repeats {f.value_range[0]:.4f}~{f.value_range[1]:.4f}"
            if not f.verdict_stable:
                line += "  verdict UNSTABLE"
        lines.append(line)
        lines.append(f"{'':25}{f.explanation}")
        if f.verdict != "ok":
            lines.append(f"{'':25}try: {f.suggested_action}")
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
    a.add_argument("path", help=".npy/.npz/.parquet/.jsonl/.csv with one vector per row")
    a.add_argument("-k", type=int, default=10, help="neighbors per item (default 10)")
    a.add_argument("--ann", help="optional (n x k) neighbor ids from an ANN index, one row per "
                                 "embedding row; needs --sample 0; self ids are ignored")
    a.add_argument("--queries", help="optional query embeddings (m x dim) for query-side findings")
    a.add_argument("--clusters", type=int, default=0,
                   help="attach a per-cluster report with this many clusters (0 = off)")
    a.add_argument("--thresholds", help='JSON file of overrides, e.g. {"hubness.skewness": [2.5, 5]}')
    a.add_argument("--near-dup", type=float, default=0.99, help="cosine threshold for near-dups")
    a.add_argument("--sample", type=int, default=100_000,
                   help="audit a seeded random subset when n exceeds this (0 = all rows)")
    a.add_argument("--sample-repeats", type=int, default=1,
                   help="when sampling applies, re-audit this many independent samples and "
                        "report each finding's value range and verdict stability (default 1)")
    a.add_argument("--json", action="store_true", help="emit JSON evidence instead of text")
    a.add_argument("--fail-on", choices=["warn", "fail"], default="fail",
                   help="exit 1 when verdict is at least this level")
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
    s.add_argument("--json", action="store_true", help="emit JSON instead of text")
    c = sub.add_parser("compare", help="compare two embeddings of the same rows (no verdicts)")
    c.add_argument("path_a", help="embedding file A, one vector per row")
    c.add_argument("path_b", help="embedding file B, same rows in the same order")
    c.add_argument("-k", type=int, default=10, help="neighbors per item (default 10)")
    c.add_argument("--near-dup", type=float, default=0.99, help="cosine threshold for near-dups")
    c.add_argument("--sample", type=int, default=100_000,
                   help="compare a seeded random subset when n exceeds this (0 = all rows)")
    c.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = p.parse_args(argv)
    try:
        return _run(args)
    except (ValueError, TypeError, FileNotFoundError, ImportError, OSError, json.JSONDecodeError) as e:
        print(f"vecscope: error: {e}", file=sys.stderr)
        return 2


def _run(args) -> int:
    if args.cmd == "audit" and args.sample < 0:
        raise ValueError("--sample must be 0 (all rows) or a positive number")
    if args.cmd == "audit" and args.clusters < 0:
        raise ValueError("--clusters must be 0 or a positive number")
    if args.cmd == "audit" and args.sample_repeats < 1:
        raise ValueError("--sample-repeats must be a positive number")
    if args.cmd == "compare":
        if args.sample < 0:
            raise ValueError("--sample must be 0 (all rows) or a positive number")
        res = compare(load_vectors(args.path_a), load_vectors(args.path_b), k=args.k,
                      near_dup_cos=args.near_dup, sample=args.sample or None)
        print(json.dumps(res.to_dict(), indent=2) if args.json else render_compare(res))
        return 0
    x = load_vectors(args.path)
    if args.cmd == "sweep":
        q = load_vectors(args.queries) if args.queries else None
        res = sweep(x, k=args.k, queries=q, n_queries=args.n_queries,
                    target_recall=args.target_recall, device=args.device,
                    timing_repeats=args.timing_repeats, margin=args.margin)
        print(json.dumps(res.to_dict(), indent=2) if args.json else render_sweep(res))
        return 0

    ann = load_neighbors(args.ann) if args.ann else None
    q = load_vectors(args.queries) if args.queries else None
    th = None
    if args.thresholds:
        with open(args.thresholds, encoding="utf-8") as fh:
            th = json.load(fh)
    ev = audit(x, k=args.k, ann_neighbors=ann, near_dup_cos=args.near_dup,
               sample=args.sample or None, queries=q, clusters=args.clusters, thresholds=th,
               sample_repeats=args.sample_repeats)
    print(ev.to_json() if args.json else _render_text(ev))
    level = {"ok": 0, "warn": 1, "fail": 2}
    return 1 if level[ev.verdict] >= level[args.fail_on] else 0


if __name__ == "__main__":
    sys.exit(main())
