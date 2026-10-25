"""vecscope CLI: `vecscope audit embeddings.npy`."""

from __future__ import annotations

import argparse
import json
import sys

from vecscope import __version__
from vecscope.evidence import audit
from vecscope.io import load_vectors
from vecscope.sweep import render_sweep, sweep


def _render_text(ev) -> str:
    mark = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}
    head = f"vecscope {ev.version}  n={ev.n} dim={ev.dim} k={ev.k}  verdict={ev.verdict.upper()}"
    if ev.sampled:
        head += f"  (sampled from {ev.n_total} rows)"
    if ev.n_queries:
        head += f"  queries={ev.n_queries}"
    lines = [head]
    for f in ev.findings:
        lines.append(
            f"  [{mark[f.verdict]}] {f.name:<16} {f.metric}={f.value:.4f}  ({f.healthy_range})"
        )
        lines.append(f"{'':25}{f.explanation}")
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
    a.add_argument("--ann", help="optional .npy of ANN neighbor indices (n x k) to score recall")
    a.add_argument("--queries", help="optional query embeddings (m x dim) for query-side findings")
    a.add_argument("--clusters", type=int, default=0,
                   help="attach a per-cluster report with this many clusters (0 = off)")
    a.add_argument("--near-dup", type=float, default=0.99, help="cosine threshold for near-dups")
    a.add_argument("--sample", type=int, default=100_000,
                   help="audit a seeded random subset when n exceeds this (0 = all rows)")
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
    s.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = p.parse_args(argv)

    x = load_vectors(args.path)
    if args.cmd == "sweep":
        q = load_vectors(args.queries) if args.queries else None
        res = sweep(x, k=args.k, queries=q, n_queries=args.n_queries,
                    target_recall=args.target_recall, device=args.device)
        print(json.dumps(res.to_dict(), indent=2) if args.json else render_sweep(res))
        return 0

    ann = load_vectors(args.ann).astype(int) if args.ann else None
    q = load_vectors(args.queries) if args.queries else None
    ev = audit(x, k=args.k, ann_neighbors=ann, near_dup_cos=args.near_dup,
               sample=args.sample or None, queries=q, clusters=args.clusters)
    print(ev.to_json() if args.json else _render_text(ev))
    level = {"ok": 0, "warn": 1, "fail": 2}
    return 1 if level[ev.verdict] >= level[args.fail_on] else 0


if __name__ == "__main__":
    sys.exit(main())
