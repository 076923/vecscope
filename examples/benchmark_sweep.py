"""Index sweep at a chosen corpus size and shape: recall, qps, single-query latency
p50/p95/p99 and index size per HNSW/IVF setting, on this machine.

Run more than one ``--corpus``: the recommendation differs by shape (HNSW M=16 ef=16 on
``clustered``, M=32 ef=128 on ``zipf`` at 100k x 768 in docs/verification.md), and the
default ``clustered`` corpus has n/200 centers, which near 500k rows lines up with the IVF
nlist of 4*sqrt(n) and makes IVF look better than real data would; ``--centers`` fixes the
count independently of n.

Run:   python examples/benchmark_sweep.py --rows 100000 --dim 768 --corpus clustered zipf --json sweep.json
       (needs faiss: pip install faiss-cpu, or faiss-gpu-cu12 for --device gpu)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import CORPORA, corpus, machine

from vecscope import __version__
from vecscope.sweep import render_sweep, sweep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rows", type=int, default=100_000)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--corpus", choices=CORPORA, nargs="+", default=["clustered", "zipf"])
    p.add_argument("--centers", type=int, default=None,
                   help="fixed cluster count for clustered-type corpora (default n // 200)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--latency-queries", type=int, default=200)
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--json", help="write every result and the machine description here")
    args = p.parse_args()
    out = {"machine": machine(), "args": vars(args), "sweeps": []}
    for kind in args.corpus:
        print(f"vecscope {__version__}  corpus={kind} rows={args.rows} dim={args.dim} k={args.k}")
        x = corpus(kind, args.rows, args.dim, seed=args.seed, centers=args.centers)
        t = time.perf_counter()
        res = sweep(x, k=args.k, n_queries=args.n_queries, device=args.device,
                    latency_queries=args.latency_queries, seed=args.seed)
        print(render_sweep(res))
        print(f"total sweep time: {time.perf_counter() - t:.1f} s\n")
        out["sweeps"].append({"corpus": kind, **res.to_dict()})
        del x
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
