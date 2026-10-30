"""Audit time and peak memory against the number of rows, so the README performance table
can be re-measured on any machine and extended to the sizes a corpus actually has.

Each row is audited twice: every row (``--sample 0``) and the default sample, so the cost
of a full audit at that size and the cost the default keeps a bigger corpus at are both
visible. Peak memory is what NumPy allocates inside the audit as seen by ``tracemalloc``;
the input matrix is extra. An independent run (docs/verification.md) found this peak within
0.04 GiB of the OS peak and a single run within 2% of a three-run median, so one run is
enough. Timings are wall-clock on this machine and single process; the JSON records the
machine so results can be compared.

Run:   python examples/benchmark_scale.py --rows 10000 50000 --dim 768
       python examples/benchmark_scale.py --rows 100000 --dim 768 --corpus zipf --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import CORPORA, corpus, machine

from vecscope import __version__, audit


def measure(x: np.ndarray, k: int, sample: int | None) -> dict:
    tracemalloc.start()
    t = time.perf_counter()
    ev = audit(x, k=k, sample=sample)
    seconds = time.perf_counter() - t
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"rows": int(x.shape[0]), "dim": int(x.shape[1]), "k": k,
            "sample": 0 if sample is None else sample, "audited_rows": ev.n,
            "seconds": round(seconds, 2), "peak_gib": round(peak / 2**30, 2),
            "verdict": ev.verdict}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rows", type=int, nargs="+", default=[10_000, 50_000])
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--corpus", choices=CORPORA, default="clustered")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample", type=int, default=100_000, help="the default sample to time too")
    p.add_argument("--json", help="also write the rows and the machine description here")
    args = p.parse_args()
    rows = []
    print(f"vecscope {__version__}  corpus={args.corpus} dim={args.dim} k={args.k}")
    print("| rows | audited rows | seconds | peak memory (input excluded) |")
    print("|---|---|---|---|")
    for n in args.rows:
        x = corpus(args.corpus, n, args.dim, seed=args.seed)
        runs = [measure(x, args.k, None)]
        if n > args.sample:
            runs.append(measure(x, args.k, args.sample))
        for r in runs:
            rows.append(r)
            print(f"| {r['rows']:,} | {r['audited_rows']:,} | {r['seconds']} | {r['peak_gib']} GiB |")
        del x
    if args.json:
        payload = {"machine": machine(), "corpus": args.corpus, "seed": args.seed, "rows": rows}
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
