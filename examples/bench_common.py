"""Shared pieces of the benchmark scripts and the scale tests: seeded corpora of several
shapes and a description of the machine that every result file carries.

The corpus kinds come from the independent verification run (docs/verification.md). One
shape is not enough: the sweep recommendation moved from HNSW M=16 ef=16 on ``clustered`` to
M=32 ef=128 on ``zipf``, and ``clustered`` has n/200 centers, which lines up with the IVF
``nlist`` of 4*sqrt(n) near 500k rows and makes IVF look better than real data would.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

CORPORA = ("clustered", "isotropic", "anisotropic", "duplicates", "lowrank", "zipf")


def corpus(kind: str, n: int, dim: int, seed: int = 0, centers: int | None = None) -> np.ndarray:
    """Seeded float32 unit vectors.

    - clustered: Gaussian clusters, ``centers`` of them (default max(20, n // 200)), spread 0.8
    - isotropic: pure Gaussian noise, no neighbor structure (hardest for ANN recall)
    - anisotropic: clustered plus a shared offset, mean random cosine about 0.6
    - duplicates: clustered with 10% of rows copied (half exact, half 1e-4 jitter)
    - lowrank: 32 intrinsic dimensions embedded in ``dim``
    - zipf: clustered with Zipf-distributed cluster sizes (a few huge topics)
    """
    rng = np.random.default_rng(seed)
    n_c = centers if centers else max(20, n // 200)
    if kind == "isotropic":
        x = rng.standard_normal((n, dim), dtype=np.float32)
    elif kind == "lowrank":
        r = min(32, dim)
        basis = np.linalg.qr(rng.standard_normal((dim, r)))[0].astype(np.float32)
        z = _clustered(rng, n, r, n_c, spread=0.8)
        x = z @ basis.T + rng.standard_normal((n, dim), dtype=np.float32) * 0.01
    elif kind == "zipf":
        p = 1.0 / np.arange(1, n_c + 1) ** 1.1
        c = rng.standard_normal((n_c, dim)).astype(np.float32)
        x = c[rng.choice(n_c, n, p=p / p.sum())]
        x += rng.standard_normal((n, dim), dtype=np.float32) * 0.8
    elif kind in ("clustered", "anisotropic", "duplicates"):
        x = _clustered(rng, n, dim, n_c, spread=0.8)
        if kind == "anisotropic":
            off = rng.standard_normal(dim).astype(np.float32)
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            x += off / np.linalg.norm(off) * 1.25
        elif kind == "duplicates":
            m = n // 10
            src = rng.choice(n - m, m, replace=False)
            x[n - m:] = x[src]
            half = m // 2
            x[n - m + half:] += rng.standard_normal((m - half, dim), dtype=np.float32) * 1e-4
    else:
        raise ValueError(f"unknown corpus kind {kind!r}; known: {CORPORA}")
    x = x.astype(np.float32, copy=False)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _clustered(rng, n, dim, n_c, spread):
    c = rng.standard_normal((n_c, dim)).astype(np.float32)
    return c[rng.integers(0, n_c, n)] + rng.standard_normal((n, dim), dtype=np.float32) * spread


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, check=False, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def machine() -> dict:
    """What a result file needs so someone else can tell whether their numbers compare."""
    import vecscope

    info = {
        "vecscope": vecscope.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_model": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "mem_total_gib": None,
        "gpu": _cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader"]) or None,
        "faiss": None,
        "faiss_gpus": 0,
        "threads": {k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                info["mem_total_gib"] = round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass
    try:
        import faiss

        info["faiss"] = faiss.__version__
        info["faiss_gpus"] = int(faiss.get_num_gpus()) if hasattr(faiss, "get_num_gpus") else 0
    except ImportError:
        pass
    return info
