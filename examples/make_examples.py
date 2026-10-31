"""Generate small example matrices: healthy.npy, anisotropic.npy, collapsed.npy, and the
golden evidence for them under tests/golden/.

Vectors are L2-normalized like the output of most embedding models. Generation is seeded, so
the matrices (and therefore the golden files) are the same on every machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GOLD = HERE.parent / "tests" / "golden"


def unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def structured(rng: np.random.Generator, n: int, centers: int = 60, spread: float = 0.8) -> np.ndarray:
    """Clustered vectors: topical structure like a real corpus (pure noise has no neighbors)."""
    c = rng.standard_normal((centers, 128))
    return c[rng.integers(0, centers, n)] + rng.standard_normal((n, 128)) * spread


def make_examples(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    healthy = unit(structured(rng, 3000))
    anisotropic = unit(structured(rng, 3000) * 0.3 + 1.0)
    base = unit(structured(rng, 1000))
    collapsed = np.concatenate([base, base[:600], base[:400]])
    return {"healthy": healthy, "anisotropic": anisotropic, "collapsed": collapsed}


def main() -> None:
    examples = make_examples()
    for name, x in examples.items():
        np.save(HERE / f"{name}.npy", x)
    print("wrote " + " ".join(f"{name}.npy" for name in examples))

    # golden evidence for tests/test_audit.py::test_golden_evidence_for_shipped_examples,
    # only when the script runs inside the vecscope repository (it can also be downloaded on
    # its own, and must not write into a stranger's tests/ directory then)
    if not (HERE.parent / "src" / "vecscope" / "evidence.py").is_file():
        return
    from vecscope import audit

    GOLD.mkdir(parents=True, exist_ok=True)
    for name, x in examples.items():
        ev = audit(x, k=10).to_dict()
        ev.pop("version")  # pinned values, not the release that wrote them
        (GOLD / f"{name}.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    print("wrote tests/golden/*.json")


if __name__ == "__main__":
    main()
