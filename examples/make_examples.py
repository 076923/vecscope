"""Generate small example matrices: healthy.npy, anisotropic.npy, collapsed.npy.

Vectors are L2-normalized like the output of most embedding models.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)


def unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)




def structured(n: int, centers: int = 60, spread: float = 0.8) -> np.ndarray:
    """Clustered vectors: topical structure like a real corpus (pure noise has no neighbors)."""
    c = rng.standard_normal((centers, 128))
    return c[rng.integers(0, centers, n)] + rng.standard_normal((n, 128)) * spread


np.save(HERE / "healthy.npy", unit(structured(3000)))
np.save(HERE / "anisotropic.npy", unit(structured(3000) * 0.3 + 1.0))
base = unit(structured(1000))
np.save(HERE / "collapsed.npy", np.concatenate([base, base[:600], base[:400]]))
print("wrote healthy.npy anisotropic.npy collapsed.npy")
