"""Generate small example matrices: healthy.npy, anisotropic.npy, collapsed.npy.

Vectors are L2-normalized like the output of most embedding models.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)


def unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


np.save(HERE / "healthy.npy", unit(rng.standard_normal((3000, 128))))
np.save(HERE / "anisotropic.npy", unit(rng.standard_normal((3000, 128)) * 0.3 + 2.0))
base = unit(rng.standard_normal((1000, 128)))
np.save(HERE / "collapsed.npy", np.concatenate([base, base[:600], base[:400]]))
print("wrote healthy.npy anisotropic.npy collapsed.npy")
