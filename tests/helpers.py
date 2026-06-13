from __future__ import annotations

import numpy as np

from clustering.types import NumericArray


def two_blobs(*, seed: int = 0) -> NumericArray:
    """Return two well-separated 2-D Gaussian blobs for clustering tests."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((24, 2))
    b = rng.standard_normal((24, 2)) + np.array([6.0, 6.0])
    return np.vstack([a, b])
