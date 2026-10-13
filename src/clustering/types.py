"""Shared type aliases for clustering data."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type NumericArray = NDArray[np.integer | np.floating]
type FloatArray = NDArray[np.floating]
type IntegerArray = NDArray[np.integer]
type BoolArray = NDArray[np.bool_]
