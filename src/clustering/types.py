"""Shared type aliases for clustering data."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


type NumericArray = NDArray[np.integer[Any] | np.floating[Any]]
type IntegerArray = NDArray[np.integer[Any]]
type BoolArray = NDArray[np.bool_]
