"""Structural type for cluster estimators used by TransductiveClusteringProcessor."""

from __future__ import annotations

from typing import Protocol

from ....types import IntegerArray, NumericArray


class TransductiveClusteringLike(Protocol):
    """
    Structural type for cluster estimators that only label their training data.

    Sklearn type stubs are incomplete for AgglomerativeClustering, DBSCAN, and
    HDBSCAN; this protocol gives a stable signature without per-class casts.
    """

    def fit_predict(
        self, X: NumericArray, y: IntegerArray | None = None
    ) -> IntegerArray: ...
