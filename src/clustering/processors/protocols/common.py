"""Structural type for sklearn cluster estimators used by CommonClusteringProcessor."""

from __future__ import annotations

from typing import Protocol

from ...types import IntegerArray, NumericArray


class CommonClusteringLike(Protocol):
    """
    Structural type for sklearn cluster estimators with fit and fit_predict only.

    Sklearn type stubs are incomplete for AgglomerativeClustering, DBSCAN, and
    HDBSCAN; this protocol gives stable signatures without per-class casts.
    """

    labels_: IntegerArray | None

    def fit(self, X: NumericArray, y: IntegerArray | None = None) -> object: ...

    def fit_predict(
        self, X: NumericArray, y: IntegerArray | None = None
    ) -> IntegerArray: ...
