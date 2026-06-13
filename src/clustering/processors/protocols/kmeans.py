"""Structural type for sklearn KMeans."""

from __future__ import annotations

from typing import Protocol

from ...types import IntegerArray, NumericArray


class KMeansLike(Protocol):
    """
    Structural type for sklearn KMeans.

    Sklearn type stubs are incomplete for KMeans; this protocol gives stable
    signatures for fit, predict, fit_predict, and labels_ without per-call ignores.
    """

    labels_: IntegerArray | None

    def fit(self, X: NumericArray, y: IntegerArray | None = None) -> object: ...

    def predict(self, X: NumericArray) -> IntegerArray: ...

    def fit_predict(
        self, X: NumericArray, y: IntegerArray | None = None
    ) -> IntegerArray: ...
