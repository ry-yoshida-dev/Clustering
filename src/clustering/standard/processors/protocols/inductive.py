"""Structural type for cluster estimators used by InductiveClusteringProcessor."""

from __future__ import annotations

from typing import Protocol

from ....types import IntegerArray, NumericArray


class InductiveClusteringLike(Protocol):
    """
    Structural type for cluster estimators that can also label unseen data.

    Sklearn type stubs leave KMeans and GaussianMixture partially unknown; this
    protocol gives stable signatures without per-call ignores.
    """

    def predict(self, X: NumericArray) -> IntegerArray: ...

    def fit_predict(
        self, X: NumericArray, y: IntegerArray | None = None
    ) -> IntegerArray: ...
