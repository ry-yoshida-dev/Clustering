"""Structural type for sklearn GaussianMixture."""

from __future__ import annotations

from typing import Protocol

from ...types import IntegerArray, NumericArray


class GaussianMixtureLike(Protocol):
    """
    Structural type for sklearn GaussianMixture.

    Sklearn type stubs leave fit, predict, and fit_predict partially unknown; this
    protocol gives stable signatures for type checking without sprinkling ignores
    on every call site.
    """

    def fit(self, X: NumericArray, y: IntegerArray | None = None) -> object: ...

    def predict(self, X: NumericArray) -> IntegerArray: ...

    def fit_predict(
        self, X: NumericArray, y: IntegerArray | None = None
    ) -> IntegerArray: ...
