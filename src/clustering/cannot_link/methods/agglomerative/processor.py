from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np

from ....types import IntegerArray, NumericArray
from ...processor import CannotLinkClusteringProcessor
from ...types import NegativeMatrix
from .estimator import ConstrainedAgglomerativeClustering


@dataclass
class CannotLinkAgglomerativeProcessor(CannotLinkClusteringProcessor):
    """
    Wraps ConstrainedAgglomerativeClustering for the cannot-link clustering pipeline.

    Every result is checked against the cannot-link constraints before it is
    returned or cached; a violation raises RuntimeError, as it indicates an
    internal invariant failure rather than a usage error.

    Attributes
    ----------
    processor : ConstrainedAgglomerativeClustering
        The underlying constrained agglomerative clustering algorithm.
    """

    processor: ConstrainedAgglomerativeClustering

    @override
    def _fit_predict_labels(
        self, X: NumericArray, negative_matrix: NegativeMatrix
    ) -> IntegerArray:
        raw_labels = self.processor.fit_predict(X, negative_matrix)
        self._verify_no_violations(raw_labels, negative_matrix)
        return raw_labels

    @staticmethod
    def _verify_no_violations(
        raw_labels: IntegerArray, negative_matrix: NegativeMatrix
    ) -> None:
        same_cluster = np.equal.outer(raw_labels, raw_labels)
        if np.any(same_cluster & negative_matrix):
            raise RuntimeError(
                "cannot-link constraint violated in the clustering result; this indicates an internal invariant failure"
            )
