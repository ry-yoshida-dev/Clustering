from __future__ import annotations

from dataclasses import dataclass
from typing import override

from ...types import IntegerArray, NumericArray
from ..processor import ClusteringProcessor
from .protocols.transductive import TransductiveClusteringLike


@dataclass
class TransductiveClusteringProcessor(ClusteringProcessor):
    """
    Wraps cluster estimators that only support labeling their training data.

    It deliberately offers no ``predict``, so labeling unseen data with these
    estimators is rejected by the type checker instead of at runtime.

    Used for AgglomerativeClustering, DBSCAN, HDBSCAN,
    SharedNearestNeighborClustering, and similar estimators.

    Attributes
    ----------
    processor : TransductiveClusteringLike
        The underlying estimator (structural type).
    is_precomputed_metric : bool
        Whether the estimator is configured with a precomputed distance metric.
    """

    processor: TransductiveClusteringLike
    is_precomputed_metric: bool

    @property
    @override
    def is_precomputed_input_required(self) -> bool:
        """
        Check if the input data must be a precomputed distance matrix.

        Returns
        -------
        bool
            True when the estimator is configured with a precomputed distance metric.
        """
        return self.is_precomputed_metric

    @override
    def _fit_predict_labels(self, X: NumericArray) -> IntegerArray:
        return self.processor.fit_predict(X)
