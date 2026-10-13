from __future__ import annotations

from dataclasses import dataclass
from typing import override

from ...result import ClusteringLabels
from ...types import IntegerArray, NumericArray
from ..processor import ClusteringProcessor
from .protocols.transductive import TransductiveClusteringLike


@dataclass
class TransductiveClusteringProcessor(ClusteringProcessor):
    """
    Wraps cluster estimators that only support labeling their training data.

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

    @override
    def predict(self, X: NumericArray) -> ClusteringLabels:
        """
        Predict is not supported for this processor type.

        Parameters
        ----------
        X : NumericArray
            Unused; present for the ClusteringProcessor interface.

        Raises
        ------
        ValueError
            Always, because these estimators cannot label unseen data.
        """
        raise ValueError(f"predict() is not supported for {type(self.processor)}")

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
