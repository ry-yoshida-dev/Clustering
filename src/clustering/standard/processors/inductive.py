from __future__ import annotations

from dataclasses import dataclass
from typing import override

from ...result import ClusteringLabels
from ...types import IntegerArray, NumericArray
from ..processor import ClusteringProcessor
from .protocols.inductive import InductiveClusteringLike


@dataclass
class InductiveClusteringProcessor(ClusteringProcessor):
    """
    Wraps cluster estimators that can also label unseen data.

    Used for KMeans and GaussianMixture, which operate on feature vectors.

    Attributes
    ----------
    processor : InductiveClusteringLike
        The underlying estimator (structural type).
    """

    processor: InductiveClusteringLike

    @override
    def predict(self, X: NumericArray) -> ClusteringLabels:
        """
        Predict the clustering labels for new data.

        Parameters
        ----------
        X : NumericArray
            The input data.

        Returns
        -------
        ClusteringLabels
            The clustering labels.
        """
        return ClusteringLabels(labels=self.processor.predict(X))

    @property
    @override
    def is_precomputed_input_required(self) -> bool:
        """
        Check if the input data must be a precomputed distance matrix.

        Returns
        -------
        bool
            Always False; these estimators operate on feature vectors.
        """
        return False

    @override
    def _fit_predict_labels(self, X: NumericArray) -> IntegerArray:
        return self.processor.fit_predict(X)
