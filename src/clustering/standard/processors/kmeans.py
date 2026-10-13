from __future__ import annotations

from dataclasses import dataclass
from typing import override

from ...result import ClusteringLabels
from ...types import NumericArray
from ..processor import ClusteringProcessor
from .protocols.kmeans import KMeansLike


@dataclass
class KMeansClusteringProcessor(ClusteringProcessor):
    """
    Wraps sklearn KMeans for the clustering pipeline.

    Attributes:
    ----------
    processor: KMeansLike
        The underlying sklearn KMeans estimator (structural type).
    """

    processor: KMeansLike

    @override
    def fit(
        self,
        X: NumericArray,
    ) -> None:
        """
        Fit the KMeans clustering processor.

        Parameters:
        ----------
        X: NumericArray
            The input data.
        """
        _ = self.processor.fit(X)

    @override
    def predict(
        self,
        X: NumericArray,
    ) -> ClusteringLabels:
        """
        Predict the clustering labels for new data.

        Parameters:
        ----------
        X: NumericArray
            The input data.

        Returns:
        ----------
        ClusteringLabels:
            The clustering labels.
        """
        return ClusteringLabels(labels=self.processor.predict(X))

    @override
    def fit_predict(
        self,
        X: NumericArray,
    ) -> ClusteringLabels:
        """
        Fit the clustering processor and predict the clustering labels.

        Parameters:
        ----------
        X: NumericArray
            The input data.

        Returns:
        ----------
        ClusteringLabels:
            The clustering labels.
        """
        return ClusteringLabels(labels=self.processor.fit_predict(X))

    @property
    @override
    def labels(self) -> ClusteringLabels:
        """
        Get the clustering labels after fitting.

        Returns:
        ----------
        ClusteringLabels:
            The clustering labels.
        """
        labels = self.processor.labels_
        if labels is None:
            raise ValueError(
                "Processor is not fitted yet. Call fit() or fit_predict() first."
            )
        return ClusteringLabels(labels=labels)

    @property
    @override
    def is_precomputed_input_required(self) -> bool:
        """
        Check if the input data is precomputed.

        Returns:
        ----------
        bool:
            False because KMeans uses feature vectors, not a precomputed distance matrix.
        """
        return False
