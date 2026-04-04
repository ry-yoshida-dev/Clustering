from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans  # type: ignore[import-untyped]

from ..processor import ClusteringProcessor
from ..result import ClusteringLabels


@dataclass
class KMeansClusteringProcessor(ClusteringProcessor):
    """
    Wraps sklearn KMeans for the clustering pipeline.

    Attributes:
    ----------
    processor: KMeans
        The underlying sklearn KMeans estimator.
    """

    processor: KMeans

    def fit(
        self,
        X: np.ndarray,
    ) -> None:
        """
        Fit the KMeans clustering processor.

        Parameters:
        ----------
        X: np.ndarray
            The input data.
        """
        self.processor.fit(X)  # type: ignore[union-attr]

    def predict(
        self,
        X: np.ndarray,
    ) -> ClusteringLabels:
        """
        Predict the clustering labels for new data.

        Parameters:
        ----------
        X: np.ndarray
            The input data.

        Returns:
        ----------
        ClusteringLabels:
            The clustering labels.
        """
        return ClusteringLabels(labels=self.processor.predict(X))  # type: ignore[arg-type]

    def fit_predict(
        self,
        X: np.ndarray,
    ) -> ClusteringLabels:
        """
        Fit the clustering processor and predict the clustering labels.

        Parameters:
        ----------
        X: np.ndarray
            The input data.

        Returns:
        ----------
        ClusteringLabels:
            The clustering labels.
        """
        return ClusteringLabels(labels=self.processor.fit_predict(X))  # type: ignore[arg-type]

    @property
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
        return ClusteringLabels(labels=labels)  # type: ignore[arg-type]

    @property
    def is_precomputed_input_required(self) -> bool:
        """
        Check if the input data is precomputed.

        Returns:
        ----------
        bool:
            False because KMeans uses feature vectors, not a precomputed distance matrix.
        """
        return False
