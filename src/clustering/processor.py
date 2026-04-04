from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .method import ClusteringMethod
from .result import ClusteringLabels


@dataclass
class ClusteringProcessor(ABC):
    method: ClusteringMethod

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the clustering processor.
        Fit function does not return labels(we can get labels from labels_ property).
        -> Usually, fit_predict() is used.

        Parameters:
        ----------
        X: np.ndarray
            The input data.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> ClusteringLabels:
        """
        Predict the clustering labels for new data.
        Predict function is implemented in KMeans and GMM.

        Parameters:
        ----------
        X: np.ndarray
            The input data.

        Returns:
        ----------
        ClusteringLabels:
            The clustering labels.
        """

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> ClusteringLabels:
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

    @property
    @abstractmethod
    def labels(self) -> ClusteringLabels:
        """
        Get the clustering labels after fitting.
        If the processor is not fitted yet, it will raise an error.

        Returns:
        ----------
        ClusteringLabels:
            The clustering labels.
        """

    @property
    @abstractmethod
    def is_precomputed_input_required(self) -> bool:
        """
        Check if the input data is precomputed.

        Returns:
        ----------
        bool
            True if the input data is precomputed, False otherwise.
        """
