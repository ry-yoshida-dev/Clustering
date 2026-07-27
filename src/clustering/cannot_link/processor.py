from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..method import ClusteringMethod
from ..result import ClusteringLabels
from ..types import NumericArray
from .types import NegativeMatrix


@dataclass
class CannotLinkClusteringProcessor(ABC):
    method: ClusteringMethod

    @abstractmethod
    def fit(self, X: NumericArray, negative_matrix: NegativeMatrix) -> None:
        """
        Fit the cannot-link clustering processor.
        Fit function does not return labels(we can get labels from labels_ property).
        -> Usually, fit_predict() is used.

        Parameters
        ----------
        X : NumericArray
            Precomputed pairwise distance matrix.
        negative_matrix : NegativeMatrix
            Symmetric boolean mask marking cannot-link pairs.
        """

    @abstractmethod
    def predict(self, X: NumericArray) -> ClusteringLabels:
        """
        Predict the clustering labels for new data.

        Parameters
        ----------
        X : NumericArray
            Unused; present for the CannotLinkClusteringProcessor interface.

        Raises
        ------
        ValueError
            Always; cannot-link processors do not support predicting on new data.
        """

    @abstractmethod
    def fit_predict(
        self, X: NumericArray, negative_matrix: NegativeMatrix
    ) -> ClusteringLabels:
        """
        Fit the cannot-link clustering processor and predict the clustering labels.

        Parameters
        ----------
        X : NumericArray
            Precomputed pairwise distance matrix.
        negative_matrix : NegativeMatrix
            Symmetric boolean mask marking cannot-link pairs.

        Returns
        -------
        ClusteringLabels
            The clustering labels.
        """

    @property
    @abstractmethod
    def labels(self) -> ClusteringLabels:
        """
        Get the clustering labels after fitting.
        If the processor is not fitted yet, it will raise an error.

        Returns
        -------
        ClusteringLabels
            The clustering labels.
        """

    @property
    @abstractmethod
    def is_precomputed_input_required(self) -> bool:
        """
        Check if the input data is precomputed.

        Returns
        -------
        bool
            True if the input data is precomputed, False otherwise.
        """
