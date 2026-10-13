from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..method import ClusteringMethod
from ..result import ClusteringLabels
from ..types import NumericArray
from .types import NegativeMatrix


@dataclass
class CannotLinkClusteringProcessor(ABC):
    """
    Unified interface for clustering under hard cannot-link constraints.

    Constrained clustering only labels its training data, so this interface
    offers no ``predict``.

    Attributes
    ----------
    method : ClusteringMethod
        The clustering method this processor implements.
    """

    method: ClusteringMethod

    @abstractmethod
    def fit(self, X: NumericArray, negative_matrix: NegativeMatrix) -> None:
        """
        Fit the cannot-link clustering processor; read the result via labels.

        Parameters
        ----------
        X : NumericArray
            Precomputed pairwise distance matrix.
        negative_matrix : NegativeMatrix
            Symmetric boolean mask marking cannot-link pairs.
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
