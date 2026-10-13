from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sklearn.exceptions import NotFittedError

from ..method import ClusteringMethod
from ..result import ClusteringLabels
from ..types import IntegerArray, NumericArray
from .types import NegativeMatrix


@dataclass
class CannotLinkClusteringProcessor(ABC):
    """
    Unified interface for clustering under hard cannot-link constraints.

    Subclasses provide the raw label computation; this base class caches the
    labels of the last fit so that ``labels`` is available after either ``fit``
    or ``fit_predict``. Constrained clustering only labels its training data,
    so this interface offers no ``predict``.

    Attributes
    ----------
    method : ClusteringMethod
        The clustering method this processor implements.
    """

    method: ClusteringMethod
    _fitted_labels: ClusteringLabels | None = field(
        default=None, init=False, repr=False
    )

    def fit(self, X: NumericArray, negative_matrix: NegativeMatrix) -> None:
        """
        Fit the cannot-link clustering processor and cache the training labels.

        Parameters
        ----------
        X : NumericArray
            Precomputed pairwise distance matrix.
        negative_matrix : NegativeMatrix
            Symmetric boolean mask marking cannot-link pairs.
        """
        _ = self.fit_predict(X, negative_matrix)

    def fit_predict(
        self, X: NumericArray, negative_matrix: NegativeMatrix
    ) -> ClusteringLabels:
        """
        Fit the cannot-link clustering processor and return the training labels.

        Parameters
        ----------
        X : NumericArray
            Precomputed pairwise distance matrix.
        negative_matrix : NegativeMatrix
            Symmetric boolean mask marking cannot-link pairs.

        Returns
        -------
        ClusteringLabels
            The clustering labels of the training data.
        """
        fitted_labels = ClusteringLabels(
            labels=self._fit_predict_labels(X, negative_matrix)
        )
        self._fitted_labels = fitted_labels
        return fitted_labels

    @property
    def labels(self) -> ClusteringLabels:
        """
        Get the clustering labels of the training data from the last fit.

        Returns
        -------
        ClusteringLabels
            The clustering labels.

        Raises
        ------
        NotFittedError
            If neither fit nor fit_predict has been called yet.
        """
        if self._fitted_labels is None:
            raise NotFittedError(
                "Processor is not fitted yet. Call fit() or fit_predict() first."
            )
        return self._fitted_labels

    @property
    def is_precomputed_input_required(self) -> bool:
        """
        Check if the input data must be a precomputed distance matrix.

        Returns
        -------
        bool
            Always True; cannot-link clustering operates on precomputed
            distance matrices only.
        """
        return True

    @abstractmethod
    def _fit_predict_labels(
        self, X: NumericArray, negative_matrix: NegativeMatrix
    ) -> IntegerArray: ...
