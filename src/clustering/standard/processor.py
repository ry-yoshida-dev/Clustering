from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sklearn.exceptions import NotFittedError

from ..method import ClusteringMethod
from ..result import ClusteringLabels
from ..types import IntegerArray, NumericArray


@dataclass
class ClusteringProcessor(ABC):
    """
    Unified interface over a fitted-on-demand clustering estimator.

    Subclasses provide the raw label computation; this base class caches the
    labels of the last fit so that ``labels`` is available after either ``fit``
    or ``fit_predict``. Labeling unseen data is not part of this interface;
    only InductiveClusteringProcessor offers ``predict``.

    Attributes
    ----------
    method : ClusteringMethod
        The clustering method this processor implements.
    """

    method: ClusteringMethod
    _fitted_labels: ClusteringLabels | None = field(
        default=None, init=False, repr=False
    )

    def fit(self, X: NumericArray) -> None:
        """
        Fit the clustering processor and cache the training labels.

        Parameters
        ----------
        X : NumericArray
            The input data.
        """
        _ = self.fit_predict(X)

    def fit_predict(self, X: NumericArray) -> ClusteringLabels:
        """
        Fit the clustering processor and return the training labels.

        Parameters
        ----------
        X : NumericArray
            The input data.

        Returns
        -------
        ClusteringLabels
            The clustering labels of the training data.
        """
        fitted_labels = ClusteringLabels(labels=self._fit_predict_labels(X))
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
    @abstractmethod
    def is_precomputed_input_required(self) -> bool:
        """
        Check if the input data must be a precomputed distance matrix.

        Returns
        -------
        bool
            True if the input data must be a precomputed distance matrix.
        """

    @abstractmethod
    def _fit_predict_labels(self, X: NumericArray) -> IntegerArray: ...
