from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..processor import ClusteringProcessor
from ..result import ClusteringLabels


class _GaussianMixtureLike(Protocol):
    """
    Structural type for sklearn GaussianMixture.

    Sklearn type stubs leave fit, predict, and fit_predict partially unknown; this
    protocol gives stable signatures for type checking without sprinkling ignores
    on every call site.
    """

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> object: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray: ...


@dataclass
class GMMClusteringProcessor(ClusteringProcessor):
    """
    Wraps sklearn GaussianMixture for the clustering pipeline.

    Attributes:
    ----------
    processor: _GaussianMixtureLike
        The underlying sklearn GaussianMixture estimator (structural type).
    labels_: np.ndarray | None
        Cached labels from the last predict or fit_predict, if any.
    """

    processor: _GaussianMixtureLike
    labels_: np.ndarray | None = field(default=None, init=False)

    def fit(
        self, 
        X: np.ndarray
        ) -> None:
        """
        Fit the GMM clustering processor.

        Parameters:
        ----------
        X: np.ndarray
            The input data.
        """
        self.processor.fit(X)

    def predict(
        self, 
        X: np.ndarray
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
        labels = self.processor.predict(X)
        self.labels_ = labels
        return ClusteringLabels(labels=labels)

    def fit_predict(
        self, 
        X: np.ndarray
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
        labels = self.processor.fit_predict(X)
        self.labels_ = labels
        return ClusteringLabels(labels=labels)

    @property
    def labels(self) -> ClusteringLabels:
        """
        Get the clustering labels after fitting.

        Returns:
        ----------
        ClusteringLabels:
            The clustering labels.
        """
        if self.labels_ is None:
            raise ValueError(
                "Processor is not fitted yet. Call fit() or fit_predict() first."
            )
        return ClusteringLabels(labels=self.labels_)

    @property
    def is_precomputed_input_required(self) -> bool:
        """
        Check if the input data is precomputed.

        Returns:
        ----------
        bool:
            False because GMM does not require a precomputed distance matrix.
        """
        return False