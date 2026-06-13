from __future__ import annotations

from dataclasses import dataclass, field

from ..types import IntegerArray, NumericArray
from ..processor import ClusteringProcessor
from ..result import ClusteringLabels
from .protocols.gmm import GaussianMixtureLike


@dataclass
class GMMClusteringProcessor(ClusteringProcessor):
    """
    Wraps sklearn GaussianMixture for the clustering pipeline.

    Attributes:
    ----------
    processor: GaussianMixtureLike
        The underlying sklearn GaussianMixture estimator (structural type).
    labels_: IntegerArray | None
        Cached labels from the last predict or fit_predict, if any.
    """

    processor: GaussianMixtureLike
    labels_: IntegerArray | None = field(default=None, init=False)

    def fit(
        self,
        X: NumericArray,
    ) -> None:
        """
        Fit the GMM clustering processor.

        Parameters:
        ----------
        X: NumericArray
            The input data.
        """
        self.processor.fit(X)

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
        labels = self.processor.predict(X)
        self.labels_ = labels
        return ClusteringLabels(labels=labels)

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
