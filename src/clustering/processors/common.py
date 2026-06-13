from __future__ import annotations

from dataclasses import dataclass

from sklearn.cluster import (  # type: ignore[import-untyped]
    AgglomerativeClustering,
    DBSCAN,
    HDBSCAN,
)

from ..types import NumericArray
from ..processor import ClusteringProcessor
from ..result import ClusteringLabels
from .protocols.common import CommonClusteringLike


@dataclass
class CommonClusteringProcessor(ClusteringProcessor):
    """
    Wraps sklearn cluster estimators that share fit and fit_predict only.

    Used for AgglomerativeClustering, DBSCAN, HDBSCAN, and similar estimators. predict is
    not supported at the sklearn level for these models in this wrapper.

    Attributes:
    ----------
    processor: CommonClusteringLike
        The underlying sklearn estimator (structural type).
    """

    processor: CommonClusteringLike

    def fit(
        self,
        X: NumericArray,
    ) -> None:
        """
        Fit the clustering processor.

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
        Predict is not supported for this processor type.

        Parameters:
        ----------
        X: NumericArray
            Unused; present for the ClusteringProcessor interface.

        Raises:
        ----------
        ValueError:
            Always, because these estimators do not support predict in this wrapper.
        """
        raise ValueError(f"predict() is not supported for {type(self.processor)}")

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
    def is_precomputed_input_required(self) -> bool:
        """
        Check if the input data must be a precomputed distance matrix.

        Returns:
        ----------
        bool:
            True when the metric is precomputed (AgglomerativeClustering, DBSCAN, or HDBSCAN).
        """
        from ..methods import AgglomerativeMetric
        from ..methods import DBSCANMetric
        from ..methods import HDBSCANMetric

        if isinstance(self.processor, AgglomerativeClustering):
            return AgglomerativeMetric(self.processor.metric) == AgglomerativeMetric.PRECOMPUTED
        if isinstance(self.processor, DBSCAN):
            return DBSCANMetric(self.processor.metric) == DBSCANMetric.PRECOMPUTED
        if isinstance(self.processor, HDBSCAN):
            return HDBSCANMetric(self.processor.metric) == HDBSCANMetric.PRECOMPUTED

        return False
