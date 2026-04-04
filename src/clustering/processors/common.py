from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike
from sklearn.cluster import (  # type: ignore[import-untyped]
    AgglomerativeClustering,
    DBSCAN,
    HDBSCAN,
)

from ..processor import ClusteringProcessor
from ..result import ClusteringLabels


class _SklearnClusteringLike(Protocol):
    """
    Structural type for sklearn cluster estimators used here.

    Sklearn type stubs are incomplete for these estimators; this protocol gives
    stable signatures for fit, fit_predict, and labels_ without per-class casts.
    """

    labels_: object | None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> object: ...

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray: ...


@dataclass
class CommonClusteringProcessor(ClusteringProcessor):
    """
    Wraps sklearn cluster estimators that share fit and fit_predict only.

    Used for AgglomerativeClustering, DBSCAN, HDBSCAN, and similar estimators. predict is
    not supported at the sklearn level for these models in this wrapper.

    Attributes:
    ----------
    processor: _SklearnClusteringLike
        The underlying sklearn estimator (structural type).
    """

    processor: _SklearnClusteringLike

    def fit(
        self,
        X: np.ndarray,
    ) -> None:
        """
        Fit the clustering processor.

        Parameters:
        ----------
        X: np.ndarray
            The input data.
        """
        self.processor.fit(X)

    def predict(
        self,
        X: np.ndarray,
    ) -> ClusteringLabels:
        """
        Predict is not supported for this processor type.

        Parameters:
        ----------
        X: np.ndarray
            Unused; present for the ClusteringProcessor interface.

        Raises:
        ----------
        ValueError:
            Always, because these estimators do not support predict in this wrapper.
        """
        raise ValueError(f"predict() is not supported for {type(self.processor)}")

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
        return ClusteringLabels(labels=cast(ArrayLike, self.processor.fit_predict(X)))

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
        return ClusteringLabels(labels=cast(ArrayLike, labels))

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
