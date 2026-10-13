from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ...types import IntegerArray, NumericArray
from .core_point_labeling import CorePointLabeling
from .shared_neighbor_graph import SharedNeighborGraph
from .snn_metric import SNNMetric


@dataclass
class SharedNearestNeighborClustering:
    """
    Density-based shared nearest neighbor (SNN) clustering.

    Implements the SNN clustering algorithm of Ertoz, Steinbach, and Kumar
    (2003), which measures similarity by how many nearest neighbors two points
    share instead of by raw distance. This makes it robust to clusters of
    differing density and to high-dimensional data where raw distances lose
    contrast.

    1. Find the k nearest neighbors of every point, ordered by distance with
       ties broken by the smaller point index.
    2. Link two points only if each is in the other's neighbor list, and weight
       the link by the number of neighbors they share (see SharedNeighborGraph).
    3. Keep strong links, whose weight is at least min_shared_neighbors.
    4. Label points around core points (see CorePointLabeling).

    Attributes
    ----------
    n_neighbors : int
        Size k of each point's nearest neighbor list.
    min_shared_neighbors : int
        Minimum shared neighbor count for a link to be strong.
    min_samples : int
        Minimum number of strong links for a point to be a core point.
    metric : SNNMetric
        Distance metric used to find nearest neighbors.
    labels_ : IntegerArray | None
        Cluster labels assigned by the last fit call, if any.
    """

    n_neighbors: int
    min_shared_neighbors: int
    min_samples: int
    metric: SNNMetric
    labels_: IntegerArray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.validate_parameters(
            self.n_neighbors, self.min_shared_neighbors, self.min_samples
        )

    @staticmethod
    def validate_parameters(
        n_neighbors: int, min_shared_neighbors: int, min_samples: int
    ) -> None:
        """
        Validate SNN hyperparameters.

        Parameters
        ----------
        n_neighbors : int
            Size k of each point's nearest neighbor list.
        min_shared_neighbors : int
            Minimum shared neighbor count for a link to be strong.
        min_samples : int
            Minimum number of strong links for a point to be a core point.

        Raises
        ------
        ValueError
            If n_neighbors is not positive, min_shared_neighbors is outside
            [0, n_neighbors), or min_samples is outside [1, n_neighbors].
        """
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be greater than 0")
        if not 0 <= min_shared_neighbors < n_neighbors:
            raise ValueError(
                "min_shared_neighbors must be in the range [0, n_neighbors)"
            )
        if not 1 <= min_samples <= n_neighbors:
            raise ValueError("min_samples must be in the range [1, n_neighbors]")

    def fit(self, X: NumericArray, y: IntegerArray | None = None) -> Self:
        """
        Fit the shared nearest neighbor clustering algorithm.

        Parameters
        ----------
        X : NumericArray
            Feature matrix with shape (n, d), or a precomputed distance matrix
            with shape (n, n) when metric is SNNMetric.PRECOMPUTED.
        y : IntegerArray | None
            Ignored; present for sklearn API compatibility.

        Returns
        -------
        Self
            The fitted instance.
        """
        del y
        self.labels_ = self._run(X)
        return self

    def fit_predict(
        self, X: NumericArray, y: IntegerArray | None = None
    ) -> IntegerArray:
        """
        Fit the algorithm and return the resulting cluster labels.

        Parameters
        ----------
        X : NumericArray
            Feature matrix with shape (n, d), or a precomputed distance matrix
            with shape (n, n) when metric is SNNMetric.PRECOMPUTED.
        y : IntegerArray | None
            Ignored; present for sklearn API compatibility.

        Returns
        -------
        IntegerArray
            Cluster label per point; noise points are labeled -1.
        """
        del y
        labels = self._run(X)
        self.labels_ = labels
        return labels

    def _run(self, X: NumericArray) -> IntegerArray:
        self._validate_input(X)
        strong_links = SharedNeighborGraph.from_neighbors(
            self._nearest_neighbors(X)
        ).strong_links(self.min_shared_neighbors)
        return CorePointLabeling(
            strong_links=strong_links,
            n_samples=len(X),
            min_samples=self.min_samples,
        ).labels()

    def _validate_input(self, X: NumericArray) -> None:
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        n_samples = len(X)
        if self.metric == SNNMetric.PRECOMPUTED and len(X.T) != n_samples:
            raise ValueError(
                "X must be a square distance matrix when metric is precomputed"
            )
        if self.n_neighbors >= n_samples:
            raise ValueError("n_neighbors must be less than the number of samples")

    def _nearest_neighbors(self, X: NumericArray) -> IntegerArray:
        distances, indices = (
            NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric.value)
            .fit(X)
            .kneighbors()
        )
        return self.sort_neighbors(distances, indices.astype(np.int64))

    @staticmethod
    def sort_neighbors(distances: NumericArray, indices: IntegerArray) -> IntegerArray:
        """
        Order each neighbor list by distance, breaking ties by point index.

        The neighbor search backend may return equidistant neighbors in any
        order; sorting makes the order, and therefore the nearest-first
        tie-breaking of border points, independent of the backend.

        Parameters
        ----------
        distances : NumericArray
            Neighbor distances with shape (n, k).
        indices : IntegerArray
            Neighbor indices with shape (n, k), aligned with distances.

        Returns
        -------
        IntegerArray
            Neighbor indices with shape (n, k), each row sorted by
            (distance, index).
        """
        order = np.lexsort((indices, distances), axis=1)
        return np.take_along_axis(indices, order, axis=1)
