from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from ...types import BoolArray, FloatArray, IntegerArray, NumericArray
from .metric import KReciprocalMetric
from .reciprocal_neighbors import KReciprocalNeighbors


@dataclass(frozen=True)
class KReciprocalReranker:
    """
    k-reciprocal re-ranking of pairwise distances (Zhong et al., 2017).

    Refines a pairwise distance matrix by comparing the k-reciprocal
    neighborhoods of points, so that points whose neighborhoods agree move
    closer and points that are only one-sidedly near move apart. The result is
    a precomputed distance matrix for DBSCAN, HDBSCAN, agglomerative, SNN, or
    cannot-link clustering.

    1. Compute the original distances and scale them into [0, 1] by their
       maximum, ranking points by distance with ties broken by the smaller
       point index and every point ranked first for itself.
    2. Find each point's k1-reciprocal neighbors and add the
       round(k1 / 2)-reciprocal neighbors of any of them that share more than
       two thirds of their own set with it.
    3. Encode each expanded set as a vector of Gaussian kernel weights
       exp(-distance), normalized to sum to one.
    4. Replace every vector by the mean vector of its k2 nearest points.
    5. Take the Jaccard distance between the vectors and blend it with the
       original distance: (1 - lambda) * jaccard + lambda * original.

    Unlike the reference implementation, which scales each column by its own
    maximum, distances are scaled by the global maximum so that the result
    stays symmetric. Time is O(n^2 log n + n^2 s), where s is the support size
    of a neighborhood vector after query expansion (roughly k1 * k2), and
    memory is O(n^2).

    Attributes
    ----------
    n_reciprocal_neighbors : int
        Neighborhood size k1 of the k-reciprocal sets. Must be positive.
    n_expansion_neighbors : int
        Number k2 of nearest points averaged in local query expansion. Must be
        positive; 1 disables the expansion.
    original_distance_weight : float
        Weight lambda of the original distance in the blend, in [0, 1]. The
        default 0 returns the pure Jaccard distance, as is common for
        clustering; the reference retrieval setting is 0.3.
    metric : KReciprocalMetric
        Original distance metric.
    """

    n_reciprocal_neighbors: int = 20
    n_expansion_neighbors: int = 6
    original_distance_weight: float = 0.0
    metric: KReciprocalMetric = KReciprocalMetric.EUCLIDEAN

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.n_reciprocal_neighbors <= 0:
            raise ValueError("n_reciprocal_neighbors must be greater than 0")
        if self.n_expansion_neighbors <= 0:
            raise ValueError("n_expansion_neighbors must be greater than 0")
        if not 0.0 <= self.original_distance_weight <= 1.0:
            raise ValueError("original_distance_weight must be in the range [0, 1]")

    def rerank(self, X: NumericArray) -> FloatArray:
        """
        Compute the re-ranked pairwise distance matrix.

        Parameters
        ----------
        X : NumericArray
            Feature matrix with shape (n, d), or a square distance matrix with
            shape (n, n) when metric is KReciprocalMetric.PRECOMPUTED.

        Returns
        -------
        FloatArray
            Symmetric distance matrix with shape (n, n), values in [0, 1], and
            a zero diagonal.

        Raises
        ------
        ValueError
            If X is not a finite 2-D array, if a precomputed X is not a square,
            non-negative, symmetric matrix, if the original distances are not
            finite (cosine distance with a zero vector), or if X has no more
            points than n_reciprocal_neighbors or fewer than
            n_expansion_neighbors.
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        n_samples: int = len(X)
        if n_samples <= self.n_reciprocal_neighbors:
            raise ValueError("X must have more points than n_reciprocal_neighbors")
        if n_samples < self.n_expansion_neighbors:
            raise ValueError("X must have at least n_expansion_neighbors points")
        original_distances = self._original_distances(X)
        ranking = self._ranking(original_distances)
        neighborhood_vectors = self._expand_query(
            self._encode(original_distances, self._expanded_membership(ranking)),
            ranking,
        )
        jaccard_distances = self._jaccard_distances(neighborhood_vectors)
        weight = self.original_distance_weight
        reranked: FloatArray = (
            1.0 - weight
        ) * jaccard_distances + weight * original_distances
        np.fill_diagonal(reranked, 0.0)
        return reranked

    def _original_distances(self, X: NumericArray) -> FloatArray:
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        match self.metric:
            case KReciprocalMetric.PRECOMPUTED:
                distances = self._validated_precomputed(X)
            case _:
                distances = cdist(X, X, metric=self.metric.value)
        if not np.all(np.isfinite(distances)):
            raise ValueError(
                "original distances must be finite; cosine distance is undefined for zero vectors"
            )
        np.fill_diagonal(distances, 0.0)
        max_distance: float = np.max(distances, initial=0.0).item()
        if max_distance > 0.0:
            distances /= max_distance
        return distances

    @staticmethod
    def _validated_precomputed(X: NumericArray) -> FloatArray:
        if X.shape[0] != X.shape[1]:
            raise ValueError("X must be a square precomputed distance matrix")
        if np.any(X < 0):
            raise ValueError("X must contain only non-negative distances")
        if not np.allclose(X, X.T):
            raise ValueError("X must be a symmetric distance matrix")
        symmetric_distances: FloatArray = (X + X.T) / 2.0
        return symmetric_distances

    @staticmethod
    def _ranking(original_distances: FloatArray) -> IntegerArray:
        ranking_distances = original_distances.copy()
        np.fill_diagonal(ranking_distances, -1.0)
        ranking: IntegerArray = np.argsort(ranking_distances, axis=1, kind="stable")
        return ranking

    def _expanded_membership(self, ranking: IntegerArray) -> BoolArray:
        reciprocal = KReciprocalNeighbors.from_ranking(
            ranking, self.n_reciprocal_neighbors
        )
        half_reciprocal = KReciprocalNeighbors.from_ranking(
            ranking, round(self.n_reciprocal_neighbors / 2)
        )
        is_member = reciprocal.membership()
        candidates = reciprocal.indices
        candidate_sets = half_reciprocal.indices[candidates]
        is_candidate_set_member = half_reciprocal.is_reciprocal[candidates]
        owners: IntegerArray = np.broadcast_to(
            np.arange(len(ranking))[:, np.newaxis, np.newaxis], candidate_sets.shape
        )
        overlaps: IntegerArray = np.sum(
            is_member[owners, candidate_sets] & is_candidate_set_member,
            axis=2,
            dtype=np.int64,
        )
        is_accepted: BoolArray = reciprocal.is_reciprocal & (
            3 * overlaps > 2 * half_reciprocal.sizes()[candidates]
        )
        is_added = is_accepted[:, :, np.newaxis] & is_candidate_set_member
        is_expanded = is_member.copy()
        is_expanded[owners[is_added], candidate_sets[is_added]] = True
        return is_expanded

    @staticmethod
    def _encode(original_distances: FloatArray, is_expanded: BoolArray) -> FloatArray:
        weights: FloatArray = np.where(is_expanded, np.exp(-original_distances), 0.0)
        normalized: FloatArray = weights / weights.sum(axis=1, keepdims=True)
        return normalized

    def _expand_query(
        self, neighborhood_vectors: FloatArray, ranking: IntegerArray
    ) -> FloatArray:
        expanded: FloatArray = np.zeros_like(neighborhood_vectors)
        for rank in range(self.n_expansion_neighbors):
            expanded += neighborhood_vectors[ranking[:, rank]]
        expanded /= self.n_expansion_neighbors
        return expanded

    @staticmethod
    def _jaccard_distances(neighborhood_vectors: FloatArray) -> FloatArray:
        n_samples = len(neighborhood_vectors)
        intersections: FloatArray = np.empty((n_samples, n_samples))
        for point in range(n_samples):
            point_row: FloatArray = neighborhood_vectors[point : point + 1]
            support: IntegerArray = np.flatnonzero(point_row)
            intersections[point] = np.minimum(
                point_row[:, support], neighborhood_vectors[:, support]
            ).sum(axis=1)
        symmetric_intersections: FloatArray = (intersections + intersections.T) / 2.0
        jaccard: FloatArray = np.maximum(
            1.0 - symmetric_intersections / (2.0 - symmetric_intersections), 0.0
        )
        return jaccard
