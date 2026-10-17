from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.distance import cdist

from clustering import (
    DBSCANMetric,
    DBSCANParameters,
    KReciprocalMetric,
    KReciprocalReranker,
)
from clustering.reranking.k_reciprocal import KReciprocalNeighbors
from clustering.types import BoolArray, FloatArray, IntegerArray
from tests.helpers import two_blobs


class TestKReciprocalReranker(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = KReciprocalReranker(n_reciprocal_neighbors=0)
        with self.assertRaises(ValueError):
            _ = KReciprocalReranker(n_expansion_neighbors=0)
        with self.assertRaises(ValueError):
            _ = KReciprocalReranker(original_distance_weight=-0.1)
        with self.assertRaises(ValueError):
            _ = KReciprocalReranker(original_distance_weight=1.1)

    def test_output_is_symmetric_bounded_with_zero_diagonal(self) -> None:
        reranked = KReciprocalReranker(n_reciprocal_neighbors=10).rerank(two_blobs())
        self.assertEqual(reranked.shape, (48, 48))
        np.testing.assert_array_equal(reranked, reranked.T)
        np.testing.assert_array_equal(np.diagonal(reranked), np.zeros(48))
        self.assertTrue(np.all(reranked >= 0.0))
        self.assertTrue(np.all(reranked <= 1.0))

    def test_full_original_weight_returns_scaled_original_distances(self) -> None:
        x = two_blobs()
        reranked = KReciprocalReranker(
            n_reciprocal_neighbors=10, original_distance_weight=1.0
        ).rerank(x)
        original = cdist(x, x)
        np.testing.assert_allclose(reranked, original / original.max())

    def test_pure_jaccard_saturates_between_blobs(self) -> None:
        reranked = KReciprocalReranker(
            n_reciprocal_neighbors=10, original_distance_weight=0.0
        ).rerank(two_blobs())
        np.testing.assert_allclose(reranked[:24, 24:], 1.0)
        self.assertLess(float(reranked[:24, :24].mean()), 1.0)

    def test_precomputed_matches_feature_input(self) -> None:
        x = two_blobs()
        from_features = KReciprocalReranker(n_reciprocal_neighbors=10).rerank(x)
        from_distances = KReciprocalReranker(
            n_reciprocal_neighbors=10, metric=KReciprocalMetric.PRECOMPUTED
        ).rerank(cdist(x, x))
        np.testing.assert_allclose(from_features, from_distances)

    def test_reranked_distances_feed_dbscan(self) -> None:
        reranked = KReciprocalReranker(
            n_reciprocal_neighbors=10, original_distance_weight=0.0
        ).rerank(two_blobs())
        out = (
            DBSCANParameters(eps=0.9, min_samples=3, metric=DBSCANMetric.PRECOMPUTED)
            .build_processor()
            .fit_predict(reranked)
        )
        self.assertEqual(out.num_clusters(is_noise_allowed=False), 2)
        self.assertEqual(len(np.unique(out.labels[:24])), 1)
        self.assertEqual(len(np.unique(out.labels[24:])), 1)

    def test_duplicate_points_rank_themselves_first(self) -> None:
        x = np.vstack([np.zeros((4, 2)), np.ones((4, 2))])
        reranked = KReciprocalReranker(
            n_reciprocal_neighbors=3, n_expansion_neighbors=2
        ).rerank(x)
        np.testing.assert_array_equal(np.diagonal(reranked), np.zeros(8))
        np.testing.assert_allclose(reranked[:4, :4], 0.0, atol=1e-12)

    def test_too_few_samples_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = KReciprocalReranker(n_reciprocal_neighbors=48).rerank(two_blobs())

    def test_invalid_precomputed_raises(self) -> None:
        reranker = KReciprocalReranker(
            n_reciprocal_neighbors=2, metric=KReciprocalMetric.PRECOMPUTED
        )
        with self.assertRaises(ValueError):
            _ = reranker.rerank(np.zeros((4, 3)))
        with self.assertRaises(ValueError):
            _ = reranker.rerank(-np.ones((4, 4)))
        asymmetric = np.arange(16, dtype=np.float64).reshape(4, 4)
        with self.assertRaises(ValueError):
            _ = reranker.rerank(asymmetric)

    def test_matches_loop_port_of_reference_implementation(self) -> None:
        rng = np.random.default_rng(0)
        for trial in range(12):
            n_samples = int(rng.integers(15, 60))
            features: FloatArray = rng.normal(size=(n_samples, 3))
            if trial % 3 == 0:
                features = np.vstack([features, features[:5]])
            n_reciprocal_neighbors = int(rng.integers(1, 20))
            n_expansion_neighbors = int(rng.integers(1, 8))
            original_distance_weight = float(rng.random())
            for metric in (
                KReciprocalMetric.EUCLIDEAN,
                KReciprocalMetric.SQEUCLIDEAN,
                KReciprocalMetric.COSINE,
            ):
                reranked = KReciprocalReranker(
                    n_reciprocal_neighbors=n_reciprocal_neighbors,
                    n_expansion_neighbors=n_expansion_neighbors,
                    original_distance_weight=original_distance_weight,
                    metric=metric,
                ).rerank(features)
                expected = self._reference_rerank(
                    features,
                    n_reciprocal_neighbors,
                    n_expansion_neighbors,
                    original_distance_weight,
                    metric,
                )
                np.testing.assert_allclose(reranked, expected, atol=1e-12)

    @staticmethod
    def _reference_rerank(
        features: FloatArray,
        n_reciprocal_neighbors: int,
        n_expansion_neighbors: int,
        original_distance_weight: float,
        metric: KReciprocalMetric,
    ) -> FloatArray:
        distances: FloatArray = cdist(features, features, metric=metric.value)
        np.fill_diagonal(distances, 0.0)
        distances /= distances.max()
        ranking_distances = distances.copy()
        np.fill_diagonal(ranking_distances, -1.0)
        ranking: IntegerArray = np.argsort(ranking_distances, axis=1, kind="stable")
        n_samples = len(features)

        def reciprocal_set(point: int, n_neighbors: int) -> IntegerArray:
            forward: IntegerArray = ranking[point, : n_neighbors + 1]
            backward: IntegerArray = ranking[forward, : n_neighbors + 1]
            is_member: BoolArray = np.any(np.equal(backward, point), axis=1)
            members: IntegerArray = forward[is_member]
            return members

        n_half_neighbors: int = round(n_reciprocal_neighbors / 2)
        vectors: FloatArray = np.zeros((n_samples, n_samples))
        for point in range(n_samples):
            reciprocal = reciprocal_set(point, n_reciprocal_neighbors)
            expansion: IntegerArray = reciprocal
            for index in range(len(reciprocal)):
                candidate: int = reciprocal.item(index)
                candidate_set = reciprocal_set(candidate, n_half_neighbors)
                overlap = len(np.intersect1d(candidate_set, reciprocal))
                if overlap > 2.0 / 3.0 * len(candidate_set):
                    expansion = np.append(expansion, candidate_set)
            expansion = np.unique(expansion)
            expansion_distances: FloatArray = distances[
                np.full_like(expansion, point), expansion
            ]
            weights: FloatArray = np.exp(-expansion_distances)
            vectors[point, expansion] = weights / weights.sum()
        vectors = np.array(
            [
                vectors[ranking[point, :n_expansion_neighbors]].mean(axis=0)
                for point in range(n_samples)
            ]
        )
        jaccard: FloatArray = np.zeros((n_samples, n_samples))
        for point in range(n_samples):
            point_vector: FloatArray = vectors[point : point + 1]
            intersections: FloatArray = np.minimum(point_vector, vectors).sum(axis=1)
            jaccard[point] = 1.0 - intersections / (2.0 - intersections)
        expected: FloatArray = (
            1.0 - original_distance_weight
        ) * jaccard + original_distance_weight * distances
        np.fill_diagonal(expected, 0.0)
        return expected

    def test_non_finite_input_raises(self) -> None:
        x = two_blobs()
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            _ = KReciprocalReranker(n_reciprocal_neighbors=10).rerank(x)

    def test_cosine_with_zero_vector_raises(self) -> None:
        x = two_blobs()
        x[0] = 0.0
        with self.assertRaises(ValueError):
            _ = KReciprocalReranker(
                n_reciprocal_neighbors=10, metric=KReciprocalMetric.COSINE
            ).rerank(x)


class TestKReciprocalNeighbors(unittest.TestCase):
    def test_one_sided_neighbor_is_not_reciprocal(self) -> None:
        x = np.array([[0.0], [1.0], [1.5], [10.0]])
        distances = cdist(x, x)
        np.fill_diagonal(distances, -1.0)
        ranking = np.argsort(distances, axis=1, kind="stable")
        neighbors = KReciprocalNeighbors.from_ranking(ranking, 1)
        is_member = neighbors.membership()
        self.assertTrue(is_member[1, 2])
        self.assertTrue(is_member[2, 1])
        self.assertFalse(is_member[0, 1])
        self.assertFalse(is_member[3, 2])
        np.testing.assert_array_equal(np.diagonal(is_member), np.ones(4, dtype=bool))
        np.testing.assert_array_equal(neighbors.sizes(), [1, 2, 2, 1])

    def test_invalid_neighborhood_size_raises(self) -> None:
        ranking = np.tile(np.arange(3), (3, 1))
        with self.assertRaises(ValueError):
            _ = KReciprocalNeighbors.from_ranking(ranking, 3)
