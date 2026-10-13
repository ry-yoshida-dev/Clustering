from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors

from clustering import ClusteringMethod
from clustering.standard.algorithms import (
    CorePointLabeling,
    SharedNearestNeighborClustering,
    SharedNeighborGraph,
)
from clustering.standard.methods import SNNMetric, SNNParameters
from tests.helpers import two_blobs


class TestSNNProcessor(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = SNNParameters(n_neighbors=0)
        with self.assertRaises(ValueError):
            _ = SNNParameters(n_neighbors=5, min_shared_neighbors=5)
        with self.assertRaises(ValueError):
            _ = SNNParameters(n_neighbors=5, min_shared_neighbors=-1)
        with self.assertRaises(ValueError):
            _ = SNNParameters(n_neighbors=5, min_samples=0)
        with self.assertRaises(ValueError):
            _ = SNNParameters(n_neighbors=5, min_samples=6)

    def test_fit_predict_separates_two_blobs(self) -> None:
        x = two_blobs()
        proc = SNNParameters(
            n_neighbors=10, min_shared_neighbors=3, min_samples=3
        ).build_processor()
        out = proc.fit_predict(x)
        self.assertEqual(proc.method, ClusteringMethod.SNN)
        self.assertEqual(len(out.labels), len(x))
        self.assertEqual(out.num_clusters(is_noise_allowed=False), 2)
        first_blob = np.setdiff1d(out.labels[:24], [-1])
        second_blob = np.setdiff1d(out.labels[24:], [-1])
        self.assertEqual(len(first_blob), 1)
        self.assertEqual(len(second_blob), 1)
        self.assertEqual(len(np.intersect1d(first_blob, second_blob)), 0)

    def test_isolated_outlier_is_noise(self) -> None:
        x = np.vstack([two_blobs(), np.array([[100.0, -100.0]])])
        out = (
            SNNParameters(n_neighbors=10, min_shared_neighbors=3, min_samples=3)
            .build_processor()
            .fit_predict(x)
        )
        np.testing.assert_array_equal(out.labels[[-1]], [-1])

    def test_elongated_chain_forms_single_cluster(self) -> None:
        x = np.column_stack([np.arange(2000, dtype=np.float64), np.zeros(2000)])
        out = (
            SNNParameters(n_neighbors=4, min_shared_neighbors=1, min_samples=2)
            .build_processor()
            .fit_predict(x)
        )
        self.assertEqual(out.num_clusters(is_noise_allowed=False), 1)
        np.testing.assert_array_equal(out.labels, np.zeros(len(x), dtype=np.int64))

    def test_precomputed_matches_feature_input(self) -> None:
        x = two_blobs()
        from_features = SNNParameters(n_neighbors=10).build_processor().fit_predict(x)
        from_distances = (
            SNNParameters(n_neighbors=10, metric=SNNMetric.PRECOMPUTED)
            .build_processor()
            .fit_predict(cdist(x, x))
        )
        self.assertEqual(from_features, from_distances)

    def test_equidistant_neighbors_are_ordered_by_index(self) -> None:
        distances = np.array([[1.0, 1.0, 0.5], [2.0, 2.0, 2.0]])
        indices = np.array([[4, 2, 7], [9, 3, 5]], dtype=np.int64)
        np.testing.assert_array_equal(
            SharedNearestNeighborClustering.sort_neighbors(distances, indices),
            np.array([[7, 2, 4], [3, 5, 9]]),
        )

    def test_fit_then_labels(self) -> None:
        x = two_blobs()
        proc = SNNParameters(n_neighbors=10).build_processor()
        proc.fit(x)
        self.assertEqual(proc.labels, proc.fit_predict(x))

    def test_labels_before_fit_raises(self) -> None:
        proc = SNNParameters().build_processor()
        with self.assertRaises(NotFittedError):
            _ = proc.labels

    def test_predict_not_supported(self) -> None:
        x = two_blobs()
        proc = SNNParameters(n_neighbors=10).build_processor()
        _ = proc.fit_predict(x)
        with self.assertRaises(ValueError):
            _ = proc.predict(x)

    def test_too_few_samples_raises(self) -> None:
        proc = SNNParameters(n_neighbors=10).build_processor()
        with self.assertRaises(ValueError):
            _ = proc.fit_predict(two_blobs()[:10])

    def test_non_square_precomputed_raises(self) -> None:
        proc = SNNParameters(
            n_neighbors=3,
            min_shared_neighbors=1,
            min_samples=2,
            metric=SNNMetric.PRECOMPUTED,
        ).build_processor()
        with self.assertRaises(ValueError):
            _ = proc.fit_predict(two_blobs())

    def test_is_precomputed_input_required(self) -> None:
        euclidean = SNNParameters(metric=SNNMetric.EUCLIDEAN).build_processor()
        precomputed = SNNParameters(metric=SNNMetric.PRECOMPUTED).build_processor()
        self.assertFalse(euclidean.is_precomputed_input_required)
        self.assertTrue(precomputed.is_precomputed_input_required)


class TestSharedNeighborGraph(unittest.TestCase):
    def test_only_mutual_links_with_shared_counts(self) -> None:
        neighbors = np.array([[1, 2], [0, 2], [0, 1], [2, 0]])
        graph = SharedNeighborGraph.from_neighbors(neighbors)
        np.testing.assert_array_equal(graph.sources, [0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(graph.targets, [1, 2, 0, 2, 0, 1])
        np.testing.assert_array_equal(graph.shared_counts, [1, 1, 1, 1, 1, 1])

    def test_strong_links_filters_by_threshold(self) -> None:
        neighbors = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 4], [0, 1, 2], [2, 3, 0]])
        graph = SharedNeighborGraph.from_neighbors(neighbors)
        strong = graph.strong_links(2)
        self.assertTrue(np.all(strong.shared_counts >= 2))
        self.assertLess(len(strong.sources), len(graph.sources))

    def test_batched_construction_matches_single_batch(self) -> None:
        neighbors = (
            NearestNeighbors(n_neighbors=6)
            .fit(two_blobs())
            .kneighbors(return_distance=False)
        )
        single_batch = SharedNeighborGraph.from_neighbors(neighbors)
        with mock.patch.object(SharedNeighborGraph, "LOOKUP_BUDGET", 1):
            batched = SharedNeighborGraph.from_neighbors(neighbors)
        np.testing.assert_array_equal(batched.sources, single_batch.sources)
        np.testing.assert_array_equal(batched.targets, single_batch.targets)
        np.testing.assert_array_equal(
            batched.shared_counts, single_batch.shared_counts
        )


class TestCorePointLabeling(unittest.TestCase):
    @staticmethod
    def _undirected_graph(
        links: list[tuple[int, int, int]],
    ) -> SharedNeighborGraph:
        directed = links + [(target, source, count) for source, target, count in links]
        return SharedNeighborGraph(
            sources=np.array([link[0] for link in directed], dtype=np.int64),
            targets=np.array([link[1] for link in directed], dtype=np.int64),
            shared_counts=np.array([link[2] for link in directed], dtype=np.int64),
        )

    @staticmethod
    def _clique_links(points: list[int], count: int) -> list[tuple[int, int, int]]:
        return [
            (source, target, count)
            for index, source in enumerate(points)
            for target in points[index + 1 :]
        ]

    def test_border_point_joins_core_with_most_shared_neighbors(self) -> None:
        graph = self._undirected_graph(
            self._clique_links([1, 3, 4, 7], 5)
            + self._clique_links([2, 5, 6, 8], 5)
            + [(0, 1, 2), (0, 2, 5)]
        )
        labels = CorePointLabeling(
            strong_links=graph, n_samples=9, min_samples=3
        ).labels()
        np.testing.assert_array_equal(labels[[0]], labels[[2]])
        self.assertFalse(np.array_equal(labels[[1]], labels[[2]]))

    def test_tied_border_point_joins_first_listed_core(self) -> None:
        graph = self._undirected_graph(
            [(0, 2, 4), (0, 1, 4)]
            + self._clique_links([1, 3, 4, 7], 5)
            + self._clique_links([2, 5, 6, 8], 5)
        )
        labels = CorePointLabeling(
            strong_links=graph, n_samples=9, min_samples=3
        ).labels()
        np.testing.assert_array_equal(labels[[0]], labels[[2]])

    def test_point_without_core_link_is_noise(self) -> None:
        graph = self._undirected_graph(self._clique_links([0, 1, 2, 3], 5))
        labels = CorePointLabeling(
            strong_links=graph, n_samples=5, min_samples=3
        ).labels()
        np.testing.assert_array_equal(labels, np.array([0, 0, 0, 0, -1]))


if __name__ == "__main__":
    _ = unittest.main()
