from __future__ import annotations

import unittest
from typing import cast

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.exceptions import NotFittedError

from clustering import AgglomerativeLinkage, CannotLinkAgglomerativeClusteringParameters
from clustering.cannot_link.types import NegativeMatrix
from clustering.types import IntegerArray, NumericArray


def _two_close_pairs() -> np.ndarray:
    points = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]])
    return cdist(points, points)


def _greedy_reference_labels(
    distance_matrix: NumericArray,
    negative_matrix: NegativeMatrix,
    linkage: AgglomerativeLinkage,
    distance_threshold: float,
) -> IntegerArray:
    clusters = [[point] for point in range(len(distance_matrix))]

    def cluster_distance(first: list[int], second: list[int]) -> float:
        if negative_matrix[np.ix_(first, second)].any():
            return float("inf")
        block = distance_matrix[np.ix_(first, second)]
        match linkage:
            case AgglomerativeLinkage.SINGLE:
                return float(block.min())
            case AgglomerativeLinkage.COMPLETE:
                return float(block.max())
            case AgglomerativeLinkage.AVERAGE:
                return float(np.mean(block, dtype=np.float64))
            case AgglomerativeLinkage.WARD:
                raise ValueError("ward linkage is not supported")

    while len(clusters) > 1:
        best_distance, first_index, second_index = min(
            (cluster_distance(clusters[i], clusters[j]), i, j)
            for i in range(len(clusters))
            for j in range(i + 1, len(clusters))
        )
        if best_distance > distance_threshold:
            break
        clusters[first_index].extend(clusters.pop(second_index))

    labels = np.empty(len(distance_matrix), dtype=np.int64)
    for label, members in enumerate(clusters):
        labels[members] = label
    return labels


def _is_same_partition(first: IntegerArray, second: IntegerArray) -> bool:
    first_labels = cast(list[int], first.tolist())
    second_labels = cast(list[int], second.tolist())
    pairs = set(zip(first_labels, second_labels, strict=True))
    return len(pairs) == len(set(first_labels)) == len(set(second_labels))


class TestCannotLinkAgglomerativeProcessor(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = CannotLinkAgglomerativeClusteringParameters(distance_threshold=0)
        with self.assertRaises(ValueError):
            _ = CannotLinkAgglomerativeClusteringParameters(distance_threshold=-1.0)
        with self.assertRaises(ValueError):
            _ = CannotLinkAgglomerativeClusteringParameters(
                distance_threshold=0.5, linkage=AgglomerativeLinkage.WARD
            )

    def test_constraint_separates_close_pair_for_every_supported_linkage(self) -> None:
        distance_matrix = _two_close_pairs()
        negative_matrix = np.zeros((4, 4), dtype=bool)
        negative_matrix[0, 1] = negative_matrix[1, 0] = True

        supported_linkages = [
            AgglomerativeLinkage.SINGLE,
            AgglomerativeLinkage.COMPLETE,
            AgglomerativeLinkage.AVERAGE,
        ]
        for linkage in supported_linkages:
            with self.subTest(linkage=linkage):
                proc = CannotLinkAgglomerativeClusteringParameters(
                    distance_threshold=1.0,
                    linkage=linkage,
                ).build_processor()
                out = proc.fit_predict(distance_matrix, negative_matrix)
                self.assertNotEqual(out.labels[0], out.labels[1])
                self.assertEqual(out.labels[2], out.labels[3])
                self.assertEqual(out.num_clusters(), 3)

    def test_without_constraint_close_pair_merges(self) -> None:
        distance_matrix = _two_close_pairs()
        negative_matrix = np.zeros((4, 4), dtype=bool)
        proc = CannotLinkAgglomerativeClusteringParameters(
            distance_threshold=1.0,
            linkage=AgglomerativeLinkage.COMPLETE,
        ).build_processor()
        out = proc.fit_predict(distance_matrix, negative_matrix)
        self.assertEqual(out.labels[0], out.labels[1])
        self.assertEqual(out.labels[2], out.labels[3])
        self.assertEqual(out.num_clusters(), 2)

    def test_negative_matrix_shape_mismatch_raises(self) -> None:
        distance_matrix = _two_close_pairs()
        negative_matrix = np.zeros((3, 3), dtype=bool)
        proc = CannotLinkAgglomerativeClusteringParameters(
            distance_threshold=1.0
        ).build_processor()
        with self.assertRaises(ValueError):
            _ = proc.fit_predict(distance_matrix, negative_matrix)

    def test_negative_matrix_dtype_mismatch_raises(self) -> None:
        distance_matrix = _two_close_pairs()
        negative_matrix = cast(NegativeMatrix, np.zeros((4, 4), dtype=np.int64))
        proc = CannotLinkAgglomerativeClusteringParameters(
            distance_threshold=1.0
        ).build_processor()
        with self.assertRaises(TypeError):
            _ = proc.fit_predict(distance_matrix, negative_matrix)

    def test_negative_matrix_asymmetric_raises(self) -> None:
        distance_matrix = _two_close_pairs()
        negative_matrix = np.zeros((4, 4), dtype=bool)
        negative_matrix[0, 1] = True
        proc = CannotLinkAgglomerativeClusteringParameters(
            distance_threshold=1.0
        ).build_processor()
        with self.assertRaises(ValueError):
            _ = proc.fit_predict(distance_matrix, negative_matrix)

    def test_asymmetric_distance_matrix_raises(self) -> None:
        distance_matrix = np.array([[0.0, 1.0, 5.0], [5.0, 0.0, 1.0], [1.0, 5.0, 0.0]])
        negative_matrix = np.zeros((3, 3), dtype=bool)
        proc = CannotLinkAgglomerativeClusteringParameters(
            distance_threshold=0.5
        ).build_processor()
        with self.assertRaises(ValueError):
            _ = proc.fit_predict(distance_matrix, negative_matrix)

    def test_negative_distance_raises(self) -> None:
        distance_matrix = _two_close_pairs()
        distance_matrix[0, 1] = distance_matrix[1, 0] = -0.1
        negative_matrix = np.zeros((4, 4), dtype=bool)
        proc = CannotLinkAgglomerativeClusteringParameters(
            distance_threshold=1.0
        ).build_processor()
        with self.assertRaises(ValueError):
            _ = proc.fit_predict(distance_matrix, negative_matrix)

    def test_distance_matrix_within_tolerance_of_symmetric_is_accepted(self) -> None:
        distance_matrix = _two_close_pairs()
        distance_matrix[0, 1] += 1e-12
        negative_matrix = np.zeros((4, 4), dtype=bool)
        proc = CannotLinkAgglomerativeClusteringParameters(
            distance_threshold=1.0
        ).build_processor()
        out = proc.fit_predict(distance_matrix, negative_matrix)
        self.assertEqual(out.num_clusters(), 2)

    def test_negative_matrix_diagonal_raises(self) -> None:
        distance_matrix = _two_close_pairs()
        negative_matrix = np.zeros((4, 4), dtype=bool)
        negative_matrix[0, 0] = True
        proc = CannotLinkAgglomerativeClusteringParameters(
            distance_threshold=1.0
        ).build_processor()
        with self.assertRaises(ValueError):
            _ = proc.fit_predict(distance_matrix, negative_matrix)

    def test_predict_is_not_offered(self) -> None:
        proc = CannotLinkAgglomerativeClusteringParameters(
            distance_threshold=1.0
        ).build_processor()
        self.assertFalse(hasattr(proc, "predict"))

    def test_matches_greedy_reference_on_random_inputs(self) -> None:
        rng = np.random.default_rng(810)
        supported_linkages = [
            AgglomerativeLinkage.SINGLE,
            AgglomerativeLinkage.COMPLETE,
            AgglomerativeLinkage.AVERAGE,
        ]
        for trial in range(100):
            n_points = int(rng.integers(3, 12))
            points = rng.random((n_points, 2))
            distance_matrix = cdist(points, points)
            upper = np.triu(rng.random((n_points, n_points)) < 0.15, k=1)
            negative_matrix = upper | upper.T
            distance_threshold = float(rng.uniform(0.1, 0.8))
            for linkage in supported_linkages:
                with self.subTest(trial=trial, linkage=linkage):
                    proc = CannotLinkAgglomerativeClusteringParameters(
                        distance_threshold=distance_threshold, linkage=linkage
                    ).build_processor()
                    actual = proc.fit_predict(distance_matrix, negative_matrix)
                    expected = _greedy_reference_labels(
                        distance_matrix, negative_matrix, linkage, distance_threshold
                    )
                    self.assertTrue(_is_same_partition(actual.labels, expected))

    def test_tied_distances_close_the_chain(self) -> None:
        distance_matrix = np.array(
            [
                [0.0, 2.0, 3.0, 2.0, 3.0],
                [2.0, 0.0, 3.0, 2.0, 1.0],
                [3.0, 3.0, 0.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 0.0, 3.0],
                [3.0, 1.0, 2.0, 3.0, 0.0],
            ]
        )
        negative_matrix = np.zeros((5, 5), dtype=bool)
        negative_matrix[0, [1, 3]] = True
        negative_matrix[[1, 3], 0] = True
        proc = CannotLinkAgglomerativeClusteringParameters(
            distance_threshold=2.5, linkage=AgglomerativeLinkage.SINGLE
        ).build_processor()
        actual = proc.fit_predict(distance_matrix, negative_matrix)
        expected = np.array([0, 1, 1, 1, 1], dtype=np.int64)
        self.assertTrue(_is_same_partition(actual.labels, expected))

    def test_respects_constraints_on_heavily_tied_inputs(self) -> None:
        rng = np.random.default_rng(114514)
        supported_linkages = [
            AgglomerativeLinkage.SINGLE,
            AgglomerativeLinkage.COMPLETE,
            AgglomerativeLinkage.AVERAGE,
        ]
        for trial in range(300):
            n_points = int(rng.integers(4, 10))
            upper_distances = np.triu(
                rng.integers(1, 4, size=(n_points, n_points)), k=1
            ).astype(np.float64)
            distance_matrix = upper_distances + upper_distances.T
            upper = np.triu(rng.random((n_points, n_points)) < 0.3, k=1)
            negative_matrix = upper | upper.T
            for linkage in supported_linkages:
                with self.subTest(trial=trial, linkage=linkage):
                    proc = CannotLinkAgglomerativeClusteringParameters(
                        distance_threshold=2.5, linkage=linkage
                    ).build_processor()
                    labels = proc.fit_predict(distance_matrix, negative_matrix).labels
                    same_cluster = np.equal.outer(labels, labels)
                    self.assertFalse(np.any(same_cluster & negative_matrix))

    def test_labels_cached_after_fit(self) -> None:
        proc = CannotLinkAgglomerativeClusteringParameters(
            distance_threshold=1.0
        ).build_processor()
        with self.assertRaises(NotFittedError):
            _ = proc.labels
        negative_matrix = np.zeros((4, 4), dtype=bool)
        proc.fit(_two_close_pairs(), negative_matrix)
        self.assertEqual(proc.labels.num_clusters(), 2)

    def test_is_precomputed_input_required(self) -> None:
        proc = CannotLinkAgglomerativeClusteringParameters(
            distance_threshold=1.0
        ).build_processor()
        self.assertTrue(proc.is_precomputed_input_required)


if __name__ == "__main__":
    _ = unittest.main()
