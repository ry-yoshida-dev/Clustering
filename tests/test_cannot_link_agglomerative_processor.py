from __future__ import annotations

import unittest
from typing import cast

import numpy as np
from scipy.spatial.distance import cdist

from clustering import AgglomerativeLinkage, CannotLinkAgglomerativeClusteringParameters
from clustering.cannot_link.types import NegativeMatrix


def _two_close_pairs() -> np.ndarray:
    points = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]])
    return cdist(points, points)


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

    def test_is_precomputed_input_required(self) -> None:
        proc = CannotLinkAgglomerativeClusteringParameters(
            distance_threshold=1.0
        ).build_processor()
        self.assertTrue(proc.is_precomputed_input_required)


if __name__ == "__main__":
    _ = unittest.main()
