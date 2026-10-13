from __future__ import annotations

import dataclasses
import unittest

import numpy as np
from scipy.spatial.distance import cdist

from clustering import (
    AgglomerativeClusteringParameters,
    AgglomerativeLinkage,
    AgglomerativeMetric,
)
from tests.helpers import two_blobs


class TestAgglomerativeProcessor(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = AgglomerativeClusteringParameters(n_clusters=0, distance_threshold=None)
        with self.assertRaises(ValueError):
            _ = AgglomerativeClusteringParameters(
                n_clusters=None,
                distance_threshold=0,
            )
        with self.assertRaises(ValueError):
            _ = AgglomerativeClusteringParameters(
                n_clusters=None, distance_threshold=None
            )
        with self.assertRaises(ValueError):
            _ = AgglomerativeClusteringParameters(n_clusters=2, distance_threshold=0.5)
        with self.assertRaises(ValueError):
            _ = AgglomerativeClusteringParameters(
                n_clusters=2,
                distance_threshold=None,
                metric=AgglomerativeMetric.COSINE,
                linkage=AgglomerativeLinkage.WARD,
            )

    def test_single_stopping_criterion_needs_no_explicit_none(self) -> None:
        by_count = AgglomerativeClusteringParameters(n_clusters=2)
        by_distance = AgglomerativeClusteringParameters(distance_threshold=0.5)
        self.assertIsNone(by_count.distance_threshold)
        self.assertIsNone(by_distance.n_clusters)
        with self.assertRaises(ValueError):
            _ = AgglomerativeClusteringParameters()

    def test_parameters_are_frozen(self) -> None:
        parameters = AgglomerativeClusteringParameters(n_clusters=2)
        field_name = "n_clusters"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            setattr(parameters, field_name, -1)

    def test_fit_predict_with_n_clusters(self) -> None:
        x = two_blobs()
        proc = AgglomerativeClusteringParameters(
            n_clusters=2,
            distance_threshold=None,
            metric=AgglomerativeMetric.EUCLIDEAN,
            linkage=AgglomerativeLinkage.COMPLETE,
        ).build_processor()
        out = proc.fit_predict(x)
        self.assertEqual(out.labels.shape[0], x.shape[0])
        self.assertEqual(len(np.unique(out.labels)), 2)

    def test_fit_predict_with_precomputed_metric(self) -> None:
        x = two_blobs()
        distance_matrix = cdist(x, x, metric="euclidean")
        proc = AgglomerativeClusteringParameters(
            n_clusters=2,
            distance_threshold=None,
            metric=AgglomerativeMetric.PRECOMPUTED,
            linkage=AgglomerativeLinkage.AVERAGE,
        ).build_processor()
        self.assertTrue(proc.is_precomputed_input_required)
        out = proc.fit_predict(distance_matrix)
        self.assertEqual(len(np.unique(out.labels)), 2)

    def test_predict_is_not_offered(self) -> None:
        proc = AgglomerativeClusteringParameters(
            n_clusters=2,
            distance_threshold=None,
            metric=AgglomerativeMetric.EUCLIDEAN,
            linkage=AgglomerativeLinkage.COMPLETE,
        ).build_processor()
        self.assertFalse(hasattr(proc, "predict"))


if __name__ == "__main__":
    _ = unittest.main()
