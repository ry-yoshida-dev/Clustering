from __future__ import annotations

import unittest

import numpy as np

from clustering import (
    ClusteringLabels,
    GMMParameters,
    KMeansParameters,
)
from clustering.methods import DBSCANParameters


def _two_blobs(*, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((24, 2))
    b = rng.standard_normal((24, 2)) + np.array([6.0, 6.0])
    return np.vstack([a, b])


class TestClusteringLabels(unittest.TestCase):
    def test_empty_labels_raises(self) -> None:
        with self.assertRaises(ValueError):
            ClusteringLabels([])

    def test_equality_and_copy(self) -> None:
        a = ClusteringLabels([0, 1, 0])
        b = ClusteringLabels(np.array([0, 1, 0], dtype=np.int64))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)
        self.assertEqual(a[1], 1)

    def test_noise_aware_counts(self) -> None:
        cl = ClusteringLabels([0, -1, 0, 1], noise_label=-1)
        self.assertTrue(cl.is_in_noise)
        self.assertEqual(cl.num_noise, 1)
        self.assertEqual(cl.num_not_noise_labels, 3)
        self.assertEqual(cl.num_clusters(is_noise_allowed=False), 2)
        self.assertEqual(cl.num_clusters(is_noise_allowed=True), 3)

    def test_get_indices_missing_label(self) -> None:
        cl = ClusteringLabels([0, 0, 1])
        with self.assertRaises(ValueError):
            cl.get_indices(99)


class TestKMeansProcessor(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            KMeansParameters(n_clusters=0)
        with self.assertRaises(ValueError):
            KMeansParameters(max_iter=0)

    def test_fit_predict_labels_property(self) -> None:
        X = _two_blobs()
        proc = KMeansParameters(n_clusters=2, random_state=42).build_processor()
        out = proc.fit_predict(X)
        self.assertEqual(out.labels.shape[0], X.shape[0])
        self.assertEqual(len(np.unique(out.labels)), 2)
        proc.fit(X)
        lab = proc.labels
        self.assertEqual(lab.labels.shape[0], X.shape[0])
        self.assertEqual(len(np.unique(lab.labels)), 2)


class TestGMMProcessor(unittest.TestCase):
    def test_fit_predict(self) -> None:
        X = _two_blobs()
        proc = GMMParameters(n_components=2, random_state=42).build_processor()
        out = proc.fit_predict(X)
        self.assertEqual(out.labels.shape[0], X.shape[0])
        self.assertEqual(len(np.unique(out.labels)), 2)


class TestDBSCANCommonProcessor(unittest.TestCase):
    def test_predict_not_supported(self) -> None:
        X = _two_blobs()
        proc = DBSCANParameters(eps=1.5, min_samples=3).build_processor()
        proc.fit_predict(X)
        with self.assertRaises(ValueError):
            proc.predict(X)


if __name__ == "__main__":
    unittest.main()
