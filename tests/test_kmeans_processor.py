from __future__ import annotations

import unittest

import numpy as np

from clustering import KMeansParameters

from tests.helpers import two_blobs


class TestKMeansProcessor(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            KMeansParameters(n_clusters=0)
        with self.assertRaises(ValueError):
            KMeansParameters(max_iter=0)
        with self.assertRaises(ValueError):
            KMeansParameters(tol=0)

    def test_fit_predict(self) -> None:
        x = two_blobs()
        proc = KMeansParameters(n_clusters=2, random_state=42).build_processor()
        out = proc.fit_predict(x)
        self.assertEqual(out.labels.shape[0], x.shape[0])
        self.assertEqual(len(np.unique(out.labels)), 2)

    def test_fit_then_labels(self) -> None:
        x = two_blobs()
        proc = KMeansParameters(n_clusters=2, random_state=42).build_processor()
        proc.fit(x)
        lab = proc.labels
        self.assertEqual(lab.labels.shape[0], x.shape[0])
        self.assertEqual(len(np.unique(lab.labels)), 2)

    def test_predict_after_fit(self) -> None:
        x = two_blobs()
        proc = KMeansParameters(n_clusters=2, random_state=42).build_processor()
        proc.fit(x)
        predicted = proc.predict(x[:5])
        self.assertEqual(predicted.labels.shape[0], 5)

    def test_labels_before_fit_raises(self) -> None:
        proc = KMeansParameters(n_clusters=2).build_processor()
        with self.assertRaises(AttributeError):
            _ = proc.labels

    def test_is_precomputed_input_required_is_false(self) -> None:
        proc = KMeansParameters(n_clusters=2).build_processor()
        self.assertFalse(proc.is_precomputed_input_required)


if __name__ == "__main__":
    unittest.main()
