from __future__ import annotations

import unittest

import numpy as np

from clustering.methods import DBSCANMetric, DBSCANParameters

from tests.helpers import two_blobs


class TestDBSCANProcessor(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            DBSCANParameters(eps=0)
        with self.assertRaises(ValueError):
            DBSCANParameters(min_samples=0)
        with self.assertRaises(ValueError):
            DBSCANParameters(leaf_size=0)

    def test_fit_predict(self) -> None:
        x = two_blobs()
        proc = DBSCANParameters(eps=1.5, min_samples=3).build_processor()
        out = proc.fit_predict(x)
        self.assertEqual(out.labels.shape[0], x.shape[0])
        self.assertGreaterEqual(out.num_clusters(is_noise_allowed=False), 2)

    def test_fit_then_labels(self) -> None:
        x = two_blobs()
        proc = DBSCANParameters(eps=1.5, min_samples=3).build_processor()
        proc.fit(x)
        lab = proc.labels
        self.assertEqual(lab.labels.shape[0], x.shape[0])

    def test_predict_not_supported(self) -> None:
        x = two_blobs()
        proc = DBSCANParameters(eps=1.5, min_samples=3).build_processor()
        proc.fit_predict(x)
        with self.assertRaises(ValueError):
            proc.predict(x)

    def test_labels_before_fit_raises(self) -> None:
        proc = DBSCANParameters().build_processor()
        with self.assertRaises(AttributeError):
            _ = proc.labels

    def test_is_precomputed_input_required(self) -> None:
        euclidean = DBSCANParameters(metric=DBSCANMetric.EUCLIDEAN).build_processor()
        precomputed = DBSCANParameters(metric=DBSCANMetric.PRECOMPUTED).build_processor()
        self.assertFalse(euclidean.is_precomputed_input_required)
        self.assertTrue(precomputed.is_precomputed_input_required)


if __name__ == "__main__":
    unittest.main()
