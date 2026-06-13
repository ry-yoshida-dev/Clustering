from __future__ import annotations

import unittest

import numpy as np

from clustering import GMMParameters, GMMCovarianceType

from tests.helpers import two_blobs


class TestGMMProcessor(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            GMMParameters(n_components=0)

    def test_fit_predict(self) -> None:
        x = two_blobs()
        proc = GMMParameters(n_components=2, random_state=42).build_processor()
        out = proc.fit_predict(x)
        self.assertEqual(out.labels.shape[0], x.shape[0])
        self.assertEqual(len(np.unique(out.labels)), 2)

    def test_predict_after_fit_predict(self) -> None:
        x = two_blobs()
        proc = GMMParameters(n_components=2, random_state=42).build_processor()
        proc.fit_predict(x)
        predicted = proc.predict(x[:5])
        self.assertEqual(predicted.labels.shape[0], 5)

    def test_fit_then_predict_sets_labels(self) -> None:
        x = two_blobs()
        proc = GMMParameters(n_components=2, random_state=42).build_processor()
        proc.fit(x)
        with self.assertRaises(ValueError):
            _ = proc.labels
        proc.predict(x)
        self.assertEqual(proc.labels.labels.shape[0], x.shape[0])

    def test_labels_before_fit_raises(self) -> None:
        proc = GMMParameters(n_components=2).build_processor()
        with self.assertRaises(ValueError):
            _ = proc.labels

    def test_build_with_covariance_type(self) -> None:
        proc = GMMParameters(
            n_components=2,
            covariance_type=GMMCovarianceType.DIAG,
        ).build_processor()
        out = proc.fit_predict(two_blobs())
        self.assertEqual(len(np.unique(out.labels)), 2)

    def test_is_precomputed_input_required_is_false(self) -> None:
        proc = GMMParameters(n_components=2).build_processor()
        self.assertFalse(proc.is_precomputed_input_required)


if __name__ == "__main__":
    unittest.main()
