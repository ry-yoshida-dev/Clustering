from __future__ import annotations

import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from clustering import GMMCovarianceType, GMMParameters
from tests.helpers import two_blobs


class TestGMMProcessor(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = GMMParameters(n_components=0)

    def test_fit_predict(self) -> None:
        x = two_blobs()
        proc = GMMParameters(n_components=2, random_state=42).build_processor()
        out = proc.fit_predict(x)
        self.assertEqual(out.labels.shape[0], x.shape[0])
        self.assertEqual(len(np.unique(out.labels)), 2)

    def test_predict_after_fit_predict(self) -> None:
        x = two_blobs()
        proc = GMMParameters(n_components=2, random_state=42).build_processor()
        _ = proc.fit_predict(x)
        predicted = proc.predict(x[:5])
        self.assertEqual(predicted.labels.shape[0], 5)

    def test_fit_then_labels(self) -> None:
        x = two_blobs()
        proc = GMMParameters(n_components=2, random_state=42).build_processor()
        proc.fit(x)
        self.assertEqual(proc.labels.labels.shape[0], x.shape[0])

    def test_predict_does_not_overwrite_training_labels(self) -> None:
        x = two_blobs()
        proc = GMMParameters(n_components=2, random_state=42).build_processor()
        training_labels = proc.fit_predict(x)
        _ = proc.predict(x[:5])
        self.assertEqual(proc.labels, training_labels)

    def test_labels_before_fit_raises(self) -> None:
        proc = GMMParameters(n_components=2).build_processor()
        with self.assertRaises(NotFittedError):
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
    _ = unittest.main()
