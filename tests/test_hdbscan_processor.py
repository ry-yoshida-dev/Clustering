from __future__ import annotations

import unittest

from clustering.standard.methods import HDBSCANParameters
from tests.helpers import two_blobs


class TestHDBSCANProcessor(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = HDBSCANParameters(min_cluster_size=0)
        with self.assertRaises(ValueError):
            _ = HDBSCANParameters(min_samples=0)
        with self.assertRaises(ValueError):
            _ = HDBSCANParameters(leaf_size=0)
        with self.assertRaises(ValueError):
            _ = HDBSCANParameters(cluster_selection_epsilon=-1)
        with self.assertRaises(ValueError):
            _ = HDBSCANParameters(max_cluster_size=0)
        with self.assertRaises(ValueError):
            _ = HDBSCANParameters(alpha=0)

    def test_fit_predict(self) -> None:
        x = two_blobs()
        proc = HDBSCANParameters(min_cluster_size=5, min_samples=3).build_processor()
        out = proc.fit_predict(x)
        self.assertEqual(out.labels.shape[0], x.shape[0])
        self.assertGreaterEqual(out.num_clusters(is_noise_allowed=False), 1)

    def test_fit_then_labels(self) -> None:
        x = two_blobs()
        proc = HDBSCANParameters(min_cluster_size=5, min_samples=3).build_processor()
        proc.fit(x)
        self.assertEqual(proc.labels.labels.shape[0], x.shape[0])

    def test_predict_not_supported(self) -> None:
        x = two_blobs()
        proc = HDBSCANParameters(min_cluster_size=5).build_processor()
        _ = proc.fit_predict(x)
        with self.assertRaises(ValueError):
            _ = proc.predict(x)


if __name__ == "__main__":
    _ = unittest.main()
