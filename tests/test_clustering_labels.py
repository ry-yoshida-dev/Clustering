from __future__ import annotations

import unittest

import numpy as np

from clustering import ClusteringLabels


class TestClusteringLabels(unittest.TestCase):
    def test_empty_labels_raises(self) -> None:
        with self.assertRaises(ValueError):
            ClusteringLabels([])

    def test_equality_and_copy_isolation(self) -> None:
        source = np.array([0, 1, 0], dtype=np.int64)
        a = ClusteringLabels(source)
        b = ClusteringLabels(np.array([0, 1, 0], dtype=np.int64))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)
        self.assertEqual(a[1], 1)
        source[0] = 99
        self.assertEqual(a[0], 0)

    def test_noise_aware_counts(self) -> None:
        cl = ClusteringLabels([0, -1, 0, 1], noise_label=-1)
        self.assertTrue(cl.is_in_noise)
        self.assertEqual(cl.num_noise, 1)
        self.assertEqual(cl.num_not_noise_labels, 3)
        self.assertEqual(cl.num_clusters(is_noise_allowed=False), 2)
        self.assertEqual(cl.num_clusters(is_noise_allowed=True), 3)

    def test_set_labels(self) -> None:
        cl = ClusteringLabels([2, 0, 2, 1])
        self.assertEqual(cl.set_labels, [0, 1, 2])

    def test_get_indices(self) -> None:
        cl = ClusteringLabels([0, 0, 1])
        self.assertEqual(cl.get_indices(0), [0, 1])
        self.assertEqual(cl.get_indices(99, is_empty_allowed=True), [])

    def test_get_indices_missing_label_raises(self) -> None:
        cl = ClusteringLabels([0, 0, 1])
        with self.assertRaises(ValueError):
            cl.get_indices(99)

    def test_get_labels(self) -> None:
        cl = ClusteringLabels([0, 1, 2])
        self.assertEqual(cl.get_labels([0, 2]), [0, 2])

    def test_get_labels_out_of_range_raises(self) -> None:
        cl = ClusteringLabels([0, 1, 2])
        with self.assertRaises(ValueError):
            cl.get_labels([0, 3])

    def test_get_mask(self) -> None:
        cl = ClusteringLabels([0, -1, 1], noise_label=-1)
        mask = cl.get_mask(-1)
        self.assertEqual(mask.tolist(), [False, True, False])

    def test_get_cluster_counts(self) -> None:
        cl = ClusteringLabels([0, -1, 0, 1], noise_label=-1)
        with_noise = cl.get_cluster_counts(include_noise=True)
        without_noise = cl.get_cluster_counts(include_noise=False)
        self.assertEqual(with_noise, {0: 2, 1: 1, -1: 1})
        self.assertEqual(without_noise, {0: 2, 1: 1})

    def test_expanded_labels(self) -> None:
        cl = ClusteringLabels([0, -1, 0, 1], noise_label=-1)
        expanded = cl.expanded_labels
        self.assertEqual(len(expanded), 4)
        self.assertNotIn(-1, expanded)
        self.assertEqual(expanded[1], 2)

    def test_get_major_labels(self) -> None:
        cl = ClusteringLabels([0, 0, 0, 1, 1, 2])
        self.assertEqual(cl.get_major_labels(min_count=2), [0, 1])
        self.assertEqual(cl.get_major_labels(min_count=3), [0])

    def test_iter_len_and_array_protocol(self) -> None:
        cl = ClusteringLabels([0, 1, 2])
        self.assertEqual(list(cl), [0, 1, 2])
        self.assertEqual(len(cl), 3)
        np.testing.assert_array_equal(np.asarray(cl), cl.labels)

    def test_inequality(self) -> None:
        a = ClusteringLabels([0, 1], noise_label=-1)
        b = ClusteringLabels([0, 1], noise_label=0)
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
