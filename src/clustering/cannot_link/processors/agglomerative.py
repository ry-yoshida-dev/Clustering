from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...result import ClusteringLabels
from ...types import IntegerArray, NumericArray
from ..algorithms.agglomerative import ConstrainedAgglomerativeClustering
from ..processor import CannotLinkClusteringProcessor
from ..types import NegativeMatrix


@dataclass
class CannotLinkAgglomerativeProcessor(CannotLinkClusteringProcessor):
    """
    Wraps ConstrainedAgglomerativeClustering for the cannot-link clustering pipeline.

    Attributes
    ----------
    processor : ConstrainedAgglomerativeClustering
        The underlying constrained agglomerative clustering algorithm.
    """

    processor: ConstrainedAgglomerativeClustering

    def fit(self, X: NumericArray, negative_matrix: NegativeMatrix) -> None:
        """
        Fit the cannot-link agglomerative clustering processor.

        Parameters
        ----------
        X : NumericArray
            Precomputed pairwise distance matrix.
        negative_matrix : NegativeMatrix
            Symmetric boolean mask marking cannot-link pairs.
        """
        self.processor.fit(X, negative_matrix)

    def predict(self, X: NumericArray) -> ClusteringLabels:
        """
        Predict is not supported for this processor type.

        Parameters
        ----------
        X : NumericArray
            Unused; present for the CannotLinkClusteringProcessor interface.

        Raises
        ------
        ValueError
            Always, because constrained agglomerative clustering does not
            support predicting on new data.
        """
        raise ValueError(f"predict() is not supported for {type(self.processor)}")

    def fit_predict(
        self, X: NumericArray, negative_matrix: NegativeMatrix
    ) -> ClusteringLabels:
        """
        Fit the cannot-link agglomerative clustering processor and predict labels.

        Parameters
        ----------
        X : NumericArray
            Precomputed pairwise distance matrix.
        negative_matrix : NegativeMatrix
            Symmetric boolean mask marking cannot-link pairs.

        Returns
        -------
        ClusteringLabels
            The clustering labels.

        Raises
        ------
        RuntimeError
            If the resulting labels violate a cannot-link constraint; this
            indicates an internal invariant failure rather than a usage error.
        """
        raw_labels = self.processor.fit_predict(X, negative_matrix)
        self._verify_no_violations(raw_labels, negative_matrix)
        return ClusteringLabels(labels=raw_labels)

    @property
    def labels(self) -> ClusteringLabels:
        """
        Get the clustering labels after fitting.

        Returns
        -------
        ClusteringLabels
            The clustering labels.
        """
        labels = self.processor.labels_
        if labels is None:
            raise ValueError(
                "Processor is not fitted yet. Call fit() or fit_predict() first."
            )
        return ClusteringLabels(labels=labels)

    @property
    def is_precomputed_input_required(self) -> bool:
        """
        Check if the input data must be a precomputed distance matrix.

        Returns
        -------
        bool
            Always True; constrained agglomerative clustering only operates on
            precomputed distance matrices.
        """
        return True

    @staticmethod
    def _verify_no_violations(
        raw_labels: IntegerArray, negative_matrix: NegativeMatrix
    ) -> None:
        same_cluster = raw_labels[:, None] == raw_labels[None, :]
        if np.any(same_cluster & negative_matrix):
            raise RuntimeError(
                "cannot-link constraint violated in the clustering result; this indicates an internal invariant failure"
            )
