from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import NumericArray
from .types import NegativeMatrix


@dataclass(frozen=True)
class CannotLinkClusteringInputs:
    """
    Validated pair of a precomputed distance matrix and its cannot-link mask.

    Attributes
    ----------
    X : NumericArray
        Precomputed pairwise distance matrix with shape (n, n).
    negative_matrix : NegativeMatrix
        Symmetric boolean mask with shape (n, n); True marks a cannot-link pair.
    """

    X: NumericArray
    negative_matrix: NegativeMatrix

    @classmethod
    def validate(
        cls,
        X: NumericArray,
        negative_matrix: NegativeMatrix,
    ) -> "CannotLinkClusteringInputs":
        """
        Validate a distance matrix and cannot-link mask before clustering.

        Parameters
        ----------
        X : NumericArray
            Precomputed pairwise distance matrix with shape (n, n).
        negative_matrix : NegativeMatrix
            Symmetric boolean mask with shape (n, n); True marks a cannot-link pair.

        Returns
        -------
        CannotLinkClusteringInputs
            The validated inputs.

        Raises
        ------
        ValueError
            If X is not a square finite matrix, if negative_matrix does not share
            X's shape, if negative_matrix is not symmetric, or if its diagonal
            contains True.
        TypeError
            If negative_matrix is not a boolean array.
        """
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("X must be a square precomputed distance matrix")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite distances")
        if negative_matrix.shape != X.shape:
            raise ValueError("negative_matrix must have the same shape as X")
        if negative_matrix.dtype != np.bool_:
            raise TypeError("negative_matrix must be a boolean array")
        if not np.array_equal(negative_matrix, negative_matrix.T):
            raise ValueError("negative_matrix must be symmetric")
        if np.any(np.diagonal(negative_matrix)):
            raise ValueError(
                "negative_matrix diagonal must be False; a point cannot be cannot-linked with itself"
            )
        return cls(X=X, negative_matrix=negative_matrix)
