from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...types import BoolArray, IntegerArray


@dataclass(frozen=True)
class KReciprocalNeighbors:
    """
    k-reciprocal nearest neighbor sets of every point, stored as padded index lists.

    Point j is a k-reciprocal neighbor of point i when each is among the other's
    k + 1 nearest points, where every point is its own nearest point. Row i of
    indices lists the k + 1 nearest points of i, nearest first, and the same row
    of is_reciprocal marks which of them are k-reciprocal neighbors of i.

    Attributes
    ----------
    indices : IntegerArray
        Nearest point indices with shape (n, k + 1), starting with the point itself.
    is_reciprocal : BoolArray
        Mask with shape (n, k + 1); True marks a k-reciprocal neighbor.
    """

    indices: IntegerArray
    is_reciprocal: BoolArray

    @classmethod
    def from_ranking(
        cls, ranking: IntegerArray, n_neighbors: int
    ) -> KReciprocalNeighbors:
        """
        Find k-reciprocal neighbors from a full nearest-point ranking.

        Parameters
        ----------
        ranking : IntegerArray
            Point indices with shape (n, n). Row i orders all points by distance
            from point i, starting with i itself.
        n_neighbors : int
            Neighborhood size k; each set is drawn from the k + 1 nearest points.

        Returns
        -------
        KReciprocalNeighbors
            The k-reciprocal neighbor sets of every point.

        Raises
        ------
        ValueError
            If ranking is not a square matrix or n_neighbors is outside [0, n).
        """
        if ranking.ndim != 2 or ranking.shape[0] != ranking.shape[1]:
            raise ValueError("ranking must be a square matrix with shape (n, n)")
        if not 0 <= n_neighbors < len(ranking):
            raise ValueError("n_neighbors must be in the range [0, n)")
        forward: IntegerArray = ranking[:, : n_neighbors + 1]
        backward: IntegerArray = forward[forward]
        points: IntegerArray = np.arange(len(ranking))[:, np.newaxis, np.newaxis]
        is_reciprocal: BoolArray = np.any(np.equal(backward, points), axis=2)
        return cls(indices=forward, is_reciprocal=is_reciprocal)

    def sizes(self) -> IntegerArray:
        """
        Count the k-reciprocal neighbors of every point.

        Returns
        -------
        IntegerArray
            Set sizes with shape (n,), each including the point itself.
        """
        sizes: IntegerArray = self.is_reciprocal.sum(axis=1, dtype=np.int64)
        return sizes

    def membership(self) -> BoolArray:
        """
        Expand the sets into a dense membership matrix.

        Returns
        -------
        BoolArray
            Matrix with shape (n, n); entry (i, j) is True when j is a
            k-reciprocal neighbor of i.
        """
        n_samples = len(self.indices)
        owners: IntegerArray = np.broadcast_to(
            np.arange(n_samples)[:, np.newaxis], self.indices.shape
        )
        is_member: BoolArray = np.zeros((n_samples, n_samples), dtype=np.bool_)
        is_member[owners[self.is_reciprocal], self.indices[self.is_reciprocal]] = True
        return is_member
