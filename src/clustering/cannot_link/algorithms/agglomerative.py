from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ...standard.methods.agglomerative.linkage import AgglomerativeLinkage
from ...types import IntegerArray, NumericArray
from ..inputs import CannotLinkClusteringInputs
from ..types import NegativeMatrix

type MergeRowFn = Callable[[NumericArray, NumericArray, int, int], NumericArray]


@dataclass
class ConstrainedAgglomerativeClustering:
    """
    Agglomerative clustering with hard cannot-link constraints.

    Unlike sklearn's `AgglomerativeClustering`, this estimator tracks which
    cluster pairs are forbidden from merging as a separate boolean state, rather
    than relying on inflating distances so that the linkage criterion happens to
    avoid a merge. Distance inflation only gives a hard guarantee for complete
    linkage (its cluster distance is a maximum, so one inflated pair dominates);
    for single linkage (a minimum) and average linkage (a mean that dilutes as
    clusters grow) it does not. Tracking forbidden merges explicitly makes the
    constraint exact for single, complete, and average linkage alike.

    Ward linkage is not supported: it requires Euclidean feature vectors to
    compute cluster centroids and variances, which does not fit a precomputed
    distance matrix. Callers must not pass AgglomerativeLinkage.WARD.

    Merges are located with the nearest-neighbor-chain algorithm (Murtagh
    1984) rather than by rescanning the full active-by-active distance
    submatrix on every merge. Single, complete, and average linkage all
    satisfy the Lance-Williams reducibility property, so the chain always
    closes on a reciprocal-nearest-neighbor pair and the resulting merge
    tree matches the one a full global-argmin search would build. The chain
    runs to completion first, without applying the distance threshold; the
    order in which it discovers reciprocal pairs is not guaranteed to be
    sorted by height (only the finished merge tree is), so every merge is
    recorded and only applied to the union-find structure afterward, sorted
    by height and cut at distance_threshold. Forbidden pairs are excluded
    from every neighbor lookup rather than folded into the distances, so
    eligibility stays exact instead of relying on distance inflation. This
    brings the running time down from O(n^3) to O(n^2) without changing the
    result.

    The linkage criterion is resolved to a merge-row function once, in
    __post_init__, so the merge hot loop never branches on the linkage enum.

    Attributes
    ----------
    linkage : AgglomerativeLinkage
        The linkage criterion used to rank eligible merges.
    distance_threshold : float
        The linkage distance above which merging stops.
    labels_ : IntegerArray | None
        Cluster labels assigned by the last fit call, if any.
    """

    linkage: AgglomerativeLinkage
    distance_threshold: float
    labels_: IntegerArray | None = field(default=None, init=False)
    _merge_row: MergeRowFn = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.distance_threshold <= 0:
            raise ValueError("distance_threshold must be greater than 0")
        self._merge_row = self._resolve_merge_row(self.linkage)

    @staticmethod
    def _resolve_merge_row(linkage: AgglomerativeLinkage) -> MergeRowFn:
        match linkage:
            case AgglomerativeLinkage.SINGLE:
                return lambda dist_a, dist_b, size_a, size_b: np.minimum(dist_a, dist_b)
            case AgglomerativeLinkage.COMPLETE:
                return lambda dist_a, dist_b, size_a, size_b: np.maximum(dist_a, dist_b)
            case AgglomerativeLinkage.AVERAGE:
                return lambda dist_a, dist_b, size_a, size_b: (
                    size_a * dist_a + size_b * dist_b
                ) / (size_a + size_b)
            case AgglomerativeLinkage.WARD:
                raise ValueError(
                    "ward linkage is not supported for cannot-link constrained agglomerative clustering"
                )

    def fit(self, X: NumericArray, negative_matrix: NegativeMatrix) -> None:
        """
        Fit the constrained agglomerative clustering algorithm.

        Parameters
        ----------
        X : NumericArray
            Precomputed pairwise distance matrix with shape (n, n).
        negative_matrix : NegativeMatrix
            Symmetric boolean mask with shape (n, n); True marks a cannot-link
            pair.
        """
        self.labels_ = self._run(X, negative_matrix)

    def fit_predict(self, X: NumericArray, negative_matrix: NegativeMatrix) -> IntegerArray:
        """
        Fit the algorithm and return the resulting cluster labels.

        Parameters
        ----------
        X : NumericArray
            Precomputed pairwise distance matrix with shape (n, n).
        negative_matrix : NegativeMatrix
            Symmetric boolean mask with shape (n, n); True marks a cannot-link
            pair.

        Returns
        -------
        IntegerArray
            Cluster label per point.
        """
        self.fit(X, negative_matrix)
        assert self.labels_ is not None
        return self.labels_

    def _run(self, X: NumericArray, negative_matrix: NegativeMatrix) -> IntegerArray:
        inputs = CannotLinkClusteringInputs.validate(X, negative_matrix)
        n = inputs.X.shape[0]
        dist = inputs.X.astype(np.float64, copy=True)
        np.fill_diagonal(dist, np.inf)
        forbidden = inputs.negative_matrix.copy()
        size = np.ones(n, dtype=np.int64)
        active = np.ones(n, dtype=bool)
        root = np.arange(n, dtype=np.int64)

        candidates = set(range(n))
        chain: list[int] = []
        merges: list[tuple[float, int, int]] = []

        while True:
            if not chain:
                if not candidates:
                    break
                chain.append(next(iter(candidates)))

            current = chain[-1]
            neighbor, distance = self._nearest_eligible(current, dist, forbidden, active)

            if neighbor is None:
                candidates.discard(current)
                del chain[-1]
                continue

            if len(chain) >= 2 and neighbor == chain[-2]:
                a, b = (current, neighbor) if current < neighbor else (neighbor, current)
                self._merge(a, b, dist, forbidden, size, active)
                merges.append((distance, a, b))
                candidates.discard(b)
                del chain[-2:]
                continue

            chain.append(neighbor)

        for height, a, b in sorted(merges, key=lambda merge: merge[0]):
            if height > self.distance_threshold:
                break
            root[b] = a

        return self._labels_from_roots(root, n)

    @staticmethod
    def _nearest_eligible(
        current: int,
        dist: NumericArray,
        forbidden: NegativeMatrix,
        active: NegativeMatrix,
    ) -> tuple[int | None, float]:
        eligible_row = np.where(active & ~forbidden[current], dist[current], np.inf)
        neighbor = int(np.argmin(eligible_row))
        distance = float(eligible_row[neighbor])
        if not np.isfinite(distance):
            return None, np.inf
        return neighbor, distance

    def _merge(
        self,
        a: int,
        b: int,
        dist: NumericArray,
        forbidden: NegativeMatrix,
        size: IntegerArray,
        active: NegativeMatrix,
    ) -> None:
        new_row = self._merge_row(dist[a], dist[b], int(size[a]), int(size[b]))

        dist[a, :] = new_row
        dist[:, a] = new_row
        dist[a, a] = np.inf
        forbidden_row = forbidden[a] | forbidden[b]
        forbidden[a, :] = forbidden_row
        forbidden[:, a] = forbidden_row
        size[a] += size[b]
        active[b] = False

    @staticmethod
    def _labels_from_roots(root: IntegerArray, n: int) -> IntegerArray:
        resolved = np.empty(n, dtype=np.int64)
        for point in range(n):
            index = point
            while root[index] != index:
                root[index] = root[root[index]]
                index = root[index]
            resolved[point] = index
        _, labels = np.unique(resolved, return_inverse=True)
        return labels.astype(np.int64)
