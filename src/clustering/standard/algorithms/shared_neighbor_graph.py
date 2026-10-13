from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ...types import BoolArray, IntegerArray


@dataclass(frozen=True)
class SharedNeighborGraph:
    """
    Mutual k-nearest-neighbor links weighted by their shared neighbor count.

    A link between points i and j exists only when each point is among the
    other's k nearest neighbors. Its weight is the number of points that appear
    in both neighbor lists; i and j themselves are never counted, because no
    point is in its own neighbor list, so a weight is at most k - 1. Every
    undirected link is stored in both directions, and links of the same source
    keep the order of that source's neighbor list, nearest first.

    Membership tests use binary search over the row-sorted neighbor lists, so
    the graph is built in O(n k^2 log(n k)) time without an n-by-n matrix.
    Candidate links are processed in batches so that peak memory stays bounded
    by LOOKUP_BUDGET simultaneous lookups regardless of the number of points.

    Attributes
    ----------
    LOOKUP_BUDGET : int
        Maximum number of neighbor-list lookups held in memory at once.
    sources : IntegerArray
        Source point index of each directed link, with shape (m,).
    targets : IntegerArray
        Target point index of each directed link, with shape (m,).
    shared_counts : IntegerArray
        Number of shared nearest neighbors of each link, with shape (m,).
    """

    LOOKUP_BUDGET: ClassVar[int] = 1 << 20

    sources: IntegerArray
    targets: IntegerArray
    shared_counts: IntegerArray

    @classmethod
    def from_neighbors(cls, neighbors: IntegerArray) -> SharedNeighborGraph:
        """
        Build the graph from k-nearest-neighbor index lists.

        Parameters
        ----------
        neighbors : IntegerArray
            Nearest neighbor indices with shape (n, k). Row i lists the k
            nearest neighbors of point i, nearest first, excluding i itself,
            without duplicates.

        Returns
        -------
        SharedNeighborGraph
            The mutual k-nearest-neighbor graph with shared neighbor counts.

        Raises
        ------
        ValueError
            If neighbors is not a 2-D array.
        """
        if neighbors.ndim != 2:
            raise ValueError("neighbors must be a 2-D array with shape (n, k)")
        n_samples = len(neighbors)
        n_neighbors = len(neighbors.T)
        neighbor_lists = neighbors.astype(np.int64)
        candidate_sources = np.repeat(np.arange(n_samples, dtype=np.int64), n_neighbors)
        candidate_targets = neighbor_lists.ravel()
        membership_keys = cls._membership_keys(
            candidate_sources,
            np.sort(neighbor_lists, axis=1).ravel(),
            n_samples,
        )
        batch_size = max(1, cls.LOOKUP_BUDGET // max(1, n_neighbors))
        batches = [
            cls._mutual_links(
                neighbor_lists,
                membership_keys,
                candidate_sources[start : start + batch_size],
                candidate_targets[start : start + batch_size],
            )
            for start in range(0, max(1, len(candidate_sources)), batch_size)
        ]
        return cls(
            sources=np.concatenate([batch.sources for batch in batches]),
            targets=np.concatenate([batch.targets for batch in batches]),
            shared_counts=np.concatenate([batch.shared_counts for batch in batches]),
        )

    @staticmethod
    def _membership_keys(
        owners: IntegerArray, members: IntegerArray, n_samples: int
    ) -> IntegerArray:
        return owners * n_samples + members

    @classmethod
    def _mutual_links(
        cls,
        neighbor_lists: IntegerArray,
        membership_keys: IntegerArray,
        candidate_sources: IntegerArray,
        candidate_targets: IntegerArray,
    ) -> SharedNeighborGraph:
        n_samples = len(neighbor_lists)
        is_mutual = cls._contains(
            membership_keys,
            cls._membership_keys(candidate_targets, candidate_sources, n_samples),
        )
        sources: IntegerArray = candidate_sources[is_mutual]
        targets: IntegerArray = candidate_targets[is_mutual]
        source_neighbors: IntegerArray = np.take(neighbor_lists, sources, axis=0)
        is_shared = cls._contains(
            membership_keys,
            cls._membership_keys(targets[:, np.newaxis], source_neighbors, n_samples),
        )
        shared_counts: IntegerArray = is_shared.sum(axis=1, dtype=np.int64)
        return cls(sources=sources, targets=targets, shared_counts=shared_counts)

    @staticmethod
    def _contains(sorted_keys: IntegerArray, query_keys: IntegerArray) -> BoolArray:
        if len(sorted_keys) == 0:
            return np.zeros(query_keys.shape, dtype=np.bool_)
        positions = np.minimum(
            np.searchsorted(sorted_keys, query_keys), len(sorted_keys) - 1
        )
        matched_keys: IntegerArray = np.take(sorted_keys, positions)
        return np.equal(matched_keys, query_keys)

    def strong_links(self, min_shared_neighbors: int) -> SharedNeighborGraph:
        """
        Keep only links whose shared neighbor count reaches a threshold.

        Parameters
        ----------
        min_shared_neighbors : int
            Minimum shared neighbor count for a link to be kept.

        Returns
        -------
        SharedNeighborGraph
            The subgraph of links with at least min_shared_neighbors shared neighbors.
        """
        is_strong = self.shared_counts >= min_shared_neighbors
        return SharedNeighborGraph(
            sources=self.sources[is_strong],
            targets=self.targets[is_strong],
            shared_counts=self.shared_counts[is_strong],
        )
