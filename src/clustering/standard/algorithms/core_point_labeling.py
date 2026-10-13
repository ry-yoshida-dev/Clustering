from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from ...types import BoolArray, IntegerArray
from .shared_neighbor_graph import SharedNeighborGraph


@dataclass(frozen=True)
class CorePointLabeling:
    """
    Cluster labeling of a strong-link graph around its core points.

    A point's SNN density is its number of strong links; points whose density
    is at least min_samples are core points. Core points joined by strong
    links form clusters. Every non-core point joins the cluster of the core
    point it shares the most neighbors with over a strong link, and ties go to
    the link that appears first in strong_links, which for graphs built by
    SharedNeighborGraph.from_neighbors is the nearer neighbor. Points without
    a strong link to any core point are noise and labeled -1.

    Attributes
    ----------
    strong_links : SharedNeighborGraph
        Strong links, stored in both directions.
    n_samples : int
        Number of points in the graph.
    min_samples : int
        Minimum number of strong links for a point to be a core point.
    """

    strong_links: SharedNeighborGraph
    n_samples: int
    min_samples: int

    def labels(self) -> IntegerArray:
        """
        Compute the cluster label of every point.

        Returns
        -------
        IntegerArray
            Cluster label per point, numbered consecutively from 0; noise
            points are labeled -1.
        """
        density = np.bincount(self.strong_links.sources, minlength=self.n_samples)
        is_core: BoolArray = density >= self.min_samples
        components = self._core_components(is_core)
        labels = np.full(self.n_samples, -1, dtype=np.int64)
        labels[is_core] = components[is_core]
        self._assign_border_points(labels, is_core, components)
        return self._compact_labels(labels)

    def _core_components(self, is_core: BoolArray) -> IntegerArray:
        is_core_link = is_core[self.strong_links.sources] & is_core[self.strong_links.targets]
        sources = self.strong_links.sources[is_core_link]
        targets = self.strong_links.targets[is_core_link]
        adjacency = coo_array(
            (np.ones_like(sources, dtype=np.int8), (sources, targets)),
            shape=(self.n_samples, self.n_samples),
        )
        _, components = connected_components(adjacency, directed=False)
        return components.astype(np.int64)

    def _assign_border_points(
        self,
        labels: IntegerArray,
        is_core: BoolArray,
        components: IntegerArray,
    ) -> None:
        is_border_link = ~is_core[self.strong_links.sources] & is_core[self.strong_links.targets]
        sources = self.strong_links.sources[is_border_link]
        targets = self.strong_links.targets[is_border_link]
        shared_counts = self.strong_links.shared_counts[is_border_link]
        order = np.lexsort((-shared_counts, sources))
        border_points, first_link = np.unique(sources[order], return_index=True)
        labels[border_points] = components[targets[order][first_link]]

    @staticmethod
    def _compact_labels(labels: IntegerArray) -> IntegerArray:
        is_clustered = labels >= 0
        compacted = np.full_like(labels, -1)
        _, inverse = np.unique(labels[is_clustered], return_inverse=True)
        compacted[is_clustered] = inverse
        return compacted
