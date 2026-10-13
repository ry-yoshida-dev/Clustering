from dataclasses import dataclass
from typing import override

from ....method import ClusteringMethod
from ...algorithms import SharedNearestNeighborClustering, SNNMetric
from ...parameter import ClusteringParameter
from ...processor import ClusteringProcessor
from ...processors import TransductiveClusteringProcessor


@dataclass
class SNNParameters(ClusteringParameter):
    """
    SNNParameters is the parameters for density-based shared nearest neighbor clustering.

    Attributes
    ----------
    n_neighbors : int
        Size k of each point's nearest neighbor list.
    min_shared_neighbors : int
        Minimum number of shared nearest neighbors for two mutual neighbors to
        be strongly linked. Must be in the range [0, n_neighbors).
    min_samples : int
        Minimum number of strong links for a point to be a core point. Must be
        in the range [1, n_neighbors].
    metric : SNNMetric
        Distance metric used to find nearest neighbors.
    """

    n_neighbors: int = 10
    min_shared_neighbors: int = 3
    min_samples: int = 5
    metric: SNNMetric = SNNMetric.EUCLIDEAN

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        SharedNearestNeighborClustering.validate_parameters(
            self.n_neighbors, self.min_shared_neighbors, self.min_samples
        )

    @override
    def build_processor(self) -> ClusteringProcessor:
        algorithm = SharedNearestNeighborClustering(
            n_neighbors=self.n_neighbors,
            min_shared_neighbors=self.min_shared_neighbors,
            min_samples=self.min_samples,
            metric=self.metric,
        )
        return TransductiveClusteringProcessor(
            method=ClusteringMethod.SNN,
            processor=algorithm,
            is_precomputed_metric=self.metric == SNNMetric.PRECOMPUTED,
        )
