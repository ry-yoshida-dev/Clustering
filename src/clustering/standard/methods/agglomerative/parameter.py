from dataclasses import dataclass
from typing import override

from sklearn.cluster import AgglomerativeClustering

from ....linkage import AgglomerativeLinkage
from ....method import ClusteringMethod
from ...parameter import ClusteringParameter
from ...processors import TransductiveClusteringProcessor
from .metric import AgglomerativeMetric


@dataclass
class AgglomerativeClusteringParameters(ClusteringParameter):
    """
    AgglomerativeClusteringParameters is the parameters for the Agglomerative Clustering algorithm.

    Attributes
    ----------
    n_clusters : int | None
        The number of clusters.
    distance_threshold : float | None
        The distance threshold for the Agglomerative Clustering algorithm.
    metric : AgglomerativeMetric
        The metric for the Agglomerative Clustering algorithm.
    linkage : AgglomerativeLinkage
        The linkage for the Agglomerative Clustering algorithm.
    """

    n_clusters: int | None = None
    distance_threshold: float | None = 0.3
    metric: AgglomerativeMetric = AgglomerativeMetric.PRECOMPUTED
    linkage: AgglomerativeLinkage = AgglomerativeLinkage.SINGLE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.n_clusters is not None and self.n_clusters <= 0:
            raise ValueError("n_clusters must be greater than 0")
        if self.distance_threshold is not None and self.distance_threshold <= 0:
            raise ValueError("distance_threshold must be greater than 0")
        if self.n_clusters is None and self.distance_threshold is None:
            raise ValueError("n_clusters or distance_threshold must be set")
        if self.n_clusters is not None and self.distance_threshold is not None:
            raise ValueError(
                "n_clusters and distance_threshold cannot be set at the same time"
            )
        if (
            self.linkage == AgglomerativeLinkage.WARD
            and self.metric != AgglomerativeMetric.EUCLIDEAN
        ):
            raise ValueError("Ward linkage requires Euclidean metric")

    @override
    def build_processor(self) -> TransductiveClusteringProcessor:
        agglomerative_processor = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric=self.metric.value,
            distance_threshold=self.distance_threshold,
            linkage=self.linkage.value,
        )
        return TransductiveClusteringProcessor(
            method=ClusteringMethod.AGGLOMERATIVE,
            processor=agglomerative_processor,
            is_precomputed_metric=self.metric == AgglomerativeMetric.PRECOMPUTED,
        )
