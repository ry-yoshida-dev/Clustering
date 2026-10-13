from dataclasses import dataclass
from typing import override

from sklearn.cluster import AgglomerativeClustering

from ....linkage import AgglomerativeLinkage
from ....method import ClusteringMethod
from ...parameter import ClusteringParameters
from ...processors import TransductiveClusteringProcessor
from .metric import AgglomerativeMetric


@dataclass(frozen=True)
class AgglomerativeClusteringParameters(ClusteringParameters):
    """
    Parameters for scikit-learn's AgglomerativeClustering.

    Exactly one stopping criterion must be set: either n_clusters or
    distance_threshold.

    Attributes
    ----------
    n_clusters : int | None
        The number of clusters to find; None when stopping by distance.
    distance_threshold : float | None
        The linkage distance above which clusters are not merged; None when
        stopping by cluster count.
    metric : AgglomerativeMetric
        The metric for the Agglomerative Clustering algorithm.
    linkage : AgglomerativeLinkage
        The linkage for the Agglomerative Clustering algorithm.
    """

    n_clusters: int | None = None
    distance_threshold: float | None = None
    metric: AgglomerativeMetric = AgglomerativeMetric.PRECOMPUTED
    linkage: AgglomerativeLinkage = AgglomerativeLinkage.SINGLE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.n_clusters is not None and self.n_clusters <= 0:
            raise ValueError("n_clusters must be greater than 0")
        if self.distance_threshold is not None and self.distance_threshold <= 0:
            raise ValueError("distance_threshold must be greater than 0")
        if (self.n_clusters is None) == (self.distance_threshold is None):
            raise ValueError(
                "exactly one of n_clusters or distance_threshold must be set"
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
