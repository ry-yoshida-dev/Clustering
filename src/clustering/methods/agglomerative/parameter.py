from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering # type: ignore

from .linkage import AgglomerativeLinkage
from .metric import AgglomerativeMetric
from ...method import ClusteringMethod
from ...parameter import ClusteringParameter
from ...processor import ClusteringProcessor
from ...processors import CommonClusteringProcessor

@dataclass
class AgglomerativeClusteringParameters(ClusteringParameter):
    """
    AgglomerativeClusteringParameters is the parameters for the Agglomerative Clustering algorithm.

    Attributes:
    ----------
    n_clusters: int | None
        The number of clusters.
    distance_threshold: float | None
        The distance threshold for the Agglomerative Clustering algorithm.
    metric: AgglomerativeMetric
        The metric for the Agglomerative Clustering algorithm.
    linkage: AgglomerativeLinkage
        The linkage for the Agglomerative Clustering algorithm.
    """
    n_clusters: int | None = None
    distance_threshold: float | None = 0.3
    metric: AgglomerativeMetric = AgglomerativeMetric.PRECOMPUTED
    linkage: AgglomerativeLinkage = AgglomerativeLinkage.SINGLE

    def __post_init__(self):
        self._validate_parameters()

    def _validate_parameters(self):
        if self.n_clusters is not None and self.n_clusters <= 0:
            raise ValueError("n_clusters must be greater than 0")
        if self.distance_threshold is not None and self.distance_threshold <= 0:
            raise ValueError("distance_threshold must be greater than 0")
        if self.n_clusters is None and self.distance_threshold is None:
            raise ValueError("n_clusters or distance_threshold must be set")
        if self.n_clusters is not None and self.distance_threshold is not None:
            raise ValueError("n_clusters and distance_threshold cannot be set at the same time")
        if self.linkage == AgglomerativeLinkage.WARD and self.metric != AgglomerativeMetric.EUCLIDEAN:
            raise ValueError("Ward linkage requires Euclidean metric")        

    def build_processor(self) -> ClusteringProcessor:
        agglomerative_processor = AgglomerativeClustering(
            n_clusters=self.n_clusters, # type: ignore
            metric=self.metric.value,
            distance_threshold=self.distance_threshold,
            linkage=self.linkage.value
            )
        return CommonClusteringProcessor(
            method=ClusteringMethod.AGGLOMERATIVE,
            processor=agglomerative_processor,
        )
        

if __name__ == "__main__":
    import numpy as np
    from scipy.spatial.distance import cdist
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    distance_matrix = cdist(X, X, metric='euclidean')
    parameters = AgglomerativeClusteringParameters(
        n_clusters=None,
        distance_threshold=0.3,
        metric=AgglomerativeMetric.EUCLIDEAN,
        linkage=AgglomerativeLinkage.SINGLE,
        )
    processor = parameters.build_processor()
    processor.fit_predict(distance_matrix)
    print(processor.labels)