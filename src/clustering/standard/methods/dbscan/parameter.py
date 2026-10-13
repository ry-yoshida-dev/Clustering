from dataclasses import dataclass
from typing import override

from sklearn.cluster import DBSCAN

from ....method import ClusteringMethod
from ...parameter import ClusteringParameters
from ...processors import TransductiveClusteringProcessor
from .algorithm import DBSCANAlgorithm
from .metric import DBSCANMetric


@dataclass(frozen=True)
class DBSCANParameters(ClusteringParameters):
    """
    Parameters for scikit-learn's DBSCAN clustering.

    Attributes
    ----------
    eps : float
        The epsilon value for the DBSCAN algorithm.
    min_samples : int
        The minimum number of samples in a cluster.
    metric : DBSCANMetric
        The metric for the DBSCAN algorithm.
    algorithm : DBSCANAlgorithm
        The algorithm for the DBSCAN algorithm.
    leaf_size : int
        The leaf size for the DBSCAN algorithm.
    """

    eps: float = 0.5
    min_samples: int = 2
    metric: DBSCANMetric = DBSCANMetric.EUCLIDEAN
    algorithm: DBSCANAlgorithm = DBSCANAlgorithm.AUTO
    leaf_size: int = 30

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.eps <= 0:
            raise ValueError("eps must be greater than 0")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be greater than 0")
        if self.leaf_size <= 0:
            raise ValueError("leaf_size must be greater than 0")

    @override
    def build_processor(self) -> TransductiveClusteringProcessor:
        dbscan_processor = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric.value,
            algorithm=self.algorithm.value,
            leaf_size=self.leaf_size,
        )
        return TransductiveClusteringProcessor(
            method=ClusteringMethod.DBSCAN,
            processor=dbscan_processor,
            is_precomputed_metric=self.metric == DBSCANMetric.PRECOMPUTED,
        )
