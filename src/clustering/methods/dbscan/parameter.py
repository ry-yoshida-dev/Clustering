from dataclasses import dataclass
from sklearn.cluster import DBSCAN # type: ignore

from .metric import DBSCANMetric
from .algorithm import DBSCANAlgorithm
from ...method import ClusteringMethod
from ...parameter import ClusteringParameter
from ...processor import ClusteringProcessor
from ...processors import CommonClusteringProcessor

@dataclass
class DBSCANParameters(ClusteringParameter):
    """
    DBSCANParameters is the parameters for the DBSCAN clustering algorithm.

    Attributes:
    ----------
    eps: float
        The epsilon value for the DBSCAN algorithm.
    min_samples: int
        The minimum number of samples in a cluster.
    metric: DBSCANMetric
        The metric for the DBSCAN algorithm.
    algorithm: DBSCANAlgorithm
        The algorithm for the DBSCAN algorithm.
    leaf_size: int
        The leaf size for the DBSCAN algorithm.
    """
    eps: float = 0.5
    min_samples: int = 2
    metric: DBSCANMetric = DBSCANMetric.EUCLIDEAN
    algorithm: DBSCANAlgorithm = DBSCANAlgorithm.AUTO
    leaf_size: int = 30

    def __post_init__(self):
        self._validate_parameters()

    def _validate_parameters(self):
        if self.eps <= 0:
            raise ValueError("eps must be greater than 0")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be greater than 0")
        if self.leaf_size <= 0:
            raise ValueError("leaf_size must be greater than 0")

    def build_processor(self) -> ClusteringProcessor:
        dbscan_processor = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric.value,
            algorithm=self.algorithm.value,
            leaf_size=self.leaf_size,
            )
        return CommonClusteringProcessor(
            method=ClusteringMethod.DBSCAN,
            processor=dbscan_processor,
        )

###########################

if __name__ == "__main__":
    import numpy as np
    from scipy.spatial.distance import cdist
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    distance_matrix = cdist(X, X, metric='euclidean')
    parameters = DBSCANParameters(
        eps=0.5,
        min_samples=2,
        metric=DBSCANMetric.EUCLIDEAN,
        algorithm=DBSCANAlgorithm.AUTO,
        leaf_size=30,
    )