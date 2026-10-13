from dataclasses import dataclass
from typing import override

from sklearn.cluster import KMeans

from ....method import ClusteringMethod
from ...parameter import ClusteringParameter
from ...processor import ClusteringProcessor
from ...processors import InductiveClusteringProcessor
from .initialization import KMeansInitialization


@dataclass
class KMeansParameters(ClusteringParameter):
    """
    KMeansParameters is the parameters for the KMeans clustering algorithm.

    Attributes
    ----------
    n_clusters : int
        The number of clusters.
    init : KMeansInitialization
        The initialization method for the KMeans clustering algorithm.
    max_iter : int
        The maximum number of iterations for the KMeans clustering algorithm.
    tol : float
        The tolerance for the KMeans clustering algorithm.
    verbose : int
        The verbosity level for the KMeans clustering algorithm.
    random_state : int | None
        The random state for the KMeans clustering algorithm.
    """
    n_clusters: int = 2
    init: KMeansInitialization = KMeansInitialization.KMEANS_PLUS_PLUS
    max_iter: int = 300
    tol: float = 1e-4
    verbose: int = 0
    random_state: int | None = 114514

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be greater than 0")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if self.tol <= 0:
            raise ValueError("tol must be greater than 0")

    @override
    def build_processor(self) -> ClusteringProcessor:
        kmeans_processor = KMeans(
            n_clusters=self.n_clusters,
            init=self.init.value,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        return InductiveClusteringProcessor(
            method=ClusteringMethod.KMEANS,
            processor=kmeans_processor,
        )