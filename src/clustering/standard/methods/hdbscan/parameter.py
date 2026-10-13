from dataclasses import dataclass
from typing import override

from sklearn.cluster import HDBSCAN

from ....method import ClusteringMethod
from ...parameter import ClusteringParameter
from ...processors import TransductiveClusteringProcessor
from .algorithm import HDBSCANAlgorithm
from .cluster_selection_method import HDBSCANClusterSelectionMethod
from .metric import HDBSCANMetric
from .store_centers import HDBSCANStoreCenters


@dataclass
class HDBSCANParameters(ClusteringParameter):
    """
    Parameters for :class:`sklearn.cluster.HDBSCAN`.

    Field names and defaults follow the stable API documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html

    ``metric`` is restricted to :class:`HDBSCANMetric`; for a custom callable
    metric, construct ``sklearn.cluster.HDBSCAN`` directly and wrap it with
    :class:`TransductiveClusteringProcessor`.

    Attributes
    ----------
    min_cluster_size : int
        Minimum number of samples in a group for it to be a cluster.
    min_samples : int | None
        Number of samples in a neighborhood for a point to be a core point;
        defaults to min_cluster_size when None.
    cluster_selection_epsilon : float
        Distance threshold below which clusters are merged.
    max_cluster_size : int | None
        Upper bound on the size of a cluster returned by EOM selection; no
        bound when None.
    metric : HDBSCANMetric
        Distance metric between points.
    metric_params : dict[str, object] | None
        Extra keyword arguments for the metric function.
    alpha : float
        Distance scaling parameter used in robust single linkage.
    algorithm : HDBSCANAlgorithm
        Nearest neighbor search backend for core distances.
    leaf_size : int
        Leaf size for tree-based neighbor search backends.
    n_jobs : int | None
        Number of parallel jobs for the neighbor search.
    cluster_selection_method : HDBSCANClusterSelectionMethod
        How flat clusters are extracted from the cluster tree.
    is_single_cluster_allowed : bool
        Whether the result may consist of a single cluster.
    store_centers : HDBSCANStoreCenters | None
        Which cluster centers to compute and store; none when None.
    is_input_copied : bool
        Whether the input is copied before any in-place modification, so the
        data passed to fit is never overwritten.
    """

    min_cluster_size: int = 5
    min_samples: int | None = None
    cluster_selection_epsilon: float = 0.0
    max_cluster_size: int | None = None
    metric: HDBSCANMetric = HDBSCANMetric.EUCLIDEAN
    metric_params: dict[str, object] | None = None
    alpha: float = 1.0
    algorithm: HDBSCANAlgorithm = HDBSCANAlgorithm.AUTO
    leaf_size: int = 40
    n_jobs: int | None = None
    cluster_selection_method: HDBSCANClusterSelectionMethod = (
        HDBSCANClusterSelectionMethod.EOM
    )
    is_single_cluster_allowed: bool = False
    store_centers: HDBSCANStoreCenters | None = None
    is_input_copied: bool = True

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be greater than 0")
        if self.min_samples is not None and self.min_samples <= 0:
            raise ValueError("min_samples must be greater than 0 when set")
        if self.leaf_size <= 0:
            raise ValueError("leaf_size must be greater than 0")
        if self.cluster_selection_epsilon < 0:
            raise ValueError("cluster_selection_epsilon must be non-negative")
        if self.max_cluster_size is not None and self.max_cluster_size <= 0:
            raise ValueError("max_cluster_size must be greater than 0 when set")
        if self.alpha <= 0:
            raise ValueError("alpha must be greater than 0")

    @override
    def build_processor(self) -> TransductiveClusteringProcessor:
        hdbscan_processor: HDBSCAN = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            max_cluster_size=self.max_cluster_size,
            metric=self.metric.value,
            metric_params=self.metric_params,
            alpha=self.alpha,
            algorithm=self.algorithm.value,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
            cluster_selection_method=self.cluster_selection_method.value,
            allow_single_cluster=self.is_single_cluster_allowed,
            store_centers=(
                None if self.store_centers is None else self.store_centers.value
            ),
            copy=self.is_input_copied,
        )
        return TransductiveClusteringProcessor(
            method=ClusteringMethod.HDBSCAN,
            processor=hdbscan_processor,
            is_precomputed_metric=self.metric == HDBSCANMetric.PRECOMPUTED,
        )
