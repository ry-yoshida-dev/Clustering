from dataclasses import dataclass
from typing import Any, Literal, cast

from sklearn.cluster import HDBSCAN  # type: ignore[import-untyped]

from .algorithm import HDBSCANAlgorithm
from .cluster_selection_method import HDBSCANClusterSelectionMethod
from .metric import HDBSCANMetric
from .store_centers import HDBSCANStoreCenters
from ...method import ClusteringMethod
from ...parameter import ClusteringParameter
from ...processor import ClusteringProcessor
from ...processors import CommonClusteringProcessor


@dataclass
class HDBSCANParameters(ClusteringParameter):
    """
    Parameters for :class:`sklearn.cluster.HDBSCAN`.

    Field names and defaults follow the stable API documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html

    ``metric`` is restricted to :class:`HDBSCANMetric`; for a custom callable
    metric, construct ``sklearn.cluster.HDBSCAN`` directly and wrap it with
    :class:`CommonClusteringProcessor`.
    """

    min_cluster_size: int = 5
    min_samples: int | None = None
    cluster_selection_epsilon: float = 0.0
    max_cluster_size: int | None = None
    metric: HDBSCANMetric = HDBSCANMetric.EUCLIDEAN
    metric_params: dict[str, Any] | None = None
    alpha: float = 1.0
    algorithm: HDBSCANAlgorithm = HDBSCANAlgorithm.AUTO
    leaf_size: int = 40
    n_jobs: int | None = None
    cluster_selection_method: HDBSCANClusterSelectionMethod = (
        HDBSCANClusterSelectionMethod.EOM
    )
    allow_single_cluster: bool = False
    store_centers: HDBSCANStoreCenters | None = None
    copy: bool | Literal["warn"] = "warn"

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

    def build_processor(self) -> ClusteringProcessor:
        store = None if self.store_centers is None else self.store_centers.value
        hdbscan_processor = HDBSCAN(
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
            allow_single_cluster=self.allow_single_cluster,
            store_centers=store,
            copy=cast(Any, self.copy),
        )
        return CommonClusteringProcessor(
            method=ClusteringMethod.HDBSCAN,
            processor=cast(Any, hdbscan_processor),
        )


if __name__ == "__main__":
    import numpy as np

    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    parameters = HDBSCANParameters(
        min_cluster_size=2,
        min_samples=1,
        metric=HDBSCANMetric.EUCLIDEAN,
        copy=True,
    )
    proc = parameters.build_processor()
    proc.fit_predict(X)
    print(proc.labels)
