from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray

class KMeans:
    n_clusters: int
    init: Literal["k-means++", "random"] | ArrayLike
    n_init: int | Literal["auto"]
    max_iter: int
    tol: float
    verbose: int
    random_state: int | RandomState | None
    copy_x: bool
    algorithm: Literal["lloyd", "elkan"]

    cluster_centers_: NDArray[np.floating[Any]]
    labels_: NDArray[np.integer[Any]] | None
    inertia_: float
    n_iter_: int
    n_features_in_: int

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        init: Literal["k-means++", "random"] | ArrayLike = "k-means++",
        n_init: int | Literal["auto"] = "auto",
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: int = 0,
        random_state: int | RandomState | None = None,
        copy_x: bool = True,
        algorithm: Literal["lloyd", "elkan"] = "lloyd",
    ) -> None: ...
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> KMeans: ...
    def fit_predict(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> NDArray[np.integer[Any]]: ...
    def predict(
        self,
        X: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> NDArray[np.integer[Any]]: ...
    def transform(self, X: ArrayLike) -> NDArray[np.floating[Any]]: ...
    def fit_transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> NDArray[np.floating[Any]]: ...
    def score(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> float: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    def set_params(self, **params: object) -> KMeans: ...

class DBSCAN:
    eps: float
    min_samples: int
    metric: str | Callable[[ArrayLike, ArrayLike], float]
    metric_params: dict[str, object] | None
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"]
    leaf_size: int
    p: float | None
    n_jobs: int | None

    core_sample_indices_: NDArray[np.integer[Any]]
    components_: NDArray[np.floating[Any]]
    labels_: NDArray[np.integer[Any]]
    n_features_in_: int

    def __init__(
        self,
        eps: float = 0.5,
        *,
        min_samples: int = 5,
        metric: str | Callable[[ArrayLike, ArrayLike], float] = "euclidean",
        metric_params: dict[str, object] | None = None,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        p: float | None = None,
        n_jobs: int | None = None,
    ) -> None: ...
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> DBSCAN: ...
    def fit_predict(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> NDArray[np.integer[Any]]: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    def set_params(self, **params: object) -> DBSCAN: ...

class AgglomerativeClustering:
    n_clusters: int | None
    metric: str | Callable[[ArrayLike], ArrayLike]
    memory: str | object | None
    connectivity: ArrayLike | Callable[[ArrayLike], ArrayLike] | None
    compute_full_tree: bool | Literal["auto"]
    linkage: Literal["ward", "complete", "average", "single"]
    distance_threshold: float | None
    compute_distances: bool

    labels_: NDArray[np.integer[Any]]
    n_clusters_: int
    n_leaves_: int
    n_connected_components_: int
    children_: NDArray[np.integer[Any]]
    distances_: NDArray[np.floating[Any]] | None
    n_features_in_: int

    def __init__(
        self,
        n_clusters: int | None = 2,
        *,
        metric: str | Callable[[ArrayLike], ArrayLike] = "euclidean",
        memory: str | object | None = None,
        connectivity: ArrayLike | Callable[[ArrayLike], ArrayLike] | None = None,
        compute_full_tree: bool | Literal["auto"] = "auto",
        linkage: Literal["ward", "complete", "average", "single"] = "ward",
        distance_threshold: float | None = None,
        compute_distances: bool = False,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> AgglomerativeClustering: ...
    def fit_predict(self, X: ArrayLike, y: ArrayLike | None = None) -> NDArray[np.integer[Any]]: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    def set_params(self, **params: object) -> AgglomerativeClustering: ...

class HDBSCAN:
    min_cluster_size: int
    min_samples: int | None
    cluster_selection_epsilon: float
    max_cluster_size: int | None
    metric: str | Callable[[ArrayLike, ArrayLike], float]
    metric_params: dict[str, object] | None
    alpha: float
    algorithm: Literal["auto", "brute", "kd_tree", "ball_tree"]
    leaf_size: int
    n_jobs: int | None
    cluster_selection_method: Literal["eom", "leaf"]
    allow_single_cluster: bool
    store_centers: Literal["centroid", "medoid", "both"] | None
    copy: bool | Literal["warn"]

    labels_: NDArray[np.integer[Any]]
    probabilities_: NDArray[np.floating[Any]]
    centroids_: NDArray[np.floating[Any]]
    medoids_: NDArray[np.floating[Any]]
    n_features_in_: int

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        cluster_selection_epsilon: float = 0.0,
        max_cluster_size: int | None = None,
        metric: str | Callable[[ArrayLike, ArrayLike], float] = "euclidean",
        metric_params: dict[str, object] | None = None,
        alpha: float = 1.0,
        algorithm: Literal["auto", "brute", "kd_tree", "ball_tree"] = "auto",
        leaf_size: int = 40,
        n_jobs: int | None = None,
        cluster_selection_method: Literal["eom", "leaf"] = "eom",
        allow_single_cluster: bool = False,
        store_centers: Literal["centroid", "medoid", "both"] | None = None,
        copy: bool | Literal["warn"] = "warn",
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> HDBSCAN: ...
    def fit_predict(self, X: ArrayLike, y: ArrayLike | None = None) -> NDArray[np.integer[Any]]: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    def set_params(self, **params: object) -> HDBSCAN: ...
