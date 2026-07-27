from typing import Any, Literal

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray

class GaussianMixture:
    n_components: int
    covariance_type: Literal["full", "tied", "diag", "spherical"]
    tol: float
    reg_covar: float
    max_iter: int
    n_init: int
    init_params: Literal["kmeans", "k-means++", "random", "random_from_data"]
    weights_init: ArrayLike | None
    means_init: ArrayLike | None
    precisions_init: ArrayLike | None
    random_state: int | RandomState | None
    warm_start: bool
    verbose: int
    verbose_interval: int

    weights_: NDArray[np.floating[Any]]
    means_: NDArray[np.floating[Any]]
    covariances_: NDArray[np.floating[Any]]
    precisions_: NDArray[np.floating[Any]]
    precisions_cholesky_: NDArray[np.floating[Any]]
    converged_: bool
    n_iter_: int
    lower_bound_: float
    n_features_in_: int

    def __init__(
        self,
        n_components: int = 1,
        *,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: Literal["kmeans", "k-means++", "random", "random_from_data"] = "kmeans",
        weights_init: ArrayLike | None = None,
        means_init: ArrayLike | None = None,
        precisions_init: ArrayLike | None = None,
        random_state: int | RandomState | None = None,
        warm_start: bool = False,
        verbose: int = 0,
        verbose_interval: int = 10,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> GaussianMixture: ...
    def fit_predict(self, X: ArrayLike, y: ArrayLike | None = None) -> NDArray[np.integer[Any]]: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer[Any]]: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.floating[Any]]: ...
    def sample(
        self, n_samples: int = 1
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.integer[Any]]]: ...
    def score(self, X: ArrayLike, y: ArrayLike | None = None) -> float: ...
    def score_samples(self, X: ArrayLike) -> NDArray[np.floating[Any]]: ...
    def bic(self, X: ArrayLike) -> float: ...
    def aic(self, X: ArrayLike) -> float: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    def set_params(self, **params: object) -> GaussianMixture: ...
