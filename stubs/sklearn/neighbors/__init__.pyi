from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

class NearestNeighbors:
    n_neighbors: int
    radius: float
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"]
    leaf_size: int
    metric: str
    p: float
    metric_params: dict[str, object] | None
    n_jobs: int | None

    n_features_in_: int
    n_samples_fit_: int

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        radius: float = 1.0,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: float = 2,
        metric_params: dict[str, object] | None = None,
        n_jobs: int | None = None,
    ) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> NearestNeighbors: ...
    @overload
    def kneighbors(
        self,
        X: ArrayLike | None = None,
        n_neighbors: int | None = None,
        return_distance: Literal[True] = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.intp]]: ...
    @overload
    def kneighbors(
        self,
        X: ArrayLike | None = None,
        n_neighbors: int | None = None,
        *,
        return_distance: Literal[False],
    ) -> NDArray[np.intp]: ...
