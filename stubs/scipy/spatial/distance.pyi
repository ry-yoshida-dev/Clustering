import numpy as np
from numpy.typing import ArrayLike, NDArray

def cdist(
    XA: ArrayLike,
    XB: ArrayLike,
    metric: str = "euclidean",
) -> NDArray[np.float64]: ...
