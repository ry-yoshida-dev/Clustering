from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array

def connected_components(
    csgraph: coo_array,
    directed: bool = True,
    connection: Literal["weak", "strong"] = "weak",
    return_labels: Literal[True] = True,
) -> tuple[int, NDArray[np.int32]]: ...
