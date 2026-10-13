from enum import Enum


class SNNMetric(Enum):
    """
    SNNMetric is the metric used to find k-nearest neighbors for shared nearest neighbor clustering.

    NOTE
    sklearn.neighbors.NearestNeighbors supports additional metrics.
    If you want to use other metrics, please update this enum class.

    Attributes:
        EUCLIDEAN: Euclidean distance
        MANHATTAN: Manhattan distance
        COSINE: Cosine distance
        PRECOMPUTED: Precomputed distance matrix
    """

    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    PRECOMPUTED = "precomputed"
