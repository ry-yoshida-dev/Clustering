from enum import Enum


class HDBSCANMetric(Enum):
    """
    HDBSCANMetric is the metric for sklearn.cluster.HDBSCAN.

    NOTE
    HDBSCAN supports additional metrics via scipy/sklearn.
    If you want to use other metrics, please update this enum class.

    Attributes:
        EUCLIDEAN: Euclidean distance
        MANHATTAN: Manhattan distance
        PRECOMPUTED: Precomputed distance matrix
    """

    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    PRECOMPUTED = "precomputed"
