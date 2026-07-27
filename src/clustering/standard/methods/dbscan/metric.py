from enum import Enum

class DBSCANMetric(Enum):
    """
    DBSCANMetric is the metric of the DBSCAN algorithm for scikit-learn.

    NOTE
    There are another metrics for DBSCAN algorithm.
    If you want to use other metrics, please update this enum class.

    Attributes:
        EUCLIDEAN: Euclidean distance
        MANHATTAN: Manhattan distance
        PRECOMPUTED: Precomputed distance
    """
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    PRECOMPUTED = "precomputed"