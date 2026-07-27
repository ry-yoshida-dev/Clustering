from enum import Enum

class AgglomerativeMetric(Enum):
    """
    AgglomerativeClusteringMetric is a enum class for the metric of the AgglomerativeClustering algorithm for scikit-learn.

    Attributes:
        EUCLIDEAN: Euclidean distance
        L1: L1 distance
        L2: L2 distance
        MANHATTAN: Manhattan distance
        COSINE: Cosine distance
        PRECOMPUTED: Precomputed distance
    """
    EUCLIDEAN = "euclidean"
    L1 = "l1"
    L2 = "l2"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    PRECOMPUTED = "precomputed"
