from enum import Enum


class KReciprocalMetric(Enum):
    """
    Original distance metric used by k-reciprocal re-ranking.

    Values are scipy.spatial.distance.cdist metric names.

    Attributes
    ----------
    EUCLIDEAN
        Euclidean distance
    SQEUCLIDEAN
        Squared Euclidean distance, as used by Zhong et al. (2017)
    MANHATTAN
        Manhattan distance
    COSINE
        Cosine distance
    PRECOMPUTED
        Precomputed distance matrix
    """

    EUCLIDEAN = "euclidean"
    SQEUCLIDEAN = "sqeuclidean"
    MANHATTAN = "cityblock"
    COSINE = "cosine"
    PRECOMPUTED = "precomputed"
