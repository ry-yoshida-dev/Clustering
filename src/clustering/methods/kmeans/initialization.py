from enum import Enum

class KMeansInitialization(Enum):
    """
    KMeansInitialization is the initialization method for the KMeans clustering algorithm.

    Attributes:
    ----------
    RANDOM: Random initialization.
    KMEANS_PLUS_PLUS: KMeans++ initialization. <- recommended
    """
    RANDOM = "random"
    KMEANS_PLUS_PLUS = "k-means++"