from enum import Enum


class KMeansInitialization(Enum):
    """
    Centroid initialization method for scikit-learn's KMeans.

    Attributes
    ----------
    RANDOM
        Random initialization.
    KMEANS_PLUS_PLUS
        KMeans++ initialization (recommended).
    """

    RANDOM = "random"
    KMEANS_PLUS_PLUS = "k-means++"
