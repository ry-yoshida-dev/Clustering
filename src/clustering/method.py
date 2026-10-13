from enum import Enum


class ClusteringMethod(Enum):
    """
    Clustering method implemented by a processor.

    Attributes
    ----------
    KMEANS
        KMeans clustering.
    AGGLOMERATIVE
        Agglomerative clustering.
    DBSCAN
        DBSCAN clustering.
    HDBSCAN
        HDBSCAN clustering.
    GMM
        GMM clustering.
    SNN
        Shared nearest neighbor clustering.
    """

    KMEANS = "KMeans"
    AGGLOMERATIVE = "Agglomerative"
    DBSCAN = "DBSCAN"
    HDBSCAN = "HDBSCAN"
    GMM = "GMM"
    SNN = "SNN"
