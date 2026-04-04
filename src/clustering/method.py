from enum import Enum

class ClusteringMethod(Enum):
    """
    ClusteringMethod is the method of the clustering processor.

    Attributes:
        KMEANS: KMeans clustering.
        AGGLOMERATIVE: Agglomerative clustering.
        DBSCAN: DBSCAN clustering.
        HDBSCAN: HDBSCAN clustering.
        GMM: GMM clustering.
    """
    KMEANS = "KMeans"
    AGGLOMERATIVE = "Agglomerative"
    DBSCAN = "DBSCAN"
    HDBSCAN = "HDBSCAN"
    GMM = "GMM"