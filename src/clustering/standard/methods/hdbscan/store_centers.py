from enum import Enum


class HDBSCANStoreCenters(Enum):
    """
    Which cluster centers scikit-learn's HDBSCAN computes and stores.

    Use None in HDBSCANParameters, not this enum, to disable center storage.

    Attributes
    ----------
    CENTROID
        Store weighted centroids.
    MEDOID
        Store medoids.
    BOTH
        Store both centroids and medoids.
    """

    CENTROID = "centroid"
    MEDOID = "medoid"
    BOTH = "both"
