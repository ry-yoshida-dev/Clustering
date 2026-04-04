from enum import Enum


class HDBSCANStoreCenters(Enum):
    """
    Which cluster centers to compute and store (sklearn ``store_centers``).

    Use ``None`` in parameters (not this enum) to disable center storage.
    """

    CENTROID = "centroid"
    MEDOID = "medoid"
    BOTH = "both"
