from enum import Enum


class AgglomerativeLinkage(Enum):
    """
    Linkage criterion for agglomerative clustering.

    Shared by the scikit-learn backed AgglomerativeClusteringParameters and the
    cannot-link constrained agglomerative clustering.

    Attributes
    ----------
    SINGLE
        Single linkage
    COMPLETE
        Complete linkage
    AVERAGE
        Average linkage
    WARD
        Ward linkage
    """

    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    WARD = "ward"
