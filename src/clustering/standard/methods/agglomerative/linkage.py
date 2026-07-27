from enum import Enum

class AgglomerativeLinkage(Enum):
    """
    AgglomerativeLinkage is a enum class for the linkage of the AgglomerativeClustering algorithm for scikit-learn.

    Attributes:
        SINGLE: Single linkage
        COMPLETE: Complete linkage
        AVERAGE: Average linkage
        WARD: Ward linkage
    """
    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    WARD = "ward"