from enum import Enum


class HDBSCANClusterSelectionMethod(Enum):
    """
    How sklearn.cluster.HDBSCAN extracts flat clusters from the hierarchy.

    Attributes:
        EOM: Excess of Mass — default, often preferred for varied density.
        LEAF: Use leaves of the condensed tree.
    """

    EOM = "eom"
    LEAF = "leaf"
