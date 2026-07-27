from enum import Enum


class HDBSCANAlgorithm(Enum):
    """
    Core-distance computation strategy for sklearn.cluster.HDBSCAN.

    Attributes correspond to sklearn's ``algorithm`` parameter.
    """

    AUTO = "auto"
    BRUTE = "brute"
    KD_TREE = "kd_tree"
    BALL_TREE = "ball_tree"
