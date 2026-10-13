from enum import Enum


class DBSCANAlgorithm(Enum):
    """
    Nearest neighbor search backend for scikit-learn's DBSCAN.

    Attributes
    ----------
    AUTO
        Auto algorithm
    BALL_TREE
        Ball tree algorithm
    KD_TREE
        KD tree algorithm
    BRUTE
        Brute force algorithm
    """

    AUTO = "auto"
    BALL_TREE = "ball_tree"
    KD_TREE = "kd_tree"
    BRUTE = "brute"
