from enum import Enum

class DBSCANAlgorithm(Enum):
    """
    DBSCANAlgorithm is the algorithm of the DBSCAN algorithm for scikit-learn.

    Attributes:
        AUTO: Auto algorithm
        BALL_TREE: Ball tree algorithm
        KD_TREE: KD tree algorithm
        BRUTE: Brute force algorithm
    """
    AUTO = "auto"
    BALL_TREE = "ball_tree"
    KD_TREE = "kd_tree"
    BRUTE = "brute"