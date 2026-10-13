from enum import Enum


class HDBSCANAlgorithm(Enum):
    """
    Nearest neighbor search backend for scikit-learn's HDBSCAN core distances.

    Attributes
    ----------
    AUTO
        Let scikit-learn choose the backend.
    BRUTE
        Brute-force search.
    KD_TREE
        KD tree search.
    BALL_TREE
        Ball tree search.
    """

    AUTO = "auto"
    BRUTE = "brute"
    KD_TREE = "kd_tree"
    BALL_TREE = "ball_tree"
