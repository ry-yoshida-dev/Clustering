from enum import Enum

class GMMCovarianceType(Enum):
    """
    Covariance type for Gaussian Mixture Model for scikit-learn.

    Parameters:
    ----------
    FULL:
        Full covariance matrix.
    DIAG:
        Diagonal covariance matrix.
    SPHERICAL:
        Spherical covariance matrix.
    TIED:
        Tied covariance matrix.
    """
    FULL = "full"
    DIAG = "diag"
    SPHERICAL = "spherical"
    TIED = "tied"

