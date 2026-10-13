from dataclasses import dataclass
from typing import override

from sklearn.mixture import GaussianMixture

from ....method import ClusteringMethod
from ...parameter import ClusteringParameters
from ...processors import InductiveClusteringProcessor
from .covariance_type import GMMCovarianceType


@dataclass(frozen=True)
class GMMParameters(ClusteringParameters):
    """
    Parameters for scikit-learn's GaussianMixture clustering.

    Attributes
    ----------
    n_components : int
        The number of components in the GMM.
    covariance_type : GMMCovarianceType
        The covariance type for the GMM.
    random_state : int | None
        The random state for the GMM.
    """

    n_components: int = 2
    covariance_type: GMMCovarianceType = GMMCovarianceType.FULL
    random_state: int | None = 810

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.n_components <= 0:
            raise ValueError("n_components must be greater than 0")

    @override
    def build_processor(self) -> InductiveClusteringProcessor:
        gmm_processor = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type.value,
            random_state=self.random_state,
        )
        return InductiveClusteringProcessor(
            method=ClusteringMethod.GMM,
            processor=gmm_processor,
        )
