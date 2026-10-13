from dataclasses import dataclass
from typing import override

from ....linkage import AgglomerativeLinkage
from ....method import ClusteringMethod
from ...parameter import CannotLinkClusteringParameters
from ...processor import CannotLinkClusteringProcessor
from .estimator import ConstrainedAgglomerativeClustering
from .processor import CannotLinkAgglomerativeProcessor


@dataclass(frozen=True)
class CannotLinkAgglomerativeClusteringParameters(CannotLinkClusteringParameters):
    """
    Parameters for cannot-link constrained agglomerative clustering.

    n_clusters is intentionally not offered: reaching an exact cluster count can
    force a merge that violates a cannot-link constraint, so only the
    distance_threshold stopping criterion is supported, which always leaves an
    exact, constraint-respecting result (unmerged points simply stay separate).

    Attributes
    ----------
    distance_threshold : float
        The distance threshold above which merging stops.
    linkage : AgglomerativeLinkage
        The linkage criterion for the constrained agglomerative clustering
        algorithm. WARD is rejected because it requires Euclidean feature
        vectors rather than a precomputed distance matrix.
    """

    distance_threshold: float = 0.3
    linkage: AgglomerativeLinkage = AgglomerativeLinkage.COMPLETE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.distance_threshold <= 0:
            raise ValueError("distance_threshold must be greater than 0")
        if self.linkage == AgglomerativeLinkage.WARD:
            raise ValueError(
                "ward linkage is not supported for cannot-link constrained agglomerative "
                + "clustering; it requires Euclidean feature vectors rather than a precomputed distance matrix"
            )

    @override
    def build_processor(self) -> CannotLinkClusteringProcessor:
        algorithm = ConstrainedAgglomerativeClustering(
            linkage=self.linkage,
            distance_threshold=self.distance_threshold,
        )
        return CannotLinkAgglomerativeProcessor(
            method=ClusteringMethod.AGGLOMERATIVE,
            processor=algorithm,
        )
