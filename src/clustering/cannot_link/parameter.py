from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .processor import CannotLinkClusteringProcessor


class CannotLinkClusteringParameters(ABC):
    """
    Base class for cannot-link clustering parameters that build a processor.

    Only methods that cluster a precomputed distance matrix are supported.
    Centroid- or EM-based methods such as K-Means and GMM are not: enforcing
    hard pairwise constraints there needs a different family of algorithms
    (e.g. COP-KMeans) built around iterative reassignment.
    """

    @abstractmethod
    def build_processor(self) -> "CannotLinkClusteringProcessor":
        """
        Build the cannot-link clustering processor.

        Returns
        -------
        CannotLinkClusteringProcessor
            The cannot-link clustering processor.
        """
