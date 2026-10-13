from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .processor import ClusteringProcessor


class ClusteringParameters(ABC):
    """
    Base class for clustering parameters that build a processor.
    """

    @abstractmethod
    def build_processor(self) -> ClusteringProcessor:
        """
        Build the clustering processor.

        Returns
        -------
        ClusteringProcessor
            The clustering processor.
        """
