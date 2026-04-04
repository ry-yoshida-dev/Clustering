from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .processor import ClusteringProcessor

class ClusteringParameter(ABC):

    @abstractmethod
    def build_processor(self) -> 'ClusteringProcessor':
        """
        Build the clustering processor.

        Returns:
        --------
        ClusteringProcessor: The clustering processor.
        """