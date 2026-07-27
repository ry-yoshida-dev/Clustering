from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .processor import CannotLinkClusteringProcessor


class CannotLinkClusteringParameter(ABC):

    @abstractmethod
    def build_processor(self) -> "CannotLinkClusteringProcessor":
        """
        Build the cannot-link clustering processor.

        Returns
        -------
        CannotLinkClusteringProcessor
            The cannot-link clustering processor.
        """
