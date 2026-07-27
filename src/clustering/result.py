from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from typing_extensions import override

from .types import BoolArray


@dataclass(frozen=True)
class ClusteringLabels:
    """
    ClusteringLabels is the labels of the clustering result.

    Attributes:
    ----------
    labels: NDArray[np.int64]
        The labels as a 1-D int64 array (owning buffer; safe to use in NumPy ops).
    noise_label: int
        The label value used for noise / unassigned points.
    """

    noise_label: int = -1
    labels: NDArray[np.int64] = field(init=False, repr=True)

    def __init__(self, labels: ArrayLike, noise_label: int = -1) -> None:
        raw: object = labels
        if isinstance(raw, tuple):
            raw = cast(object, raw[0])
        arr = np.asarray(raw, dtype=np.int64).ravel()
        if arr.size == 0:
            raise ValueError("labels must not be empty")
        object.__setattr__(self, "noise_label", noise_label)
        object.__setattr__(self, "labels", arr.copy())

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClusteringLabels):
            return NotImplemented
        return self.noise_label == other.noise_label and bool(
            np.array_equal(self.labels, other.labels)
        )

    @property
    def is_in_noise(self) -> bool:
        """
        Check if the label is in the noise.

        Returns:
        ----------
        bool
            True if the label is in the noise, False otherwise.
        """
        return self.num_noise > 0

    def num_clusters(
        self,
        is_noise_allowed: bool = False,
    ) -> int:
        """
        Get the number of clusters.

        Parameters:
        ----------
        is_noise_allowed: bool
            Whether to allow noise labels in the number of clusters calculation.

        Returns:
        ----------
        int
            The number of clusters.
        """
        if is_noise_allowed:
            return len(self.set_labels)
        if self.is_in_noise:
            return len(self.set_labels) - 1
        return len(self.set_labels)

    @property
    def set_labels(self) -> list[int]:
        """
        Get the set of labels.

        Returns:
        ----------
        list[int]
            The set of labels.
        """
        return cast(list[int], np.unique(self.labels).tolist())

    @property
    def num_noise(self) -> int:
        """
        Get the number of noise labels.

        Returns:
        ----------
        int
            The number of noise labels.
        """
        return int(np.count_nonzero(np.equal(self.labels, self.noise_label)))

    @property
    def num_not_noise_labels(self) -> int:
        """
        Get the number of not noise labels.

        Returns:
        ----------
        int
            The number of not noise labels.
        """
        return int(np.count_nonzero(np.not_equal(self.labels, self.noise_label)))

    @property
    def expanded_labels(self) -> list[int]:
        """
        Get the expanded labels.
        Noise labels are assigned the next available label.

        Returns:
        ----------
        list[int]
            The unique labels.
        """
        new_labels = self.labels.copy()
        noise_mask = self.get_mask(self.noise_label)
        if self.num_noise > 0:
            max_label = int(np.max(new_labels))
            new_labels[noise_mask] = np.arange(
                max_label + 1, max_label + 1 + self.num_noise, dtype=np.int64
            )
        return cast(list[int], new_labels.tolist())

    def get_cluster_counts(
        self,
        include_noise: bool = False,
    ) -> dict[int, int]:
        """
        Get the cluster counts.

        Parameters:
        ----------
        include_noise: bool
            Whether to include noise labels.

        Returns:
        ----------
        dict[int, int]
            The dictionary of cluster counts.
            The keys are the labels and the values are the counts.
        """
        unique, counts_arr = np.unique(self.labels, return_counts=True)
        unique_list = cast(list[int], unique.tolist())
        counts_list = cast(list[int], counts_arr.tolist())
        sorted_counts = sorted(
            zip(unique_list, counts_list),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if not include_noise:
            sorted_counts = [
                (label, count) for (label, count) in sorted_counts if label != self.noise_label
            ]
        return dict(sorted_counts)

    def get_indices(
        self,
        label: int,
        is_empty_allowed: bool = False,
    ) -> list[int]:
        """
        Get the indices of the label.

        Parameters:
        ----------
        label: int
            The label to get the indices of.
        is_empty_allowed: bool
            Whether to allow empty indices.
        """
        indices = np.where(np.equal(self.labels, label))[0]
        if len(indices) == 0 and not is_empty_allowed:
            raise ValueError(
                f"label: {label} not found. If you want to allow empty indices, set is_empty_allowed to True."
            )
        return indices.tolist()

    def get_labels(
        self,
        indices: list[int],
    ) -> list[int]:
        """
        Get the labels of the indices.

        Parameters:
        ----------
        indices: list[int]
            The indices to get the labels of.
        """
        max_index = max(indices)
        if max_index >= len(self):
            raise ValueError(f"index: {max_index} is out of range")
        labels_list = cast(list[int], self.labels.tolist())
        return [labels_list[i] for i in indices]

    def get_mask(
        self,
        label: int,
    ) -> BoolArray:
        """
        Get the mask of the label.

        Parameters:
        ----------
        label: int
            The label to get the mask of.

        Returns:
        ----------
        BoolArray
            The mask of the label with shape (n,).
        """
        return np.equal(self.labels, label)

    def get_major_labels(
        self,
        min_count: int,
    ) -> list[int]:
        """
        Get the major labels.

        Parameters:
        ----------
        min_count: int
            The minimum count of the label to be considered as a major label.
        """
        counts = self.get_cluster_counts()
        major_labels = [label for label, count in counts.items() if count >= min_count]
        return major_labels

    def __array__(self, dtype: DTypeLike | None = None) -> NDArray[np.generic]:
        if dtype is None:
            return self.labels
        return self.labels.astype(dtype, copy=False)

    def __iter__(self) -> Iterator[int]:
        return iter(cast(list[int], self.labels.tolist()))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        index: int,
    ) -> int:
        """
        Get the label at the given index.

        Parameters:
        ----------
        index: int
            The index to get the label at.

        Returns:
        ----------
        int
            The label at the given index.
        """
        return cast(list[int], self.labels.tolist())[index]
