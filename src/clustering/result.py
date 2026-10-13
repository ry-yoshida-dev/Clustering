from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast, override

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .types import BoolArray


@dataclass(frozen=True)
class ClusteringLabels:
    """
    Immutable cluster labels produced by a clustering run.

    Attributes
    ----------
    noise_label : int
        The label value used for noise / unassigned points.
    labels : NDArray[np.int64]
        The labels as a read-only 1-D int64 array owned by this instance.
    """

    noise_label: int = -1
    labels: NDArray[np.int64] = field(init=False, repr=True)

    def __init__(self, labels: ArrayLike, noise_label: int = -1) -> None:
        owned_labels = np.array(labels, dtype=np.int64).ravel()
        if owned_labels.size == 0:
            raise ValueError("labels must not be empty")
        owned_labels.setflags(write=False)
        object.__setattr__(self, "noise_label", noise_label)
        object.__setattr__(self, "labels", owned_labels)

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClusteringLabels):
            return NotImplemented
        return self.noise_label == other.noise_label and bool(
            np.array_equal(self.labels, other.labels)
        )

    @override
    def __hash__(self) -> int:
        return hash((self.noise_label, self.labels.tobytes()))

    @property
    def is_in_noise(self) -> bool:
        """
        Check whether any point carries the noise label.

        Returns
        -------
        bool
            True if at least one point is labeled as noise.
        """
        return self.num_noise > 0

    def num_clusters(
        self,
        is_noise_allowed: bool = False,
    ) -> int:
        """
        Get the number of clusters.

        Parameters
        ----------
        is_noise_allowed : bool
            Whether to count the noise label as a cluster.

        Returns
        -------
        int
            The number of clusters.
        """
        if is_noise_allowed or not self.is_in_noise:
            return len(self.set_labels)
        return len(self.set_labels) - 1

    @property
    def set_labels(self) -> list[int]:
        """
        Get the sorted unique labels.

        Returns
        -------
        list[int]
            The sorted unique labels.
        """
        return cast(list[int], np.unique(self.labels).tolist())

    @property
    def num_noise(self) -> int:
        """
        Get the number of points labeled as noise.

        Returns
        -------
        int
            The number of noise points.
        """
        return int(np.count_nonzero(np.equal(self.labels, self.noise_label)))

    @property
    def num_not_noise_labels(self) -> int:
        """
        Get the number of points not labeled as noise.

        Returns
        -------
        int
            The number of non-noise points.
        """
        return int(np.count_nonzero(np.not_equal(self.labels, self.noise_label)))

    @property
    def expanded_labels(self) -> list[int]:
        """
        Get the labels with each noise point assigned its own new label.

        Returns
        -------
        list[int]
            The labels, where noise points receive consecutive labels after the
            current maximum label.
        """
        expanded = self.labels.copy()
        if self.is_in_noise:
            max_label = int(np.max(expanded))
            expanded[self.get_mask(self.noise_label)] = np.arange(
                max_label + 1, max_label + 1 + self.num_noise, dtype=np.int64
            )
        return cast(list[int], expanded.tolist())

    def get_cluster_counts(
        self,
        is_noise_included: bool = False,
    ) -> dict[int, int]:
        """
        Get the number of points per label, largest first.

        Parameters
        ----------
        is_noise_included : bool
            Whether to include the noise label.

        Returns
        -------
        dict[int, int]
            Mapping from label to point count, ordered by descending count.
        """
        unique, counts_arr = np.unique(self.labels, return_counts=True)
        unique_list = cast(list[int], unique.tolist())
        counts_list = cast(list[int], counts_arr.tolist())
        sorted_counts = sorted(
            zip(unique_list, counts_list),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if not is_noise_included:
            sorted_counts = [
                (label, count)
                for (label, count) in sorted_counts
                if label != self.noise_label
            ]
        return dict(sorted_counts)

    def get_indices(
        self,
        label: int,
        is_empty_allowed: bool = False,
    ) -> list[int]:
        """
        Get the indices of the points carrying a label.

        Parameters
        ----------
        label : int
            The label to look up.
        is_empty_allowed : bool
            Whether to return an empty list instead of raising when the label
            does not occur.

        Returns
        -------
        list[int]
            The indices of the points carrying the label.

        Raises
        ------
        ValueError
            If the label does not occur and is_empty_allowed is False.
        """
        indices: list[int] = np.flatnonzero(self.get_mask(label)).tolist()
        if not indices and not is_empty_allowed:
            raise ValueError(
                f"label: {label} not found. If you want to allow empty indices, set is_empty_allowed to True."
            )
        return indices

    def get_labels(
        self,
        indices: list[int],
    ) -> list[int]:
        """
        Get the labels at the given indices.

        Parameters
        ----------
        indices : list[int]
            The indices to look up; each must be in the range [0, len(self)).

        Returns
        -------
        list[int]
            The labels at the given indices.

        Raises
        ------
        IndexError
            If any index is outside the range [0, len(self)).
        """
        out_of_range = [index for index in indices if not 0 <= index < len(self)]
        if out_of_range:
            raise IndexError(
                f"indices: {out_of_range} are out of range for {len(self)} labels"
            )
        return cast(list[int], self.labels[indices].tolist())

    def get_mask(
        self,
        label: int,
    ) -> BoolArray:
        """
        Get the boolean mask of the points carrying a label.

        Parameters
        ----------
        label : int
            The label to look up.

        Returns
        -------
        BoolArray
            The mask with shape (n,).
        """
        return np.equal(self.labels, label)

    def get_major_labels(
        self,
        min_count: int,
    ) -> list[int]:
        """
        Get the non-noise labels with at least a minimum number of points.

        Parameters
        ----------
        min_count : int
            The minimum point count for a label to be considered major.

        Returns
        -------
        list[int]
            The major labels, ordered by descending count.
        """
        counts = self.get_cluster_counts()
        return [label for label, count in counts.items() if count >= min_count]

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> NDArray[np.generic]:
        if dtype is None:
            return self.labels.copy() if copy else self.labels
        return self.labels.astype(dtype, copy=bool(copy))

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

        Parameters
        ----------
        index : int
            The index to look up.

        Returns
        -------
        int
            The label at the given index.
        """
        return self.labels.item(index)
