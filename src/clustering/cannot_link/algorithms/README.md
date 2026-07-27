# algorithms

## Overview

From-scratch clustering algorithms with hard cannot-link constraints. scikit-learn's estimators have no notion of a forbidden pair, so these algorithms track which cluster pairs are forbidden from merging as state separate from the linkage distance, rather than relying on distance-matrix tricks (which only give a hard guarantee for complete linkage; single and average linkage can still violate the constraint).

## Components

| Component | Description |
|-----------|-------------|
| [agglomerative.py](./agglomerative.py) | `ConstrainedAgglomerativeClustering`: single/complete/average-linkage agglomerative clustering that never merges a forbidden cluster pair. Resolves the linkage enum to a merge-row function once, in `__post_init__`, so the merge hot loop never branches on it. Finds merges with the nearest-neighbor-chain algorithm (O(n^2)) instead of rescanning the full active-by-active distance submatrix on every merge (O(n^3)); the chain runs to completion and every merge it finds is cut against `distance_threshold` afterward, since the order it discovers merges in is not guaranteed to be sorted by height. |
