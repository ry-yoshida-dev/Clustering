# agglomerative

## Overview

Cannot-link constrained agglomerative clustering, implemented from scratch because scikit-learn's estimators have no notion of a forbidden pair. The estimator tracks which cluster pairs are forbidden from merging as state separate from the linkage distance, rather than relying on distance-matrix tricks (which only give a hard guarantee for complete linkage; single and average linkage can still violate the constraint).

Reuses the shared `clustering.AgglomerativeLinkage` rather than declaring a separate enum; ward linkage is rejected because it requires Euclidean feature vectors rather than a precomputed distance matrix. `n_clusters` is not offered: forcing an exact cluster count can require a merge that violates a cannot-link constraint, so only `distance_threshold` is supported.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `CannotLinkAgglomerativeClusteringParameters`. |
| [estimator.py](./estimator.py) | `ConstrainedAgglomerativeClustering`: single/complete/average linkage that never merges a forbidden cluster pair. Finds merges with the nearest-neighbor-chain algorithm (O(n^2)); the chain runs to completion and every merge is cut against `distance_threshold` afterward, since the order it discovers merges in is not guaranteed to be sorted by height. |
| [processor.py](./processor.py) | `CannotLinkAgglomerativeProcessor`: wraps the estimator behind `CannotLinkClusteringProcessor`, with a post-fit check that the result never places a cannot-link pair in the same cluster. |
