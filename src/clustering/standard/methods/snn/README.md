# snn

## Overview

Density-based shared nearest neighbor (SNN) clustering (Ertoz, Steinbach, and Kumar 2003), implemented from scratch because scikit-learn does not provide it. Similarity between two points is the number of k-nearest neighbors they share, which makes the method robust to clusters of differing density and to high-dimensional data. Input is either a feature matrix or, with `SNNMetric.PRECOMPUTED`, a square distance matrix. The estimator exposes the sklearn-style `fit` / `fit_predict` / `labels_` surface so that `TransductiveClusteringProcessor` wraps it like any sklearn estimator.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `SNNParameters` (`n_neighbors`, `min_shared_neighbors`, `min_samples`, `metric`). |
| [metric.py](./metric.py) | `SNNMetric` enum of distances used for the nearest neighbor search. |
| [estimator.py](./estimator.py) | `SharedNearestNeighborClustering`: finds k nearest neighbors (ties broken by point index), builds the shared neighbor graph, keeps strong links, and delegates labeling to `CorePointLabeling`. |
| [shared_neighbor_graph.py](./shared_neighbor_graph.py) | `SharedNeighborGraph`: mutual k-nearest-neighbor links weighted by shared neighbor count (the two endpoints are not counted), built in O(n k^2 log(n k)) time via binary search over row-sorted neighbor lists without an n-by-n matrix; candidate links are processed in batches so peak memory stays bounded. |
| [core_point_labeling.py](./core_point_labeling.py) | `CorePointLabeling`: core points (enough strong links) connected by strong links form clusters (via `scipy.sparse.csgraph`); other points join the core point they share the most neighbors with, ties going to the nearer neighbor, or become noise (`-1`). |

## Examples

```python
from clustering import SNNParameters

proc = SNNParameters(
    n_neighbors=10, min_shared_neighbors=3, min_samples=3
).build_processor()
result = proc.fit_predict(X)
print(result.labels)
```
