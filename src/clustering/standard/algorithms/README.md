# algorithms

## Overview

From-scratch clustering algorithms for methods that scikit-learn does not provide. Each algorithm exposes the sklearn-style `fit` / `fit_predict` / `labels_` surface so that `TransductiveClusteringProcessor` can wrap it like any sklearn estimator. Algorithms depend only on shared types, never on `methods/` or `processors/`.

## Components

| Component | Description |
|-----------|-------------|
| [snn.py](./snn.py) | `SharedNearestNeighborClustering`: density-based shared nearest neighbor clustering (Ertoz, Steinbach, and Kumar 2003). Finds k nearest neighbors (ties broken by point index), builds the shared neighbor graph, keeps strong links, and delegates labeling to `CorePointLabeling`. |
| [shared_neighbor_graph.py](./shared_neighbor_graph.py) | `SharedNeighborGraph`: mutual k-nearest-neighbor links weighted by shared neighbor count (the two endpoints are not counted), built in O(n k^2 log(n k)) time via binary search over row-sorted neighbor lists without an n-by-n matrix; candidate links are processed in batches so peak memory stays bounded. |
| [snn_metric.py](./snn_metric.py) | `SNNMetric` enum of distances used for the nearest neighbor search. |
| [core_point_labeling.py](./core_point_labeling.py) | `CorePointLabeling`: core points (enough strong links) connected by strong links form clusters (via `scipy.sparse.csgraph`); other points join the core point they share the most neighbors with, ties going to the nearer neighbor, or become noise (`-1`). |
