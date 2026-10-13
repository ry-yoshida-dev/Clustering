# clustering

## Overview

This module provides a unified interface for clustering. `standard/` wraps scikit-learn clustering algorithms (K-Means, Agglomerative Clustering, DBSCAN, HDBSCAN, and Gaussian Mixture Models) and adds shared nearest neighbor (SNN) clustering. `cannot_link/` adds cannot-link constrained clustering, where a boolean pair matrix marks points that must not end up in the same cluster; only methods that fit a precomputed-distance-matrix architecture are supported there. `method.py`, `linkage.py`, `result.py`, and `types.py` at this level are shared by both.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Shared type aliases (`NumericArray`, `FloatArray`, `IntegerArray`, `BoolArray`). |
| [method.py](./method.py) | Enum defining available clustering methods (KMeans, Agglomerative, DBSCAN, HDBSCAN, GMM, SNN), shared by `standard/` and `cannot_link/`. |
| [linkage.py](./linkage.py) | `AgglomerativeLinkage` enum, shared by standard and cannot-link agglomerative clustering. |
| [result.py](./result.py) | `ClusteringLabels`, the shared label result type. |
| [standard/](./standard/README.md) | Unconstrained clustering: parameter classes, processors, and sklearn structural typing protocols. |
| [cannot_link/](./cannot_link/README.md) | Cannot-link constrained clustering: parameter classes, processors, and the underlying algorithms. |
