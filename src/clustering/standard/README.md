# standard

## Overview

Unconstrained clustering: parameter dataclasses build `ClusteringProcessor` instances with `fit` and `fit_predict` (plus `predict` on `InductiveClusteringProcessor` only), wrapping scikit-learn estimators (K-Means, Agglomerative Clustering, DBSCAN, HDBSCAN, GMM) and from-scratch algorithms for methods sklearn lacks (shared nearest neighbor clustering).

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | Abstract base class for clustering parameters. |
| [processor.py](./processor.py) | Abstract base class for clustering processors; implements `fit`, `fit_predict`, and `labels` caching on top of a per-estimator label computation. |
| [processors/](./processors/README.md) | Concrete processors (`TransductiveClusteringProcessor`, `InductiveClusteringProcessor`) and structural typing protocols. |
| [methods/](./methods/README.md) | Parameter classes and enums for specific clustering methods (K-Means, Agglomerative, DBSCAN, HDBSCAN, GMM, SNN). |
| [algorithms/](./algorithms/README.md) | From-scratch clustering algorithms not provided by sklearn (shared nearest neighbor clustering). |

## Examples

```python
from clustering.standard import KMeansParameters

proc = KMeansParameters(n_clusters=2, random_state=42).build_processor()
result = proc.fit_predict(X)
print(result.labels)
```
