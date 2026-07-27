# standard

## Overview

Unconstrained clustering: parameter dataclasses build `ClusteringProcessor` instances with `fit`, `fit_predict`, and (where supported) `predict`, wrapping scikit-learn estimators (K-Means, Agglomerative Clustering, DBSCAN, HDBSCAN, GMM).

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | Abstract base class for clustering parameters. |
| [processor.py](./processor.py) | Abstract base class for clustering processors with unified interface (fit, predict, fit_predict). |
| [processors/](./processors/README.md) | Concrete implementations of clustering processors and sklearn structural typing protocols. |
| [methods/](./methods/README.md) | Parameter classes and enums for specific clustering methods (K-Means, Agglomerative, DBSCAN, HDBSCAN, GMM). |

## Examples

```python
from clustering.standard import KMeansParameters

proc = KMeansParameters(n_clusters=2, random_state=42).build_processor()
result = proc.fit_predict(X)
print(result.labels)
```
