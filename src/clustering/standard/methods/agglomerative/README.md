# agglomerative

## Overview

Parameters for scikit-learn `AgglomerativeClustering`. Exactly one stopping criterion must be set: `n_clusters` or `distance_threshold` (both default to `None`). The default metric is `AgglomerativeMetric.PRECOMPUTED`, which clusters a square distance matrix; pass `AgglomerativeMetric.EUCLIDEAN` for feature vectors (required by ward linkage). The linkage enum `AgglomerativeLinkage` lives in the shared [linkage.py](../../../linkage.py) and is re-exported here.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `AgglomerativeClusteringParameters`. |
| [metric.py](./metric.py) | `AgglomerativeMetric` enum of supported distance metrics. |

## Examples

```python
from clustering import (
    AgglomerativeClusteringParameters,
    AgglomerativeLinkage,
    AgglomerativeMetric,
)

proc = AgglomerativeClusteringParameters(
    n_clusters=2,
    metric=AgglomerativeMetric.EUCLIDEAN,
    linkage=AgglomerativeLinkage.WARD,
).build_processor()
result = proc.fit_predict(X)

by_distance = AgglomerativeClusteringParameters(
    distance_threshold=0.3
).build_processor()
result = by_distance.fit_predict(distance_matrix)
```
