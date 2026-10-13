# agglomerative

## Overview

Parameters for scikit-learn `AgglomerativeClustering`. Use `AgglomerativeMetric.PRECOMPUTED` to cluster a square distance matrix (ward linkage requires Euclidean feature vectors).

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `AgglomerativeClusteringParameters`. |
| [linkage.py](./linkage.py) | `AgglomerativeLinkage` enum (ward, complete, average, single). |
| [metric.py](./metric.py) | `AgglomerativeMetric` enum of supported distance metrics. |

## Examples

```python
from clustering import AgglomerativeClusteringParameters

proc = AgglomerativeClusteringParameters(n_clusters=2).build_processor()
result = proc.fit_predict(X)
```
