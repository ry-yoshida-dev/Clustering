# agglomerative

## Overview

Parameters for scikit-learn `AgglomerativeClustering`. Use `AgglomerativeMetric.PRECOMPUTED` to cluster a square distance matrix (ward linkage requires Euclidean feature vectors). The linkage enum `AgglomerativeLinkage` lives in the shared [linkage.py](../../../linkage.py) and is re-exported here.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `AgglomerativeClusteringParameters`. |
| [metric.py](./metric.py) | `AgglomerativeMetric` enum of supported distance metrics. |

## Examples

```python
from clustering import AgglomerativeClusteringParameters

proc = AgglomerativeClusteringParameters(n_clusters=2).build_processor()
result = proc.fit_predict(X)
```
