# hdbscan

## Overview

Parameters for scikit-learn `HDBSCAN`. Noise points are labeled `-1`; `predict` on unseen data is not supported.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `HDBSCANParameters`. |
| [algorithm.py](./algorithm.py) | `HDBSCANAlgorithm` enum for the nearest neighbor search backend. |
| [cluster_selection_method.py](./cluster_selection_method.py) | `HDBSCANClusterSelectionMethod` enum (EOM, leaf). |
| [metric.py](./metric.py) | `HDBSCANMetric` enum of supported distance metrics. |
| [store_centers.py](./store_centers.py) | `HDBSCANStoreCenters` enum selecting which cluster centers to store. |

## Examples

```python
from clustering import HDBSCANParameters

proc = HDBSCANParameters(min_cluster_size=5).build_processor()
result = proc.fit_predict(X)
```
