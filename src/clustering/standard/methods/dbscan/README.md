# dbscan

## Overview

Parameters for scikit-learn `DBSCAN`. Noise points are labeled `-1`; `predict` on unseen data is not supported.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `DBSCANParameters`. |
| [algorithm.py](./algorithm.py) | `DBSCANAlgorithm` enum for the nearest neighbor search backend. |
| [metric.py](./metric.py) | `DBSCANMetric` enum of supported distance metrics. |

## Examples

```python
from clustering import DBSCANParameters

proc = DBSCANParameters(eps=0.5, min_samples=5).build_processor()
result = proc.fit_predict(X)
```
