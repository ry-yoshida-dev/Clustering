# snn

## Overview

Density-based shared nearest neighbor (SNN) clustering (Ertoz, Steinbach, and Kumar 2003), implemented from scratch because scikit-learn does not provide it. Similarity is the number of nearest neighbors two points share, which makes the method robust to clusters of differing density and to high-dimensional data.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `SNNParameters`. |
| [metric.py](./metric.py) | `SNNMetric` enum of distances used for the nearest neighbor search. |
| [estimator.py](./estimator.py) | `SharedNearestNeighborClustering`, the scikit-learn-style estimator. |
| [shared_neighbor_graph.py](./shared_neighbor_graph.py) | `SharedNeighborGraph`, mutual k-nearest-neighbor links weighted by shared neighbor count. |
| [core_point_labeling.py](./core_point_labeling.py) | `CorePointLabeling`, which labels clusters around core points and marks the rest as noise. |

## Examples

```python
from clustering import SNNParameters

proc = SNNParameters(
    n_neighbors=10, min_shared_neighbors=3, min_samples=3
).build_processor()
result = proc.fit_predict(X)
print(result.labels)
```
