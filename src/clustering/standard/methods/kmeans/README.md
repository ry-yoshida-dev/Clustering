# kmeans

## Overview

Parameters for scikit-learn `KMeans`. The resulting processor supports `predict` on unseen data.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `KMeansParameters` (`n_clusters`, `init`, `max_iter`, `tol`, `verbose`, `random_state`). |
| [initialization.py](./initialization.py) | `KMeansInitialization` enum (k-means++, random). |

## Examples

```python
from clustering import KMeansParameters

proc = KMeansParameters(n_clusters=2).build_processor()
result = proc.fit_predict(X)
```
