# snn

## Overview

Parameters for density-based shared nearest neighbor (SNN) clustering. Similarity between two points is the number of k-nearest neighbors they share, which makes the method robust to clusters of differing density and to high-dimensional data. Input is either a feature matrix or, with `SNNMetric.PRECOMPUTED`, a square distance matrix. The algorithm and `SNNMetric` live in [algorithms/](../../algorithms/README.md); this package re-exports `SNNMetric` so that methods depend on algorithms and never the reverse.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `SNNParameters` (`n_neighbors`, `min_shared_neighbors`, `min_samples`, `metric`). |

## Examples

```python
from clustering.standard.methods import SNNParameters

proc = SNNParameters(
    n_neighbors=10, min_shared_neighbors=3, min_samples=3
).build_processor()
result = proc.fit_predict(X)
print(result.labels)
```
