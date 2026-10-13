# gmm

## Overview

Parameters for scikit-learn `GaussianMixture`. The resulting processor supports `predict` on unseen data.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `GMMParameters` (`n_components`, `covariance_type`, `random_state`). |
| [covariance_type.py](./covariance_type.py) | `GMMCovarianceType` enum. |

## Examples

```python
from clustering import GMMParameters

proc = GMMParameters(n_components=2).build_processor()
result = proc.fit_predict(X)
```
