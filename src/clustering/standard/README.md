# standard

## Overview

Unconstrained clustering: parameter dataclasses build processors with `fit` and `fit_predict`, plus `predict` for methods that can label unseen data.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | Abstract base class for clustering parameters. |
| [processor.py](./processor.py) | Abstract base class for clustering processors. |
| [processors/](./processors/README.md) | Concrete processors wrapping cluster estimators. |
| [methods/](./methods/README.md) | Per-method parameter classes and enums. |

## Examples

```python
from clustering.standard import KMeansParameters

proc = KMeansParameters(n_clusters=2, random_state=42).build_processor()
result = proc.fit_predict(X)
print(result.labels)
```
