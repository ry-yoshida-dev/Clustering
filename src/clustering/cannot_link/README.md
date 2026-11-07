# cannot_link

## Overview

Cannot-link constrained clustering of a precomputed distance matrix `X`, given `negative_matrix`, a symmetric boolean matrix marking pairs of points that must never share a cluster. Processors offer `fit` and `fit_predict` but no `predict`, since constrained clustering only labels its training data.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | Abstract base class for cannot-link clustering parameters. |
| [processor.py](./processor.py) | Abstract base class for cannot-link clustering processors. |
| [inputs.py](./inputs.py) | `CannotLinkClusteringInputs`, validating `X` and `negative_matrix` before clustering. |
| [types.py](./types.py) | `NegativeMatrix` type alias. |
| [methods/](./methods/README.md) | Cannot-link clustering methods. |

## Examples

```python
from clustering import CannotLinkAgglomerativeClusteringParameters
from clustering import AgglomerativeLinkage

proc = CannotLinkAgglomerativeClusteringParameters(
    distance_threshold=0.5,
    linkage=AgglomerativeLinkage.COMPLETE,
).build_processor()
result = proc.fit_predict(X, negative_matrix)
print(result.labels)
```
