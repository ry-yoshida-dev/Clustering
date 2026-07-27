# cannot_link

## Overview

Cannot-link constrained clustering. Alongside the main precomputed distance matrix `X`, callers supply `negative_matrix`, a symmetric boolean matrix marking pairs of points that must never end up in the same cluster. `CannotLinkClusteringParameter` builds a `CannotLinkClusteringProcessor`, whose `fit`/`fit_predict` take both `X` and `negative_matrix`.

Only clustering methods that fit a precomputed-distance-matrix architecture are supported here. Feature-vector/centroid- or EM-based methods (K-Means, GMM) are not, since enforcing hard pairwise constraints there would require a different family of algorithms (e.g. COP-KMeans) built around iterative reassignment rather than a distance matrix.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | Abstract base class for cannot-link clustering parameters. |
| [processor.py](./processor.py) | Abstract base class for cannot-link clustering processors (fit/fit_predict take `X` and `negative_matrix`). |
| [inputs.py](./inputs.py) | `CannotLinkClusteringInputs`, validating `X` and `negative_matrix` before clustering. |
| [types.py](./types.py) | `NegativeMatrix` type alias. |
| [algorithms/](./algorithms/README.md) | From-scratch constrained clustering algorithms (sklearn does not support hard pairwise constraints). |
| [processors/](./processors/README.md) | Concrete `CannotLinkClusteringProcessor` implementations wrapping the algorithms. |
| [methods/](./methods/README.md) | Parameter classes for specific cannot-link clustering methods. |

## Examples

```python
from clustering import CannotLinkAgglomerativeClusteringParameters
from clustering.standard import AgglomerativeLinkage

proc = CannotLinkAgglomerativeClusteringParameters(
    distance_threshold=0.5,
    linkage=AgglomerativeLinkage.COMPLETE,
).build_processor()
result = proc.fit_predict(X, negative_matrix)
print(result.labels)
```
