# cannot_link

## Overview

Cannot-link constrained clustering. Alongside the main precomputed distance matrix `X` (square, finite, non-negative, and symmetric up to floating-point tolerance), callers supply `negative_matrix`, a symmetric boolean matrix marking pairs of points that must never end up in the same cluster. `CannotLinkClusteringParameters` builds a `CannotLinkClusteringProcessor`, whose `fit`/`fit_predict` take both `X` and `negative_matrix`. There is no `predict`, since constrained clustering only labels its training data.

Only clustering methods that fit a precomputed-distance-matrix architecture are supported here. Feature-vector/centroid- or EM-based methods (K-Means, GMM) are not, since enforcing hard pairwise constraints there would require a different family of algorithms (e.g. COP-KMeans) built around iterative reassignment rather than a distance matrix.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | Abstract base class for cannot-link clustering parameters. |
| [processor.py](./processor.py) | Abstract base class for cannot-link clustering processors; implements `fit`, `fit_predict`, and `labels` caching on top of a per-method label computation (both take `X` and `negative_matrix`). |
| [inputs.py](./inputs.py) | `CannotLinkClusteringInputs`, validating `X` and `negative_matrix` before clustering. |
| [types.py](./types.py) | `NegativeMatrix` type alias. |
| [methods/](./methods/README.md) | Per-method packages, each holding its parameters, from-scratch estimator, and processor. |

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
