# Clustering

## Overview

Clustering (`clustering`) is a Python package that wraps scikit-learn cluster estimators behind a small, consistent API: parameter dataclasses build `ClusteringProcessor` instances with `fit`, `fit_predict`, and (where supported) `predict`.

Supported method families include K-Means, agglomerative clustering, DBSCAN, HDBSCAN, and Gaussian mixture models. Method-specific options live under `clustering.methods`.

For module-level design notes, see [src/clustering/README.md](src/clustering/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development:

```bash
pip install -e .
```

If you only need dependencies:

```bash
pip install -r requirements.txt
```

## Quick example

```python
import numpy as np

from clustering import KMeansParameters

rng = np.random.default_rng(0)
X = np.vstack([rng.standard_normal((40, 2)), rng.standard_normal((40, 2)) + 5.0])

proc = KMeansParameters(n_clusters=2, random_state=42).build_processor()
result = proc.fit_predict(X)
print(result.labels)
```

DBSCAN and related parameter types are imported from `clustering.methods`:

```python
from clustering.methods import DBSCANParameters

proc = DBSCANParameters(eps=0.5, min_samples=5).build_processor()
labels = proc.fit_predict(X)
```

## Notes

- `fit_predict` and `labels` return `ClusteringLabels`, which wraps a 1-D `numpy.int64` vector (`ClusteringLabels.labels`).
- Estimators that do not support `predict` in sklearn (for example DBSCAN / HDBSCAN in this wrapper) will not implement meaningful `predict` on the processor either; use `fit_predict` on the training data.
