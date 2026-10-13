# Clustering

## Overview

Clustering (`clustering`) is a Python package that wraps scikit-learn cluster estimators behind a small, consistent API: parameter dataclasses build `ClusteringProcessor` instances with `fit` and `fit_predict`; processors for methods that can label unseen data also offer `predict`.

Supported method families include K-Means, agglomerative clustering, DBSCAN, HDBSCAN, Gaussian mixture models, and shared nearest neighbor (SNN) clustering. All parameter classes and enums are exported from the top-level `clustering` package.

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

Density-based methods label noise points with `-1`:

```python
from clustering import DBSCANParameters

proc = DBSCANParameters(eps=0.5, min_samples=5).build_processor()
labels = proc.fit_predict(X)
```

## Notes

- `fit_predict` and `labels` return `ClusteringLabels`, an immutable wrapper around a read-only 1-D `numpy.int64` vector (`ClusteringLabels.labels`).
- `labels` returns the training labels of the last `fit` / `fit_predict`; reading it before fitting raises `sklearn.exceptions.NotFittedError`.
- Only K-Means and GMM support `predict` on unseen data: their parameters build an `InductiveClusteringProcessor`. Agglomerative, DBSCAN, HDBSCAN, SNN, and cannot-link processors have no `predict` method at all, so calling it is a type error; use `fit_predict` on the training data.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run basedpyright src tests
uv run pytest
```

The same checks run in CI on every push to `main` and every pull request.
