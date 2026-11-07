# Clustering

## Overview

Clustering (`clustering`) wraps scikit-learn cluster estimators, plus from-scratch methods scikit-learn lacks, behind one API: parameter dataclasses build processors with `fit` and `fit_predict`, and methods that can label unseen data also offer `predict`. `KReciprocalReranker` refines pairwise distances before clustering them as a precomputed matrix. Everything is exported from the top-level `clustering` package.

| Method | Parameters | `predict` on unseen data | Precomputed distances | Noise label `-1` |
|---|---|---|---|---|
| K-Means | `KMeansParameters` | Yes | No | No |
| Gaussian mixture | `GMMParameters` | Yes | No | No |
| Agglomerative | `AgglomerativeClusteringParameters` | No | Yes | No |
| DBSCAN | `DBSCANParameters` | No | Yes | Yes |
| HDBSCAN | `HDBSCANParameters` | No | Yes | Yes |
| Shared nearest neighbor | `SNNParameters` | No | Yes | Yes |
| Cannot-link agglomerative | `CannotLinkAgglomerativeClusteringParameters` | No | Required | No |

## Installation

```bash
pip install .
```

## Usage

```python
import numpy as np

from clustering import KMeansParameters

rng = np.random.default_rng(0)
X = np.vstack([rng.standard_normal((40, 2)), rng.standard_normal((40, 2)) + 5.0])

processor = KMeansParameters(n_clusters=2, random_state=42).build_processor()
result = processor.fit_predict(X)
print(result.labels)
```

Re-rank distances first, then cluster them as a precomputed matrix:

```python
from clustering import DBSCANMetric, DBSCANParameters, KReciprocalReranker

distances = KReciprocalReranker(n_reciprocal_neighbors=20).rerank(X)
processor = DBSCANParameters(
    eps=0.6, min_samples=4, metric=DBSCANMetric.PRECOMPUTED
).build_processor()
labels = processor.fit_predict(distances)
```
