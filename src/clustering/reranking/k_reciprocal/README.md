# k_reciprocal

## Overview

k-reciprocal re-ranking (Zhong, Zheng, Cao, and Li 2017): points whose k-reciprocal neighborhoods agree move closer, and points that are only one-sidedly near move apart. The output is a symmetric distance matrix.

## Components

| Component | Description |
|-----------|-------------|
| [reranker.py](./reranker.py) | `KReciprocalReranker`, which re-ranks pairwise distances with `rerank(X)`. |
| [metric.py](./metric.py) | `KReciprocalMetric` enum of original distances (Euclidean, squared Euclidean, Manhattan, cosine, precomputed). |
| [reciprocal_neighbors.py](./reciprocal_neighbors.py) | `KReciprocalNeighbors`, the k-reciprocal neighbor sets of every point. |

## Examples

```python
from clustering import DBSCANMetric, DBSCANParameters, KReciprocalReranker

distances = KReciprocalReranker(
    n_reciprocal_neighbors=20, n_expansion_neighbors=6
).rerank(features)
labels = (
    DBSCANParameters(eps=0.6, min_samples=4, metric=DBSCANMetric.PRECOMPUTED)
    .build_processor()
    .fit_predict(distances)
)
```
