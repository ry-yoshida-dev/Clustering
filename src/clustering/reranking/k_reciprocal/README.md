# k_reciprocal

## Overview

k-reciprocal re-ranking (Zhong, Zheng, Cao, and Li 2017), implemented with vectorized numpy over all points. Each point's k1-reciprocal neighbor set is expanded with overlapping round(k1 / 2)-reciprocal sets, encoded as Gaussian-weighted vectors, averaged over its k2 nearest points, and compared by Jaccard distance; the result is blended with the original distance by `original_distance_weight` (default 0, pure Jaccard as is common for clustering; the reference retrieval setting is 0.3). Distances are scaled by their global maximum (the reference implementation scales per column), so the output is symmetric and can be passed to cannot-link clustering. Time is O(n^2 log n + n^2 s), where s is the support size of a neighborhood vector after query expansion (roughly k1 * k2), and memory is O(n^2).

This overlaps conceptually with SNN clustering, which also compares neighbor lists; re-ranking differs in producing a reusable distance matrix with weighted, reciprocity-filtered neighborhoods, so it can be combined with any precomputed-distance method.

## Components

| Component | Description |
|-----------|-------------|
| [reranker.py](./reranker.py) | `KReciprocalReranker` (`n_reciprocal_neighbors` = k1, `n_expansion_neighbors` = k2, `original_distance_weight` = lambda, `metric`) with `rerank(X)`. |
| [metric.py](./metric.py) | `KReciprocalMetric` enum of original distances (Euclidean, squared Euclidean, Manhattan, cosine, precomputed). |
| [reciprocal_neighbors.py](./reciprocal_neighbors.py) | `KReciprocalNeighbors`: k-reciprocal sets of every point as padded index lists, derived from a full distance ranking. |

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
