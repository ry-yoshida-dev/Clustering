# processors/protocols

## Overview

Structural typing protocols for scikit-learn cluster estimators. Each file mirrors the corresponding processor module and provides stable method signatures where sklearn type stubs are incomplete.

## Components

| Component | Description |
|-----------|-------------|
| [common.py](./common.py) | `CommonClusteringLike` for AgglomerativeClustering, DBSCAN, and HDBSCAN. |
| [gmm.py](./gmm.py) | `GaussianMixtureLike` for sklearn GaussianMixture. |
| [kmeans.py](./kmeans.py) | `KMeansLike` for sklearn KMeans. |
