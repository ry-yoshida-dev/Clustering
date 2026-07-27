# processors

## Overview

Concrete `ClusteringProcessor` implementations that wrap scikit-learn cluster estimators behind a unified API. Each processor pairs with a matching protocol module under [protocols/](./protocols/).

## Components

| Component | Description |
|-----------|-------------|
| [common.py](./common.py) | `CommonClusteringProcessor` for estimators with `fit` and `fit_predict` only (Agglomerative, DBSCAN, HDBSCAN). |
| [gmm.py](./gmm.py) | `GMMClusteringProcessor` wrapping `GaussianMixture` with `fit`, `predict`, and `fit_predict`. |
| [kmeans.py](./kmeans.py) | `KMeansClusteringProcessor` wrapping `KMeans` with full fit/predict support. |
| [protocols/common.py](./protocols/common.py) | `CommonClusteringLike` structural type for common processors. |
| [protocols/gmm.py](./protocols/gmm.py) | `GaussianMixtureLike` structural type for GMM. |
| [protocols/kmeans.py](./protocols/kmeans.py) | `KMeansLike` structural type for KMeans. |
