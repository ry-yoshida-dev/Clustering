# methods

## Overview

Parameter classes and enums for each supported clustering method. Every parameter class validates its fields on construction and builds a `ClusteringProcessor` via `build_processor()`.

## Components

| Component | Description |
|-----------|-------------|
| [kmeans/](./kmeans/README.md) | K-Means parameters and initialization methods. |
| [agglomerative/](./agglomerative/README.md) | Agglomerative clustering parameters, linkage methods, and distance metrics. |
| [dbscan/](./dbscan/README.md) | DBSCAN parameters, search algorithms, and distance metrics. |
| [hdbscan/](./hdbscan/README.md) | HDBSCAN parameters and related enums. |
| [gmm/](./gmm/README.md) | Gaussian Mixture Model parameters and covariance types. |
| [snn/](./snn/README.md) | Shared nearest neighbor clustering parameters and distance metrics. |
