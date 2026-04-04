# clustering

## Overview

This module provides a unified interface for scikit-learn clustering algorithms, including K-Means, Agglomerative Clustering, and Gaussian Mixture Models (GMM). It offers a consistent API for building, configuring, and using different clustering methods with parameter validation and processor abstraction.

## Components

| Component | Description |
|-----------|-------------|
| [method.py](./method.py) | Enum defining available clustering methods (KMeans, Agglomerative, DBSCAN, HDBSCAN, GMM). |
| [parameter.py](./parameter.py) | Abstract base class for clustering parameters. |
| [processor.py](./processor.py) | Abstract base class for clustering processors with unified interface (fit, predict, fit_predict). |
| [builder.py](./builder.py) | Factory function for building clustering processors from configuration. |
| [processors/](./processors/) | Concrete implementations of clustering processors (CommonClusteringProcessor, GMMClusteringProcessor). |
| [methods/](./methods/README.md) | Parameter classes and enums for specific clustering methods (K-Means, Agglomerative, GMM). |


