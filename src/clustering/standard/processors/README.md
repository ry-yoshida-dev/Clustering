# processors

## Overview

Concrete processors that wrap cluster estimators behind a unified API.

## Components

| Component | Description |
|-----------|-------------|
| [transductive.py](./transductive.py) | `TransductiveClusteringProcessor` for estimators that only label their training data (Agglomerative, DBSCAN, HDBSCAN, SNN). |
| [inductive.py](./inductive.py) | `InductiveClusteringProcessor` for estimators that can also `predict` unseen data (KMeans, GMM). |
| [protocols/](./protocols/README.md) | Structural types for the wrapped estimators. |
