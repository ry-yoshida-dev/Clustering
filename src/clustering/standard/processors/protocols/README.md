# processors/protocols

## Overview

Structural types for the wrapped cluster estimators, one per processor module.

## Components

| Component | Description |
|-----------|-------------|
| [transductive.py](./transductive.py) | `TransductiveClusteringLike` for AgglomerativeClustering, DBSCAN, HDBSCAN, and SNN. |
| [inductive.py](./inductive.py) | `InductiveClusteringLike` for KMeans and GaussianMixture. |
