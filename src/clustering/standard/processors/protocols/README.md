# processors/protocols

## Overview

Structural typing protocols for wrapped cluster estimators. Each file mirrors the corresponding processor module and provides stable method signatures where sklearn type stubs are incomplete.

## Components

| Component | Description |
|-----------|-------------|
| [transductive.py](./transductive.py) | `TransductiveClusteringLike` for AgglomerativeClustering, DBSCAN, HDBSCAN, and SNN. |
| [inductive.py](./inductive.py) | `InductiveClusteringLike` for KMeans and GaussianMixture. |
