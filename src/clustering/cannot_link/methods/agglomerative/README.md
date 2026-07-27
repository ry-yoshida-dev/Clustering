# agglomerative

## Overview

Parameters for cannot-link constrained agglomerative clustering. Reuses `clustering.standard.methods.agglomerative.AgglomerativeLinkage` rather than declaring a separate enum; ward linkage is rejected because it requires Euclidean feature vectors rather than a precomputed distance matrix. `n_clusters` is not offered: forcing an exact cluster count can require a merge that violates a cannot-link constraint, so only `distance_threshold` is supported.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `CannotLinkAgglomerativeClusteringParameters`. |
