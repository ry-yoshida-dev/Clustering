# clustering

## Overview

Unified clustering interface. `standard/` covers unconstrained clustering, `cannot_link/` clustering where marked pairs of points must not share a cluster, and `reranking/` distance refinement that feeds precomputed-distance methods.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Shared type aliases (`NumericArray`, `FloatArray`, `IntegerArray`, `BoolArray`). |
| [method.py](./method.py) | `ClusteringMethod` enum of available clustering methods. |
| [linkage.py](./linkage.py) | `AgglomerativeLinkage` enum of linkage criteria. |
| [result.py](./result.py) | `ClusteringLabels`, the shared label result type. |
| [standard/](./standard/README.md) | Unconstrained clustering. |
| [cannot_link/](./cannot_link/README.md) | Cannot-link constrained clustering. |
| [reranking/](./reranking/README.md) | Distance re-ranking into precomputed distance matrices. |
