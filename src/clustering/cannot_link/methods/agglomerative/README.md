# agglomerative

## Overview

Cannot-link constrained agglomerative clustering with single, complete, or average linkage, stopping at `distance_threshold`.

## Components

| Component | Description |
|-----------|-------------|
| [parameter.py](./parameter.py) | `CannotLinkAgglomerativeClusteringParameters`. |
| [estimator.py](./estimator.py) | `ConstrainedAgglomerativeClustering`, the from-scratch estimator that never merges a forbidden cluster pair. |
| [processor.py](./processor.py) | `CannotLinkAgglomerativeProcessor`, which wraps the estimator. |
