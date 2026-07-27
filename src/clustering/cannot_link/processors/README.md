# processors

## Overview

Concrete `CannotLinkClusteringProcessor` implementations wrapping the algorithms in [algorithms/](../algorithms/README.md) behind the cannot-link clustering API.

## Components

| Component | Description |
|-----------|-------------|
| [agglomerative.py](./agglomerative.py) | `CannotLinkAgglomerativeProcessor` wrapping `ConstrainedAgglomerativeClustering`, with a post-fit check that the result never places a cannot-link pair in the same cluster. |
