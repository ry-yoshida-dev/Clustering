# reranking

## Overview

Distance re-ranking: transforms features or a distance matrix into a refined precomputed distance matrix before clustering. Rerankers are not clustering methods; their output is passed to any method that accepts `PRECOMPUTED` distances (DBSCAN, HDBSCAN, agglomerative, SNN) or to cannot-link clustering. K-Means and GMM require feature vectors and cannot consume the result.

## Components

| Component | Description |
|-----------|-------------|
| [k_reciprocal/](./k_reciprocal/README.md) | k-reciprocal re-ranking (Zhong et al., 2017) with a Jaccard distance over k-reciprocal neighborhoods. |
