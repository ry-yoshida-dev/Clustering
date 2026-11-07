# Tests

## Overview

Unit tests for the `clustering` package, one file per method or utility.

## Components

| Component | Description |
| --- | --- |
| [`helpers.py`](helpers.py) | Shared synthetic datasets for clustering tests. |
| [`test_clustering_labels.py`](test_clustering_labels.py) | Tests for `ClusteringLabels` helpers and invariants. |
| [`test_kmeans_processor.py`](test_kmeans_processor.py) | Tests for KMeans parameters and processor API. |
| [`test_gmm_processor.py`](test_gmm_processor.py) | Tests for GMM parameters and processor API. |
| [`test_dbscan_processor.py`](test_dbscan_processor.py) | Tests for DBSCAN parameters and processor API. |
| [`test_hdbscan_processor.py`](test_hdbscan_processor.py) | Tests for HDBSCAN parameters and processor API. |
| [`test_agglomerative_processor.py`](test_agglomerative_processor.py) | Tests for agglomerative parameters and processor API. |
| [`test_snn_processor.py`](test_snn_processor.py) | Tests for shared nearest neighbor parameters, graph construction, and processor API. |
| [`test_cannot_link_agglomerative_processor.py`](test_cannot_link_agglomerative_processor.py) | Tests for cannot-link constrained agglomerative parameters and processor API. |
| [`test_k_reciprocal_reranker.py`](test_k_reciprocal_reranker.py) | Tests for k-reciprocal re-ranking. |

