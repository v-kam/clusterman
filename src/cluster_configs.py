import streamlit as st
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, HDBSCAN

cluster_configs = {
    DBSCAN: {
        "description": """
Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

A density-based clustering algorithm that groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions. It's particularly good at:
- Finding non-linearly shaped clusters
- Finding clusters with varying densities
- Identifying outliers/noise points
- Not requiring specification of number of clusters
    """,
        "params": {
            "eps": {
                "type": st.slider,
                "args": {
                    "label": "Epsilon (eps) - Maximum distance between points in a cluster (Increase for fewer clusters)",
                    "min_value": 0.0001,
                    "max_value": 1.5,
                    "value": 0.5,
                    "step": 0.001,
                    "help": "The maximum distance between two samples for one to be considered as in the neighborhood of the other.",
                },
            },
            "min_samples": {
                "type": st.slider,
                "args": {
                    "label": "Minimum Samples - Points required to form a dense region (Increase for fewer clusters)",
                    "min_value": 2,
                    "max_value": 20,
                    "value": 5,
                    "step": 1,
                    "help": "The number of samples in a neighborhood for a point to be considered as a core point.",
                },
            },
            "metric": {
                "type": st.selectbox,
                "args": {
                    "label": "Distance Metric",
                    "options": ["euclidean", "manhattan", "cosine", "l1", "l2"],
                    "index": 0,
                    "help": "Metric to use for distance computation between instances.",
                },
            },
            "algorithm": {
                "type": st.selectbox,
                "args": {
                    "label": "Algorithm",
                    "options": ["auto", "ball_tree", "kd_tree", "brute"],
                    "index": 0,
                    "help": "Algorithm to compute pointwise distances and find nearest neighbors.",
                },
            },
        },
    },
    AgglomerativeClustering: {
        "description": """
Agglomerative Hierarchical Clustering.

A hierarchical clustering algorithm that recursively merges pairs of clusters. Features include:
- Bottom-up approach starting with each point as its own cluster
- Multiple linkage criteria for cluster merging
- Can find hierarchical relationships in data
- Flexibility in choosing number of clusters or distance threshold
- No assumption about cluster shapes
        """,
        "params": {
            "n_clusters": {
                "type": st.number_input,
                "args": {
                    "label": "Number of Clusters (Increase for more clusters)",
                    "min_value": 2,
                    "max_value": 9999,
                    "value": None,
                    "step": 1,
                    "help": "The number of clusters to find. Must be None if distance_threshold is set.",
                },
            },
            "linkage": {
                "type": st.selectbox,
                "args": {
                    "label": "Linkage Criterion",
                    "options": ["ward", "complete", "average", "single"],
                    "index": 0,
                    "help": """Method for calculating distance between clusters:
                    - ward: minimizes variance within clusters
                    - complete: maximum distances between all observations
                    - average: average of distances between all observations
                    - single: minimum of distances between all observations""",
                },
            },
            "metric": {
                "type": st.selectbox,
                "args": {
                    "label": "Distance Metric",
                    "options": [
                        "euclidean",
                        "l1",
                        "l2",
                        "manhattan",
                        "cosine",
                        "precomputed",
                    ],
                    "index": 0,
                    "help": "Metric used to compute the linkage. Note: 'ward' linkage only works with 'euclidean' metric.",
                },
            },
            "distance_threshold": {
                "type": st.number_input,
                "args": {
                    "label": "Distance Threshold (Increase for fewer clusters)",
                    "min_value": 0.0,
                    "value": None,
                    "step": 0.1,
                    "help": "The linkage distance threshold above which clusters will not be merged. If set, n_clusters must be None.",
                },
            },
            "compute_distances": {
                "type": st.checkbox,
                "args": {
                    "label": "Compute Distances",
                    "value": False,
                    "help": "Whether to compute distances between clusters. Useful for dendrogram visualization.",
                },
            },
        },
    },
    KMeans: {
        "description": """
K-Means Clustering.

A popular clustering algorithm that partitions data into k clusters by:
- Iteratively assigning points to nearest cluster centers
- Updating centers as mean of assigned points
- Minimizing within-cluster variance

Key features:
- Fast and memory efficient
- Requires number of clusters to be specified
- Works well with isotropic clusters
- Assumes equal cluster size and variance
- Sensitive to outliers and initial center positions
        """,
        "params": {
            "n_clusters": {
                "type": st.number_input,
                "args": {
                    "label": "Number of Clusters (k)",
                    "min_value": 2,
                    "max_value": 9999,
                    "value": 8,
                    "step": 1,
                    "help": "The number of clusters to form and centroids to generate.",
                },
            },
            "init": {
                "type": st.selectbox,
                "args": {
                    "label": "Initialization Method",
                    "options": ["k-means++", "random"],
                    "index": 0,
                    "help": """Method for initialization:
                    - k-means++: Smart initialization that speeds up convergence
                    - random: Random selection of initial centroids""",
                },
            },
            "n_init": {
                "type": st.selectbox,
                "args": {
                    "label": "Number of Initializations",
                    "options": ["auto", "1", "3", "5", "10"],
                    "index": 0,
                    "help": "Number of times algorithm runs with different centroid seeds. Final result is the best output of n_init runs.",
                },
            },
            "max_iter": {
                "type": st.slider,
                "args": {
                    "label": "Maximum Iterations",
                    "min_value": 100,
                    "max_value": 1000,
                    "value": 300,
                    "step": 50,
                    "help": "Maximum number of iterations for a single run.",
                },
            },
            "tol": {
                "type": st.slider,
                "args": {
                    "label": "Tolerance",
                    "min_value": 0.0001,
                    "max_value": 0.01,
                    "value": 0.0001,
                    "step": 0.0001,
                    "format": "%.4f",
                    "help": "Relative tolerance with regards to Frobenius norm of the difference in cluster centers to declare convergence.",
                },
            },
            "algorithm": {
                "type": st.selectbox,
                "args": {
                    "label": "Algorithm",
                    "options": ["lloyd", "elkan"],
                    "index": 0,
                    "help": """Algorithm to use:
                    - lloyd: Classic EM-style algorithm
                    - elkan: More efficient for well-defined clusters""",
                },
            },
        },
    },
    HDBSCAN: {
        "description": """
Hierarchical Density-Based Spatial Clustering of Applications with Noise.

A density-based clustering algorithm that:
- Performs DBSCAN over varying epsilon values
- Integrates results to find optimal clustering stability
- Builds a hierarchy of clusters
- Extracts flat clustering based on cluster stability

Key features:
- Automatically determines number of clusters
- Handles clusters of varying densities and shapes
- Robust to parameter selection
- Identifies noise points
- More computationally intensive than DBSCAN
        """,
        "params": {
            "min_cluster_size": {
                "type": st.slider,
                "args": {
                    "label": "Minimum Cluster Size",
                    "min_value": 2,
                    "max_value": 100,
                    "value": 5,
                    "step": 1,
                    "help": "The minimum number of samples in a group for it to be considered a cluster.",
                },
            },
            "min_samples": {
                "type": st.slider,
                "args": {
                    "label": "Minimum Samples",
                    "min_value": 1,
                    "max_value": 100,
                    "value": None,
                    "step": 1,
                    "help": "Number of samples in a neighborhood for a point to be considered a core point. Defaults to min_cluster_size.",
                },
            },
            "cluster_selection_epsilon": {
                "type": st.number_input,
                "args": {
                    "label": "Cluster Selection Epsilon",
                    "min_value": 0.0,
                    "max_value": 1.5,
                    "value": 0.5,
                    "step": 0.05,
                    "help": "Distance threshold for cluster merging. Clusters below this value will be merged.",
                },
            },
            "metric": {
                "type": st.selectbox,
                "args": {
                    "label": "Distance Metric",
                    "options": ["euclidean", "manhattan", "cosine"],
                    "index": 0,
                    "help": "Metric to use for distance computation between instances.",
                },
            },
            "cluster_selection_method": {
                "type": st.selectbox,
                "args": {
                    "label": "Cluster Selection Method",
                    "options": ["eom", "leaf"],
                    "index": 0,
                    "help": """Method to select clusters:
                    - eom: Excess of Mass - finds most persistent clusters
                    - leaf: Selects leaf nodes - gives finest grained clustering""",
                },
            },
            "allow_single_cluster": {
                "type": st.checkbox,
                "args": {
                    "label": "Allow Single Cluster",
                    "value": False,
                    "help": "Allow the algorithm to find a single cluster if it's a valid result for the dataset.",
                },
            },
            "max_cluster_size": {
                "type": st.number_input,
                "args": {
                    "label": "Maximum Cluster Size",
                    "min_value": None,
                    "value": None,
                    "step": 1,
                    "help": "Maximum size limit for clusters. No limit if None.",
                },
            },
        },
    },
}