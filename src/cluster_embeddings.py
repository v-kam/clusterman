import polars as pl
import numpy as np
from typing import Any, Union, Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def cluster_embeddings(
    df: pl.DataFrame,
    embeddings_col: str,
    output_col: str,
    clusterer: Union[AgglomerativeClustering, Any],
    cluster_kwargs: dict,
):
    """
    Clusters embeddings in a DataFrame using the specified clustering algorithm.

    Parameters:
    - df (pl.DataFrame): The DataFrame containing the embeddings.
    - embeddings_col (str): The name of the column containing the embeddings.
    - output_col (str): The name of the column to store the cluster labels.
    - clusterer (Union[AgglomerativeClustering, Any]): The clustering algorithm to use.
    - cluster_kwargs (dict): Additional keyword arguments for the clustering algorithm.

    Returns:
    - pl.DataFrame: The original DataFrame with an additional column for cluster labels.
    """
    vectors = np.vstack(df[embeddings_col].to_list())

    clusterer = clusterer(**cluster_kwargs)
    cluster_labels = clusterer.fit_predict(vectors)

    return df.with_columns(pl.Series("cluster", cluster_labels).alias(output_col))


def perform_clustering(
    df_with_embeddings, text_column, algorithm, cluster_params
) -> Tuple[pl.DataFrame, float]:
    """
    Performs clustering on a DataFrame with embeddings and calculates the silhouette score.

    Parameters:
    - df_with_embeddings (pl.DataFrame): The DataFrame containing the embeddings.
    - text_column (str): The name of the text column used for generating embeddings.
    - algorithm: The clustering algorithm to use.
    - cluster_params (dict): Parameters for the clustering algorithm.

    Returns:
    - Tuple[pl.DataFrame, float]: A tuple containing the clustered DataFrame and the silhouette score.
    """
    df_clustered = df_with_embeddings.pipe(
        cluster_embeddings,
        embeddings_col=f"{text_column}_embedding",
        output_col=f"{text_column}_cluster_id",
        clusterer=algorithm,
        cluster_kwargs=cluster_params,
    )

    vectors = np.vstack(df_with_embeddings[f"{text_column}_embedding"].to_list())
    labels = df_clustered[f"{text_column}_cluster_id"].to_list()

    silhouette_avg = silhouette_score(vectors, labels) if len(set(labels)) > 1 else None

    return df_clustered, silhouette_avg
