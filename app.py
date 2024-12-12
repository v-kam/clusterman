import streamlit as st
import polars as pl
import pandas as pd
from typing import Union
from src.cluster_configs import cluster_configs
from src.models import emb_small, gpt35
from src.polars_api_request import run_bulk_api_requests_chunk, run_bulk_api_requests
from src.cluster_embeddings import perform_clustering
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence


# Load environment variables
def load_environment_variables():
    load_dotenv(dotenv_path=".env")
    assert (
        os.getenv("AZURE_OPENAI_ENDPOINT") is not None
    ), "Environment variables not loaded"


# Initialize embedding model
@st.cache_resource
def get_embeddings_model():
    return emb_small()


@st.cache_resource
def get_cluster_describer() -> RunnableSequence:
    # init llm for finding descriptions for each cluster
    template = PromptTemplate(
        input_variables=["input"],
        template="""Create one description heading for the following cluster items (3-5 words total). Focus on the lowest common denominator\n{input}\description:""",
    )

    llm = gpt35()

    return template | llm | StrOutputParser()


# Generate embeddings without caching due to unhashable 'embeddings' parameter
@st.cache_data
def generate_embeddings(df: pd.DataFrame, text_column: str):
    global embeddings

    df = pl.from_pandas(df.astype(str)).pipe(
        run_bulk_api_requests_chunk,
        worker_func=embeddings.aembed_documents,
        input_col_name=text_column,
        output_col_name=f"{text_column}_embedding",
        chunk_size=10,
        rate_limit=3,
        num_workers=2,
    )
    return df


def get_clustering_config():
    with st.sidebar:
        # Select clustering algorithm
        algorithm = st.selectbox(
            "Select Clustering Algorithm",
            options=list(cluster_configs.keys()),
            format_func=lambda x: x.__name__,
        )

        # Display algorithm description
        with st.expander("Algorithm Description"):
            st.markdown(cluster_configs[algorithm]["description"])

        # Collect parameters for the selected algorithm
        params = {}
        for param, config in cluster_configs[algorithm]["params"].items():
            input_type = config["type"]
            input_args = config["args"]
            params[param] = input_type(**input_args)

        # Display selected parameters
        st.write("Selected Parameters:", params)

        # Return the clustering function and parameters
        return algorithm, params


@st.cache_data
def load_file(uploaded_file) -> Union[pd.DataFrame, None]:
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        return df
    return None


def blocking_selectbox(label, options, key=None):
    selection = st.selectbox(label, options=[""] + list(options), key=key)
    if selection == "":
        st.stop()
    return selection


load_environment_variables()
st.set_page_config(
    layout="wide",
    page_title="Clusterman",
    page_icon="static/logo.png",
)
st.logo(
    image="static/logo_wide.png",
    icon_image="static/logo.png",
)
st.title("Clusterman")
embeddings = get_embeddings_model()
cluster_describer = get_cluster_describer()
algorithm, cluster_params = get_clustering_config()


# Step 1: Upload a file
uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file to upload. This file should contain the data you want to cluster.",
    type=["csv", "xlsx"],
)
df: Union[pd.DataFrame, None] = load_file(uploaded_file)

if df is not None:
    # Step 2: View Raw Data
    with st.expander("Raw Data"):
        st.write(
            "This is the raw data from the uploaded file. You can inspect it before proceeding."
        )
        st.dataframe(df, hide_index=True)

    # Step 3: Select Column for Clustering
    text_column = blocking_selectbox(
        "Select the column you want to use for clustering. This column should contain the text data.",
        options=df.columns,
    )

    st.divider()

    # Step 4: Generate Embeddings
    with st.spinner("Generating embeddings..."):
        df_with_embeddings: pl.DataFrame = generate_embeddings(df, text_column)

    # Step 5: Perform Clustering
    with st.spinner("Clustering data..."):
        df_clustered, silhouette_score = perform_clustering(
            df_with_embeddings, text_column, algorithm, cluster_params
        )

        # Display Clustering Summary
        st.write(
            "Summary of clustering results. This table shows the items that were clustered together:"
        )

        col1, col2 = st.columns(2)
        col1.metric(
            "Silhouette Score",
            round(silhouette_score, 4) if silhouette_score is not None else "-",
            help="The silhouette score is a measure of how well the clusters are separated from each other. A higher score indicates better separation.",
        )

        col2.metric(
            "Number of Clusters",
            len(set(df_clustered[f"{text_column}_cluster_id"].to_list())),
            help="The number of clusters found by the clustering algorithm.",
        )

        df_clusters = (
            df_clustered.group_by(f"{text_column}_cluster_id")
            .agg(
                pl.col(text_column).alias("items"),
                pl.len().alias("count"),
            )
            .sort("count", descending=True)
        )
        st.dataframe(df_clusters)

    cluster_descriptions: Union[pl.DataFrame, None] = None

    # Step 6: Describe Clusters
    if st.button("Describe Clusters"):
        with st.spinner("Describing clusters..."):
            st.write(
                "Generating descriptions for each cluster. This step provides a brief summary of the items in each cluster."
            )
            cluster_descriptions = (
                df_clusters.with_columns(
                    pl.col("items")
                    .list.join(", ")
                    .str.slice(0, 300)
                    .alias("items_joined")
                )
                .pipe(
                    run_bulk_api_requests,
                    worker_func=cluster_describer.ainvoke,
                    input_col_name="items_joined",
                    output_col_name="description",
                    rate_limit=3,
                    num_workers=2,
                )
                .drop("items_joined")
            )

            # Todo: make this editable
            st.dataframe(cluster_descriptions)

            cluster_descriptions = cluster_descriptions.select(
                pl.col("description").alias(f"{text_column}_cluster_name"),
                pl.col(f"{text_column}_cluster_id"),
            )

    st.divider()

    # Step 7: Display Results
    with st.expander("Results"):
        st.write(
            "Here are the final results with cluster IDs and optional descriptions. This table shows the full dataset with clustering information."
        )
        df_clustered = df_clustered.drop(pl.col(f"{text_column}_embedding"))

        if cluster_descriptions is not None:
            df_clustered = df_clustered.join(
                cluster_descriptions, on=f"{text_column}_cluster_id", how="left"
            )

        st.dataframe(df_clustered)
