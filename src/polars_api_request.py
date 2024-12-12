import polars as pl
import asyncio
from more_itertools import chunked, flatten
from src.async_queue import process_tasks


def run_bulk_api_requests(
    df: pl.DataFrame,
    worker_func: callable,
    input_col_name: str,
    output_col_name: str,
    rate_limit=3,
    num_workers=3,
    **process_tasks_kwargs,
) -> pl.DataFrame:
    """
    Run bulk API requests on a given DataFrame column and store the results in a new column.

    Parameters:
    df (pl.DataFrame): The input DataFrame containing the data to process.
    worker_func (callable): The function to be applied to each element in the input column.
    input_col_name (str): The name of the column in the DataFrame to process.
    output_col_name (str): The name of the column to store the results.
    rate_limit (int, optional): The rate limit for API requests. Defaults to 3.
    num_workers (int, optional): The number of worker tasks to run concurrently. Defaults to 3.

    Returns:
    pl.DataFrame: The DataFrame with the new column containing the results of the API requests.
    """

    result = asyncio.run(
        process_tasks(
            tasks=df[input_col_name],
            worker_func=worker_func,
            rate_limit=rate_limit,  # sonnet: 1.2 (100/60), haiku: 6.2 (400/60)
            num_workers=num_workers,
            **process_tasks_kwargs,
        )
    )

    return df.with_columns(pl.Series(result, strict=False).alias(output_col_name))


def run_bulk_api_requests_chunk(
    df: pl.DataFrame,
    worker_func: callable,
    input_col_name: str,
    output_col_name: str,
    chunk_size: int,
    rate_limit=3,
    num_workers=3,
    **process_tasks_kwargs,
) -> pl.DataFrame:
    """
    Run bulk API requests on a given DataFrame column in chunks and store the results in a new column.
    Used for example, if you have a model that works better with batches than with individual elements.

    Parameters:
    df (pl.DataFrame): The input DataFrame containing the data to process.
    worker_func (callable): The function to be applied to each element in the input column.
    input_col_name (str): The name of the column in the DataFrame to process.
    output_col_name (str): The name of the column to store the results.
    chunk_size (int): The size of each chunk to split the input data into.
    rate_limit (int, optional): The rate limit for API requests. Defaults to 3.
    num_workers (int, optional): The number of worker tasks to run concurrently. Defaults to 3.

    Returns:
    pl.DataFrame: The DataFrame with the new column containing the results of the API requests.
    """

    input_series = df[input_col_name]
    chunks = list(chunked(input_series, chunk_size))

    result = asyncio.run(
        process_tasks(
            tasks=chunks,
            worker_func=worker_func,
            rate_limit=rate_limit,
            num_workers=num_workers,
            **process_tasks_kwargs,
        )
    )

    flattened_result = list(flatten(result))

    return df.with_columns(
        pl.Series(flattened_result, strict=False).alias(output_col_name)
    )
