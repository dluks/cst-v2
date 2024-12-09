"""Utility functions for Dask. For Dask-CUDA, see src/utils/dask_cuda_utils.py."""

import dask.dataframe as dd
import pandas as pd
from dask.distributed import TimeoutError
from distributed import Client, LocalCluster
from distributed.utils import TimeoutError

from src.conf.environment import log


def init_dask(**kwargs) -> tuple[Client, LocalCluster]:
    """Initialize the Dask client and cluster."""
    cluster = LocalCluster(**kwargs)

    client = Client(cluster)

    # Print the Dask dashboard URL
    log.info("Dask dashboard URL: %s", cluster.dashboard_link)
    return client, cluster


def close_dask(client: Client) -> None:
    """Close the Dask client and its associated cluster.

    Args:
        client: The Dask client to shut down

    Note:
        This function first attempts to close the client with a timeout.
        If that fails, it will try a force shutdown. The function will
        log any failures but will not terminate the process, allowing
        the rest of the pipeline to continue.
    """
    if client is None:
        return

    try:
        # First attempt: Graceful close with timeout
        client.close(timeout=30)
    except (TimeoutError, Exception) as e:
        log.warning(f"Initial close attempt failed: {e}. Attempting shutdown...")
        try:
            # Second attempt: Force shutdown
            client.shutdown()
        except Exception as e:
            log.warning(f"Shutdown failed: {e}. Resources may not be fully cleaned up.")


def df_to_dd(df: pd.DataFrame, npartitions: int) -> dd.DataFrame:  # pyright: ignore[reportPrivateImportUsage]
    """Convert a Pandas DataFrame to a Dask DataFrame."""
    return dd.from_pandas(  # pyright: ignore[reportPrivateImportUsage]
        df, npartitions=npartitions
    )


def repartition_if_set(df: dd.DataFrame, npartitions: int | None) -> dd.DataFrame:
    return df.repartition(npartitions=npartitions) if npartitions is not None else df
