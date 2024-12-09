"""Utility functions for Dask. For Dask-CUDA, see src/utils/dask_cuda_utils.py."""

import os
import signal
from typing import Any

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
    return client, cluster


def close_dask(client: Client) -> None:
    """Close the Dask client and its associated cluster.
    
    Args:
        client: The Dask client to shut down
        
    Note:
        This function uses client.shutdown() to ensure a thorough cleanup of the Dask cluster,
        including all workers and resources. Uses a 30-second timeout for normal shutdown
        and 60 seconds for force shutdown.
    """
    try:
        if client is not None:
            try:
                # Use shutdown() instead of separate close() calls
                client.shutdown()
            except TimeoutError:
                log.warning("Timeout while shutting down Dask client, forcing shutdown with 60s timeout")
                # Force shutdown with a longer timeout
                client.shutdown()
            except Exception as e:
                log.warning(f"Error shutting down Dask client: {e}")
                # Try one last time to force shutdown with a longer timeout
                try:
                    client.shutdown()
                except Exception:
                    pass
    except Exception as e:
        log.error(f"Unexpected error during Dask cleanup: {e}")


def df_to_dd(
    df: pd.DataFrame, npartitions: int
) -> dd.DataFrame:  # pyright: ignore[reportPrivateImportUsage]
    """Convert a Pandas DataFrame to a Dask DataFrame."""
    return dd.from_pandas(  # pyright: ignore[reportPrivateImportUsage]
        df, npartitions=npartitions
    )
