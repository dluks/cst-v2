"""Featurize training data."""

import gc
import shutil
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import xarray as xr
from box import ConfigBox
from distributed import Client

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import (
    check_y_set,
    get_trait_map_fns,
    get_y_fn,
    load_rasters_parallel,
)
from src.utils.trait_utils import get_trait_number_from_id


def ds_to_ddf(ds: xr.Dataset) -> dd.DataFrame:
    """Convert an xarray Dataset to a dask DataFrame"""
    return (
        ds.to_dask_dataframe()
        .drop(columns=["band", "spatial_ref"])
        .pipe(
            lambda _ddf: _ddf.dropna(
                how="all", subset=_ddf.columns.difference(["x", "y"])
            )
        )
    )


def build_y_df(
    fns: list[Path], cfg: ConfigBox, syscfg: ConfigBox, trait_set: str
) -> pd.DataFrame:
    """Build dataframe of Y data for a given trait set."""
    check_y_set(trait_set)
    log.info("Loading Y data (%s)...", trait_set)
    y_ds = load_rasters_parallel(fns, cfg.datasets.Y.trait_stat, syscfg.n_chunks)

    log.info("Computing Y data (%s)...", trait_set)
    y_ddf = ds_to_ddf(y_ds).assign(source=trait_set[0]).astype({"source": "category"})

    y_ds.close()
    return y_ddf


def main(cfg: ConfigBox = get_config()) -> None:
    """
    Main function for featurizing training data.

    Args:
        args (argparse.Namespace): Command-line arguments.
        cfg (ConfigBox): Configuration settings.

    Returns:
        None
    """
    syscfg = cfg[detect_system()][cfg.model_res]["featurize_train"]

    with Client(
        n_workers=syscfg.n_workers,
        threads_per_worker=syscfg.threads_per_worker,
        memory_limit=syscfg.memory_limit,
    ):
        log.info("Gathering trait map filenames...")
        valid_traits = [str(trait_num) for trait_num in cfg.datasets.Y.traits]

        # Process traits in batches
        batch_size = 8  # Adjust based on your system
        all_y_dfs = []

        for i in range(0, len(valid_traits), batch_size):
            batch_traits = valid_traits[i : i + batch_size]
            log.info(f"Processing trait batch {i // batch_size + 1}: {batch_traits}")

            gbif_batch_fns = [
                fn
                for fn in get_trait_map_fns("gbif", cfg)
                if get_trait_number_from_id(fn.stem) in batch_traits
            ]
            splot_batch_fns = [
                fn
                for fn in get_trait_map_fns("splot", cfg)
                if get_trait_number_from_id(fn.stem) in batch_traits
            ]

            batch_df = dd.concat(
                [
                    build_y_df(gbif_batch_fns, cfg, syscfg, "gbif"),
                    build_y_df(splot_batch_fns, cfg, syscfg, "splot"),
                ],
                axis=0,
                ignore_index=True,
            )

            # Write intermediate results
            temp_path = Path(cfg.tmp_dir) / f"y_batch_{i // batch_size}.parquet"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            batch_df.to_parquet(temp_path, compression="zstd")
            all_y_dfs.append(temp_path)

            # Force garbage collection
            gc.collect()

        # Start with the first batch
        y_df = None
        for i, parquet_path in enumerate(all_y_dfs):
            batch_df = dd.read_parquet(parquet_path)

            if y_df is None:
                y_df = batch_df
            else:
                # Merge on common spatial coordinates and source
                y_df = y_df.merge(batch_df, on=["x", "y", "source"], how="outer")

        if y_df is None:
            raise ValueError("No Y data was loaded. Check the trait map filenames.")

        out_path = get_y_fn(cfg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Writing Y data to %s...", str(out_path))
        y_df.to_parquet(out_path, compression="zstd")

        # Clean up intermediate files
        log.info("Cleaning up intermediate files...")
        for parquet_path in all_y_dfs:
            if Path(parquet_path).is_dir():
                shutil.rmtree(parquet_path, ignore_errors=True)
            else:
                Path(parquet_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
