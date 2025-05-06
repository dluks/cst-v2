""" "Match sPlot data with filtered trait data, calculate CWMs, and grid it."""

import argparse
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import repartition_if_set
from src.utils.raster_utils import xr_to_raster
from src.utils.splot_utils import filter_certain_plots


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="""
        Match sPlot data with filtered trait data, calculate CWMs, and grid it.
        """
    )
    parser.add_argument(
        "-r", "--resume", action="store_true", help="Resume from last run."
    )
    parser.add_argument(
        "-t",
        "--trait",
        type=str,
        help="Trait ID to process (e.g. '3'). If not provided, all traits will be processed.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """Match sPlot data with filtered trait data, calculate CWMs, and grid it."""
    log.info("Starting sPlot map generation...")
    sys_cfg = cfg[detect_system()][cfg.model_res]["build_splot_maps"]

    # Setup ################
    log.info("=== STAGE 1: Initial Setup ===")
    splot_dir = (
        Path(cfg.interim_dir, cfg.splot.interim.dir) / cfg.splot.interim.extracted
    )
    log.info("Using sPlot data from: %s", splot_dir)

    # Set up output directory
    out_dir = (
        Path(cfg.interim_dir, cfg.splot.interim.dir)
        / cfg.splot.interim.traits
        / cfg.PFT
        / cfg.model_res
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    # Determine which traits to process early
    log.info("=== STAGE 2: Trait Selection ===")
    valid_traits = [str(trait_num) for trait_num in cfg.datasets.Y.traits]
    log.info("Valid traits in configuration: %s", ", ".join(valid_traits))

    # If trait is specified, only process that one
    if args.trait:
        if args.trait not in valid_traits:
            raise ValueError(
                f"Invalid trait ID: {args.trait}. Valid traits are: {', '.join(valid_traits)}"
            )
        traits_to_process = [args.trait]
        log.info("Processing single trait: %s", args.trait)
    else:
        traits_to_process = valid_traits
        log.info("Processing all valid traits")

    # Check which traits actually need processing
    log.info("Checking for existing output files...")
    traits_to_process_filtered = []
    for trait in traits_to_process:
        trait_col = f"X{trait}"
        out_path = out_dir / f"{trait_col}.tif"
        if args.resume and out_path.exists():
            log.info("✓ %s.tif already exists, skipping...", trait_col)
            continue
        traits_to_process_filtered.append(trait)
        log.info("✗ %s.tif needs processing", trait_col)

    # If all traits are already processed, exit early
    if not traits_to_process_filtered:
        log.info("All requested traits already processed. Done!")
        return

    # Only proceed with data loading and processing if there are traits to process
    traits_to_process = traits_to_process_filtered
    cols = [f"X{trait}" for trait in traits_to_process]
    log.info(
        "Will process %d traits: %s",
        len(traits_to_process),
        ", ".join(traits_to_process),
    )

    def _repartition_if_set(df: dd.DataFrame, npartitions: int | None) -> dd.DataFrame:
        return (
            df.repartition(npartitions=npartitions) if npartitions is not None else df
        )

    # create dict of dask kws, but only if they are not None
    dask_kws = {k: v for k, v in sys_cfg.dask.items() if v is not None}
    log.info("=== STAGE 3: Dask Initialization ===")
    log.info("Initializing Dask client with parameters: %s", dask_kws)
    client, _ = init_dask(dashboard_address=cfg.dask_dashboard, **dask_kws)
    # /Setup ################

    try:
        # Load header and set plot IDs as index for later joining with vegetation data
        log.info("=== STAGE 4: Data Loading ===")
        log.info("Loading sPlot header data...")
        header = (
            dd.read_parquet(
                splot_dir / "header.parquet",
                columns=["PlotObservationID", "Longitude", "Latitude", "GIVD_NU"],
            )
            .astype(
                {
                    "PlotObservationID": "uint32[pyarrow]",
                    "GIVD_NU": "category",
                }
            )
            .pipe(_repartition_if_set, sys_cfg.npartitions)
            .pipe(_filter_certain_plots, "00-RU-008")
            .drop(columns=["GIVD_NU"])
            .astype({"Longitude": np.float64, "Latitude": np.float64})
            .set_index("PlotObservationID")
            .map_partitions(
                reproject_geo_to_xy, to_crs=cfg.crs, x="Longitude", y="Latitude"
            )
            .drop(columns=["Longitude", "Latitude"])
        )
        log.info("Header data loaded and processed")

        # Only load the needed trait columns
        needed_columns = ["speciesname"]
        needed_columns.extend([f"X{trait_num}" for trait_num in traits_to_process])
        log.info("Loading trait data for columns: %s", ", ".join(needed_columns))

        # Load pre-cleaned and filtered TRY traits and set species as index
        traits = (
            dd.read_parquet(
                Path(cfg.interim_dir, cfg.trydb.interim.dir)
                / cfg.trydb.interim.filtered,
                columns=needed_columns,
            )
            .pipe(_repartition_if_set, sys_cfg.npartitions)
            .set_index("speciesname")
        )
        log.info("Trait data loaded and indexed")

        # Load PFT data, filter by desired PFT, clean species names, and set them as index
        # for joining
        log.info("Loading and processing PFT data...")
        pft_path = Path(cfg.raw_dir, cfg.trydb.raw.pfts)
        if pft_path.suffix == ".csv":
            pfts = dd.read_csv(
                Path(cfg.raw_dir, cfg.trydb.raw.pfts), encoding="latin-1"
            )
        elif pft_path.suffix == ".parquet":
            pfts = dd.read_parquet(Path(cfg.raw_dir, cfg.trydb.raw.pfts))
        else:
            raise ValueError(f"Unsupported PFT file format: {pft_path.suffix}")

        pfts = (
            pfts.astype(
                {
                    "AccSpeciesName": "string[pyarrow]",
                    "pft": "category",
                }
            )
            .pipe(_repartition_if_set, sys_cfg.npartitions)
            .pipe(filter_pft, cfg.PFT)
            .drop(columns=["AccSpeciesID"])
            .dropna(subset=["AccSpeciesName"])
            .pipe(clean_species_name, "AccSpeciesName", "speciesname")
            .drop(columns=["AccSpeciesName"])
            .drop_duplicates(subset=["speciesname"])
            .set_index("speciesname")
        )
        log.info("PFT data loaded and processed")

        # Load sPlot vegetation records, clean species names, match with desired PFT, and
        # merge with trait data
        log.info("Loading and processing sPlot vegetation data...")
        merged = (
            dd.read_parquet(
                splot_dir / "vegetation.parquet",
                columns=[
                    "PlotObservationID",
                    "Species",
                    "Rel_Abund_Plot",
                ],
            )
            .astype(
                {
                    "PlotObservationID": "uint32[pyarrow]",
                    "Species": "string[pyarrow]",
                    "Rel_Abund_Plot": "float64[pyarrow]",
                }
            )
            .pipe(_repartition_if_set, sys_cfg.npartitions)
            .dropna(subset=["Species"])
            .pipe(clean_species_name, "Species", "speciesname")
            .drop(columns=["Species"])
            .set_index("speciesname")
            .join(pfts, how="inner")
            .join(traits, how="inner")
            .reset_index()
            .drop(columns=["pft", "speciesname"])
            .persist()
        )
        log.info("Data merging complete")

        log.info("=== STAGE 5: Trait Processing ===")
        for col in cols:
            log.info("Processing trait %s...", col)
            # Calculate community-weighted means per plot, join with `header` to get
            # plot lat/lons, and grid at the configured resolution.
            log.info("Calculating community-weighted statistics...")
            df = (
                merged[["PlotObservationID", "Rel_Abund_Plot", col]]
                .set_index("PlotObservationID")
                .groupby("PlotObservationID")
                .apply(
                    _cw_stats,
                    col,
                    meta={
                        "cwm": "f8",
                        "cw_std": "f8",
                        "cw_med": "f8",
                        "cw_q05": "f8",
                        "cw_q95": "f8",
                    },
                )
                .join(header, how="inner")
                .reset_index(drop=True)
                .compute()
            )
            log.info("Community-weighted statistics calculated")

            mean = {"mean": "mean"}
            mean_and_count = {"mean": "mean", "count": "count"}
            stat_cols = ["cwm", "cw_std", "cw_med", "cw_q05", "cw_q95"]
            stat_names = ["mean", "std", "median", "q05", "q95"]

            log.info("Rasterizing statistics...")
            grids = []
            for stat_col, stat_name in zip(stat_cols, stat_names):
                log.info("Rasterizing %s...", stat_col)
                funcs = mean

                # cw_95 is the last column and so we get the count now so that it will
                # be the last layer on the final trait map.
                if stat_col == "cw_q95":
                    funcs = mean_and_count

                ds = rasterize_points(
                    df,
                    data=stat_col,
                    res=cfg.target_resolution,
                    crs=cfg.crs,
                    agg=True,
                    funcs=funcs,
                )

                if stat_col != "cwm":
                    ds = ds.rename({"mean": stat_name})

                grids.append(ds)
                log.info("✓ %s rasterized", stat_col)

            # Merge all of the gridded data into a single dataset
            log.info("Merging gridded data...")
            gridded_trait = xr.merge(grids)
            log.info("Gridded data merged")

            out_fn = out_dir / f"{col}.tif"
            log.info("Writing %s to disk...", col)
            xr_to_raster(gridded_trait, out_fn)
            log.info("✓ Wrote %s", out_fn)
    finally:
        if "client" in locals():
            log.info("=== STAGE 6: Cleanup ===")
            log.info("Closing Dask client...")
            close_dask(client)
        log.info("=== COMPLETE ===")


if __name__ == "__main__":
    main()
