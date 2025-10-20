""" "Match sPlot data with filtered trait data, calculate CWMs, and grid it."""

import argparse
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from box import ConfigBox
from dask.distributed import Client

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import repartition_if_set
from src.utils.dataset_utils import load_pfts
from src.utils.df_utils import pipe_log, rasterize_points, reproject_geo_to_xy
from src.utils.raster_utils import xr_to_raster
from src.utils.splot_utils import filter_certain_plots
from src.utils.trait_aggregation_utils import cw_stats, fd_metrics
from src.utils.trait_utils import (
    check_for_existing_maps,
    clean_species_name,
    filter_pft,
    get_traits_to_process,
    load_try_traits,
)

# TODO: Allow specification of PCA component(s) to use


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
        type=int,
        help="""
        Trait ID to process (e.g. '3'). If not provided, all traits will be processed.
        """,
    )
    parser.add_argument(
        "-f",
        "--fd-metric",
        type=str,
        help="""
        FD metric to process (e.g. 'f_ric'). If not provided, all FD metrics will be
        processed.
        """,
    )
    return parser.parse_args()


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """Match sPlot data with filtered trait data, calculate CWMs, and grid it."""
    log.info("Starting sPlot map generation...")
    sys_cfg = cfg[detect_system()][cfg.model_res]["build_splot_maps"]

    # Setup ################
    splot_dir, out_dir = _setup_in_and_out_dirs(cfg)

    # Determine which traits to process early
    trait_stats = cfg.datasets.Y.trait_stats
    fd_metrics = ["f_ric", "f_eve", "f_div", "f_red", "sp_ric", "f_ric_ses"]
    fd_mode = any(stat in fd_metrics for stat in trait_stats)
    using_pca = cfg.trydb.interim.perform_pca

    if args.trait and using_pca:
        raise ValueError("Trait ID not supported when using PCA.")

    if args.fd_metric and args.fd_metric not in fd_metrics:
        raise ValueError(
            f"Invalid FD metric: {args.fd_metric}. "
            f"Valid metrics are: {', '.join(fd_metrics)}"
        )

    if args.fd_metric and not fd_mode:
        raise ValueError("FD metric not supported when not in FD mode.")

    fd_stats_to_process = []
    if args.fd_metric:
        fd_stats_to_process = [args.fd_metric]
    elif fd_mode:
        fd_stats_to_process = trait_stats

    traits_to_process = get_traits_to_process(
        cfg.datasets.Y.traits, using_pca, args.trait
    )

    if args.resume:
        traits_to_process, fd_stats_to_process = check_for_existing_maps(
            out_dir, traits_to_process, fd_stats_to_process
        )

    if traits_to_process is None and fd_stats_to_process is None:
        log.info("All requested traits already processed. Done!")
        return

    if fd_stats_to_process:
        log.info(
            "Will process %d FD stats: %s",
            len(fd_stats_to_process),
            ", ".join(fd_stats_to_process),
        )
    else:
        log.info(
            "Will process %d traits: %s",
            len(traits_to_process),
            ", ".join(traits_to_process),
        )

    # create dict of dask kws, but only if they are not None
    dask_kws = {k: v for k, v in sys_cfg.dask.items() if v is not None}
    with Client(dashboard_address=cfg.dask_dashboard, **dask_kws):
        header = _load_splot_header(
            cfg.crs, sys_cfg.npartitions, splot_dir, cfg.splot_open
        )
        traits = load_try_traits(sys_cfg.npartitions, traits_to_process)
        pfts = _load_pfts(cfg.PFT)
        vegetation, abund_col = _load_splot_vegetation(
            sys_cfg.npartitions, splot_dir, cfg.splot_open
        )
        veg_and_traits = _filter_by_pft_and_merge_with_traits(
            pfts=pfts, traits=traits, vegetation=vegetation
        )

        if fd_mode:
            _generate_fd_maps(
                cfg=cfg,
                out_dir=out_dir,
                traits_to_process=traits_to_process,
                fd_stats_to_process=fd_stats_to_process,
                veg_and_traits=veg_and_traits,
                header=header,
            )
        else:
            _generate_trait_maps(
                cfg=cfg,
                out_dir=out_dir,
                traits_to_process=traits_to_process,
                veg_and_traits=veg_and_traits,
                header=header,
                abund_col=abund_col,
            )


############################################################################


def _load_pfts(PFT: str) -> pd.DataFrame:
    """Load PFT data, filter by desired PFT, clean species names, and set them as index
    for joining"""

    log.info("Loading and processing PFT data...")
    pfts = load_pfts()
    pfts = (
        pfts.astype(
            {
                "AccSpeciesName": "string[pyarrow]",
                "pft": "category",
            }
        )
        .pipe(filter_pft, PFT)
        .drop(columns=["AccSpeciesID"])
        .dropna(subset=["AccSpeciesName"])
        .pipe(clean_species_name, "AccSpeciesName", "speciesname")
        .drop(columns=["AccSpeciesName"])
        .drop_duplicates(subset=["speciesname"])
        .set_index("speciesname")
    )
    return pfts


def _load_splot_header(
    to_crs: str, npartitions: int, splot_dir: Path, splot_open: bool
) -> dd.DataFrame:
    """Load header and set plot IDs as index for later joining with vegetation data."""
    if splot_open:
        log.info("Loading sPlot Open header data...")
    else:
        log.info("Loading sPlot Full header data...")

    givd_col = "GIVD_ID" if splot_open else "GIVD_NU"

    header = (
        dd.read_parquet(
            splot_dir / "header.parquet",
            columns=["PlotObservationID", "Longitude", "Latitude", givd_col],
        )
        .astype(
            {
                "PlotObservationID": "uint32[pyarrow]",
                givd_col: "category",
            }
        )
        .pipe(repartition_if_set, npartitions)
        .pipe(filter_certain_plots, givd_col, "00-RU-008")
        .drop(columns=[givd_col])
        .astype({"Longitude": np.float64, "Latitude": np.float64})
        .set_index("PlotObservationID")
        .map_partitions(reproject_geo_to_xy, to_crs=to_crs, x="Longitude", y="Latitude")
        .drop(columns=["Longitude", "Latitude"])
    )
    log.info("Header data loaded and processed")
    return header


def _load_splot_vegetation(
    npartitions: int, splot_dir: Path, splot_open: bool
) -> tuple[dd.DataFrame, str]:
    """Load sPlot vegetation data, clean species names, and set them as index for later
    joining with trait data."""

    log.info("Loading sPlot vegetation data...")

    abund_col = "Relative_cover" if splot_open else "Rel_Abund_Plot"

    veg = (
        dd.read_parquet(
            splot_dir / "vegetation.parquet",
            columns=[
                "PlotObservationID",
                "Species",
                abund_col,
            ],
        )
        .astype(
            {
                "PlotObservationID": "uint32[pyarrow]",
                "Species": "string[pyarrow]",
                abund_col: "float64[pyarrow]",
            }
        )
        .pipe(repartition_if_set, npartitions)
        .dropna(subset=["Species", abund_col])
        .pipe(clean_species_name, "Species", "speciesname")
        .drop(columns=["Species"])
        .set_index("speciesname")
    )

    return veg, abund_col


def _filter_by_pft_and_merge_with_traits(
    pfts: pd.DataFrame, traits: dd.DataFrame, vegetation: dd.DataFrame
) -> dd.DataFrame:
    """Filter splot species observations by PFT and join with trait data."""

    log.info("Filtering sPlot vegetation data by PFT and joining with trait data...")
    return (
        vegetation.join(pfts, how="inner")
        .join(traits, how="inner")
        .reset_index()
        .drop(columns=["pft"])
        .persist()
    )


def _setup_in_and_out_dirs(cfg: ConfigBox) -> tuple[Path, Path]:
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
    return splot_dir, out_dir


def _generate_fd_maps(
    cfg: ConfigBox,
    out_dir: Path,
    traits_to_process: list[str],
    fd_stats_to_process: list[str],
    veg_and_traits: dd.DataFrame,
    header: dd.DataFrame,
    abund_col: str,
) -> None:
    log.info("Functional diversity mode detected.")

    grids = []
    for i, stat in enumerate(fd_stats_to_process):
        log.info("Processing FD stat %s...", stat)
        # Calculate community-weighted means per plot, join with `header` to get
        # plot lat/lons, and grid at the configured resolution.
        log.info(
            "Filtering plots with insufficient observations and relative abundance..."
        )
        df = (
            veg_and_traits[
                [
                    "PlotObservationID",
                    abund_col,
                    "speciesname",
                    *traits_to_process,
                ]
            ]
            .set_index("PlotObservationID")
            .groupby("PlotObservationID")
            .apply(
                _plot_level_fd_metrics,
                traits_to_process=traits_to_process,
                stats=[stat],
                use_ses=True,
                random_seed=cfg.random_seed,
                abund_col=abund_col,
                meta={stat: "f8" for stat in [stat]},
            )
            .pipe(pipe_log, "Joining with header...")
            .join(header, how="inner")
            .reset_index(drop=True)
            .pipe(pipe_log, "Computing dask dataframe...")
            .compute()
        )

        log.info("Rasterizing FD stat %s...", stat)
        if i == len(fd_stats_to_process) - 1:
            funcs = {stat: "mean", "count": "count"}
        else:
            funcs = {stat: "mean"}

        ds = rasterize_points(
            df,
            data_cols=stat,
            res=cfg.target_resolution,
            crs=cfg.crs,
            agg=True,
            funcs=funcs,
        )
        grids.append(ds)
        log.info("✓ %s rasterized", stat)

    # Merge all of the gridded data into a single dataset
    if len(grids) > 1:
        log.info("Merging gridded data...")
        gridded_trait = xr.merge(grids)
    else:
        gridded_trait = grids[0]
    log.info("Gridded data merged")

    out_fn = out_dir / f"{stat}.tif"
    log.info("Writing %s to disk...", stat)
    xr_to_raster(gridded_trait, out_fn)
    log.info("✓ Wrote %s", out_fn)


def _plot_level_fd_metrics(
    df: pd.DataFrame,
    traits_to_process: list[str],
    stats: list[str],
    use_ses: bool,
    random_seed: int,
    abund_col: str,
) -> pd.Series:
    """Calculate FD metrics at the plot level."""
    # Filter out plots with insufficient observations and relative abundance
    if len(df) < len(traits_to_process) or df["Rel_Abund_Plot"].sum() < 0.8:
        return pd.Series({stat: np.nan for stat in stats})

    # Calculate FD metrics
    return fd_metrics(
        df,
        trait_cols=traits_to_process,
        stats=stats,
        species_col="speciesname",
        abundance_col=abund_col,
        use_ses=use_ses,
        random_seed=random_seed,
    )


def _generate_trait_maps(
    cfg: ConfigBox,
    out_dir: Path,
    traits_to_process: list[str],
    veg_and_traits: dd.DataFrame,
    header: dd.DataFrame,
    abund_col: str,
) -> None:
    log.info("Trait mode detected.")

    for col in traits_to_process:
        log.info("Processing trait %s...", col)
        # Calculate community-weighted means per plot, join with `header` to get
        # plot lat/lons, and grid at the configured resolution.
        log.info("Calculating community-weighted statistics...")
        df = (
            veg_and_traits[["PlotObservationID", abund_col, col]]
            .set_index("PlotObservationID")
            .groupby("PlotObservationID")
            .apply(
                cw_stats,
                col,
                abund_col,
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
                data_cols=stat_col,
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


if __name__ == "__main__":
    main()
