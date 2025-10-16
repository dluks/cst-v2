"""
Match subsampled GBIF data with filtered trait data, grid it, generate grid cell
statistics, and write each trait's corresponding raster stack to GeoTIFF files.
"""

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
from src.utils.df_utils import rasterize_points, reproject_geo_to_xy, xy_to_rowcol_df
from src.utils.raster_utils import create_sample_raster, xr_to_raster
from src.utils.trait_aggregation_utils import fd_metrics
from src.utils.trait_utils import (
    check_for_existing_maps,
    filter_pft,
    get_traits_to_process,
    load_try_traits,
)


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Match GBIF data with TRY mean trait data, grid it, generate grid cell statistics, and write each trait's corresponding raster stack to GeoTIFF files."
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Resume processing from existing files.",
    )
    parser.add_argument(
        "-t",
        "--trait",
        type=int,
        help="Trait ID to process (e.g. '3'). If not provided, all traits will be processed.",
    )
    parser.add_argument(
        "-f",
        "--fd-metric",
        type=str,
        help="Functional diversity metric to process (e.g. 'f_ric'). If not provided, all metrics will be processed.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """Main function."""
    syscfg = cfg[detect_system()][cfg.model_res]["build_gbif_maps"]

    npartitions = syscfg.get("npartitions", None)

    out_dir = (
        Path(cfg.interim_dir)
        / cfg.gbif.interim.dir
        / cfg.gbif.interim.traits
        / cfg.PFT
        / cfg.model_res
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if we need to compute functional diversity metrics
    trait_stats = cfg.datasets.Y.trait_stats
    fd_metrics = ["f_ric", "f_eve", "f_div", "f_red", "sp_ric", "f_ric_ses"]
    fd_mode = any(stat in fd_metrics for stat in trait_stats)
    using_pca = cfg.trydb.interim.perform_pca

    if args.trait and using_pca:
        raise ValueError("Trait ID not supported when using PCA.")

    if args.fd_metric and args.fd_metric not in fd_metrics:
        raise ValueError(
            f"Invalid FD metric: {args.fd_metric}. Valid metrics are: {', '.join(fd_metrics)}"
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

    if traits_to_process is None:
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

    with Client(
        dashboard_address=cfg.dask_dashboard,
        n_workers=syscfg.n_workers,
        threads_per_worker=syscfg.threads_per_worker,
    ):
        # Load data
        log.info("Loading GBIF data...")
        gbif = _load_gbif(cfg, npartitions)
        log.info("Loading trait data...")
        traits = load_try_traits(npartitions, traits_to_process)
        log.info("Joining GBIF and trait data...")
        gbif_traits = gbif.join(traits, how="inner").reset_index().drop(columns=["pft"])
        log.info("Reprojecting GBIF and trait data...")
        gbif_traits = _reproject(cfg, gbif_traits)

        if fd_mode:
            # Calculate functional diversity metrics
            _generate_fd_maps(
                cfg, out_dir, traits_to_process, fd_stats_to_process, gbif_traits
            )

        else:
            for col in traits_to_process:
                out_fn = out_dir / f"{col}.tif"
                if out_fn.exists() and args.resume:
                    log.info("%s.tif already exists. Skipping...", col)
                    continue

                log.info("Processing trait %s...", col)
                raster = rasterize_points(
                    gbif_traits[["x", "y", col]],
                    data_cols=str(col),
                    res=cfg.target_resolution,
                    crs=cfg.crs,
                    agg=True,
                    n_min=cfg.gbif.interim.min_count,
                    n_max=cfg.gbif.interim.max_count,
                )

                log.info("Writing to disk...")
                xr_to_raster(raster, out_fn)
                log.info("Wrote %s.tif.", col)

    log.info("Done!")


def _load_gbif(cfg, npartitions):
    gbif = (
        dd.read_parquet(
            Path(cfg.interim_dir, cfg.gbif.interim.dir, cfg.gbif.interim.matched)
        )
        .pipe(repartition_if_set, npartitions)
        .pipe(filter_pft, cfg.PFT)
        .set_index("speciesname")
        # .sample(frac=0.01)
    )

    return gbif


def _reproject(cfg, gbif_traits):
    if cfg.crs != "EPSG:4326":
        if cfg.crs != "EPSG:6933":
            raise ValueError(f"Unsupported CRS: {cfg.crs}")

        gbif_traits = gbif_traits.map_partitions(
            reproject_geo_to_xy,
            to_crs=cfg.crs,
            x="decimallongitude",
            y="decimallatitude",
        ).drop(columns=["decimallatitude", "decimallongitude"])

    return gbif_traits


def _generate_fd_maps(
    cfg, out_dir, traits_to_process, fd_stats_to_process, gbif_traits
):
    for stat in fd_stats_to_process:
        log.info("Processing FD stat %s...", stat)

        ref = create_sample_raster(resolution=cfg.target_resolution, crs=cfg.crs)
        transform = ref.rio.transform().to_gdal()

        log.info("Retrieving row and column indices and calculating FD metrics...")
        grid_df = (
            xy_to_rowcol_df(gbif_traits, transform, x="x", y="y")
            .drop(columns=["x", "y"])
            .groupby(["row", "col"])
            .apply(
                _grid_cell_fd_metrics,
                traits_to_process=traits_to_process,
                stats=[stat],
                use_ses=False,
                random_seed=cfg.random_seed,
                min_count=cfg.gbif.interim.min_count,
                max_count=cfg.gbif.interim.max_count,
                meta={stat: "f8", "count": "i4"},
            )
        )

        log.info("Rasterizing gridded data...")
        ds = rasterize_points(
            grid_df.reset_index().compute().astype({"row": "int32", "col": "int32"}),
            data_cols=stat,
            ref_raster=ref,
            already_row_col=True,
        )

        out_fn = out_dir / f"{stat}.tif"
        log.info("Writing %s to disk...", stat)
        xr_to_raster(ds, out_fn)
        log.info("✓ Wrote %s", out_fn)


def _grid_cell_fd_metrics(
    df: pd.DataFrame,
    traits_to_process: list[str],
    stats: list[str],
    use_ses: bool,
    random_seed: int,
    min_count: int = 10,
    max_count: int = 500,
) -> pd.Series:
    """Calculate FD metrics and the number of observations at the plot level."""
    count = len(df)
    final_series_dict = {stat: np.nan for stat in stats}
    final_series_dict.update({"count": 0})

    if count < min_count:
        return pd.Series(final_series_dict)

    if count > max_count:
        df = df.sample(max_count, random_state=random_seed)
        final_series_dict.update({"count": max_count})

    # Calculate FD metrics
    fd_stat = fd_metrics(
        df,
        trait_cols=traits_to_process,
        stats=stats,
        species_col="speciesname",
        use_ses=use_ses,
        random_seed=random_seed,
    )

    final_series_dict.update(fd_stat)

    return pd.Series(final_series_dict)


if __name__ == "__main__":
    main()
