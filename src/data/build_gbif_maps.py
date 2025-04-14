"""
Match subsampled GBIF data with filtered trait data, grid it, generate grid cell
statistics, and write each trait's corresponding raster stack to GeoTIFF files.
"""

import argparse
from pathlib import Path

import dask.dataframe as dd
from box import ConfigBox
from dask.distributed import Client

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.df_utils import rasterize_points, reproject_geo_to_xy
from src.utils.raster_utils import xr_to_raster
from src.utils.trait_utils import filter_pft


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Match GBIF data with TRY mean trait data, grid it, generate grid cell statistics, and write each trait's corresponding raster stack to GeoTIFF files."
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite existing files."
    )
    parser.add_argument(
        "-t",
        "--trait",
        type=str,
        help="Trait ID to process (e.g. '3'). If not provided, all traits will be processed.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """Main function."""
    syscfg = cfg[detect_system()][cfg.model_res]["build_gbif_maps"]

    def _repartition_if_set(df: dd.DataFrame, npartitions: int | None) -> dd.DataFrame:
        return (
            df.repartition(npartitions=npartitions) if npartitions is not None else df
        )

    npartitions = syscfg.get("npartitions", None)

    out_dir = (
        Path(cfg.interim_dir)
        / cfg.gbif.interim.dir
        / cfg.gbif.interim.traits
        / cfg.PFT
        / cfg.model_res
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_traits = [str(trait_num) for trait_num in cfg.datasets.Y.traits]

    # If trait is specified, only process that one
    if args.trait:
        if args.trait not in valid_traits:
            raise ValueError(
                f"Invalid trait ID: {args.trait}. Valid traits are: {', '.join(valid_traits)}"
            )
        traits_to_process = [args.trait]
    else:
        traits_to_process = valid_traits

    with Client(
        dashboard_address=cfg.dask_dashboard,
        n_workers=syscfg.n_workers,
        threads_per_worker=syscfg.threads_per_worker,
    ):
        # Load data
        gbif = (
            dd.read_parquet(
                Path(cfg.interim_dir, cfg.gbif.interim.dir, cfg.gbif.interim.matched)
            )
            .pipe(_repartition_if_set, npartitions)
            .pipe(filter_pft, cfg.PFT)
            .set_index("speciesname")
        )

        # Only load the needed trait columns
        needed_columns = ["speciesname"]
        needed_columns.extend([f"X{trait_num}" for trait_num in traits_to_process])

        mn_traits = (
            dd.read_parquet(
                Path(
                    cfg.interim_dir, cfg.trydb.interim.dir, cfg.trydb.interim.filtered
                ),
                columns=needed_columns,
            )
            .pipe(_repartition_if_set, npartitions)
            .set_index("speciesname")
        )

        # Merge GBIF and trait data
        merged = (
            gbif.join(mn_traits, how="inner")
            .reset_index(drop=True)
            .drop(columns=["pft"])
        )

        # Reproject coordinates to target CRS
        if cfg.crs != "EPSG:4326":
            if cfg.crs != "EPSG:6933":
                raise ValueError(f"Unsupported CRS: {cfg.crs}")

            merged = merged.map_partitions(
                reproject_geo_to_xy,
                to_crs=cfg.crs,
                x="decimallongitude",
                y="decimallatitude",
            ).drop(columns=["decimallatitude", "decimallongitude"])

        # Grid trait stats for each specified trait
        cols = [f"X{trait}" for trait in traits_to_process]

        for col in cols:
            out_fn = out_dir / f"{col}.tif"
            if out_fn.exists() and not args.overwrite:
                log.info("%s.tif already exists. Skipping...", col)
                continue

            log.info("Processing trait %s...", col)
            raster = rasterize_points(
                merged[["x", "y", col]],
                data=str(col),
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


if __name__ == "__main__":
    main()
