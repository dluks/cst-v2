"""
Match filtered GBIF data with trait data, grid it using weighted aggregation,
and write each trait's corresponding raster stack to GeoTIFF files.

Weights from resurvey calculations are applied during grid cell aggregation to
ensure locations sampled across multiple years contribute equally.
"""

import argparse
from pathlib import Path

import dask.dataframe as dd
from dask.distributed import Client

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import repartition_if_set
from src.utils.df_utils import rasterize_points, reproject_geo_to_xy
from src.utils.raster_utils import xr_to_raster
from src.utils.trait_utils import filter_pft


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Match GBIF data with TRY mean trait data, grid it, generate grid "
        "cell statistics, and write each trait's corresponding raster stack to GeoTIFF "
        "files."
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        help="Path to the parameters file.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "-t",
        "--trait",
        type=str,
        help="Trait name to process (e.g. 'X11', 'PC1', or 'gsmax').",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Main function."""
    args = cli() if args is None else args
    cfg = get_config(params_path=args.params)

    syscfg = cfg[detect_system()]["build_gbif_maps"]

    npartitions = syscfg.get("npartitions", None)

    out_dir = Path(cfg.gbif.maps.out_dir, cfg.product_code)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Client(
        n_workers=syscfg.n_workers,
        threads_per_worker=syscfg.threads_per_worker,
    ):
        # Load data
        log.info("Loading GBIF data...")
        gbif = _load_gbif(
            Path(cfg.gbif.filtered.out_dir, cfg.trait_type, cfg.gbif.filtered.fp),
            cfg.PFT,
            npartitions,
        )

        log.info("Loading trait data...")
        trait = (
            dd.read_parquet(cfg.traits.interim_out, columns=["GBIFKeyGBIF", args.trait])
            .rename(columns={"GBIFKeyGBIF": "specieskey"})
            .set_index("specieskey")
        )

        log.info("Joining GBIF and trait data...")
        gbif_traits = gbif.join(trait, how="inner").reset_index(drop=True)

        log.info("Reprojecting GBIF and trait data...")
        gbif_traits = _reproject(cfg, gbif_traits)

        out_fn = out_dir / f"{args.trait}.tif"
        if out_fn.exists() and not args.overwrite:
            log.info("%s.tif already exists. Skipping...", args.trait)
            return

        log.info("Processing trait %s...", args.trait)
        raster = rasterize_points(
            gbif_traits[["x", "y", args.trait, "weight"]],
            data_cols=str(args.trait),
            res=cfg.target_resolution,
            crs=cfg.crs,
            agg=True,
            n_min=cfg.gbif.maps.min_count,
            n_max=cfg.gbif.maps.max_count,
            weights="weight",
        )

        log.info("Writing to disk...")
        xr_to_raster(raster, out_fn)
        log.info("Wrote %s.tif.", args.trait)

    log.info("Done!")


def _load_gbif(fp: Path, pfts: list[str], npartitions: int = 1) -> dd.DataFrame:
    """Load filtered GBIF data and filter by PFT, keeping weights."""
    gbif = (
        dd.read_parquet(fp)
        .pipe(repartition_if_set, npartitions)
        .pipe(filter_pft, pfts)
        .drop(columns=["pft"])
        .set_index("specieskey")
        # .sample(frac=0.01)
    )

    return gbif


def _reproject(cfg, gbif_traits):
    """Reproject coordinates if needed."""
    if cfg.crs != "EPSG:4326":
        if cfg.crs != "EPSG:6933":
            raise ValueError(f"Unsupported CRS: {cfg.crs}")

        gbif_traits = gbif_traits.map_partitions(
            reproject_geo_to_xy,
            to_crs=cfg.crs,
            x="decimallongitude",
            y="decimallatitude",
        ).drop(columns=["decimallatitude", "decimallongitude"])
    else:
        # Rename columns to x and y for consistency
        gbif_traits = gbif_traits.rename(
            columns={"decimallongitude": "x", "decimallatitude": "y"}
        )

    return gbif_traits


if __name__ == "__main__":
    main()
