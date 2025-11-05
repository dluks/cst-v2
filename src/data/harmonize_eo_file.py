#!/usr/bin/env python
"""
Worker script to harmonize a single EO data file.

This script processes a single raster file by reprojecting it to the target
resolution, masking out non-vegetation pixels, and saving as a GeoTIFF.
"""

import argparse
import gc
import os
from pathlib import Path

import numpy as np

from src.conf.conf import get_config
from src.conf.environment import log
from src.data.mask import get_mask, mask_raster
from src.utils.raster_utils import (
    create_sample_raster,
    open_raster,
    pack_xr,
    xr_to_raster_rasterio,
)


def cli() -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Harmonize a single EO data file.")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Path to the input raster file to process.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., modis, worldclim, soilgrids).",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        default=None,
        help=(
            "Optional path to a params.yaml whose values should be layered as the "
            "final override (equivalent to setting PRODUCT_PARAMS)."
        ),
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Main function."""
    log.info("Current working directory: %s", os.getcwd())

    # Enable GDAL multi-threading based on allocated CPUs
    # This significantly speeds up reprojection operations
    n_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    os.environ["GDAL_NUM_THREADS"] = n_cpus
    log.info("Setting GDAL_NUM_THREADS=%s", n_cpus)

    # Resolve file path and params path
    filename = Path(args.file).resolve()
    params_path = Path(args.params).resolve() if args.params else None

    log.info("Processing file: %s", str(filename))
    log.info("Dataset: %s", args.dataset)
    if params_path:
        log.info("Params path: %s", str(params_path))

    # Load configuration
    cfg = get_config(params_path=params_path)

    # Setup output directory
    out_dir = Path(cfg.interim.out_dir).resolve()
    out_path = out_dir / args.dataset / filename.with_suffix(".tif").name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if output already exists
    if out_path.exists() and not args.overwrite:
        log.info("Output file already exists, skipping: %s", str(out_path))
        print(f"✓ Skipped (already exists): {filename.name}")
        return

    log.info("Building reference raster...")
    target_sample_raster = create_sample_raster(
        resolution=cfg.target_resolution, crs=cfg.crs
    )

    landcover_fp = Path(cfg.landcover_mask.path).resolve()
    log.info("Building landcover mask from %s...", str(landcover_fp))
    mask = get_mask(
        landcover_fp, cfg.landcover_mask.keep_classes, cfg.base_resolution, cfg.crs
    )

    log.info("Processing file: %s", str(filename))

    # Process the file (reusing logic from harmonize_eo_data.py)
    rast = open_raster(filename).sel(band=1)

    if rast.rio.nodata is None:
        # Make sure that the raster has a nodata value or else the reproject_match
        # method will treat nan values as actual data
        rast = rast.rio.write_nodata(np.nan)

    rast = rast.rio.reproject_match(mask)
    rast_masked = mask_raster(rast, mask)

    rast.close()
    mask.close()
    del rast, mask

    if rast_masked.rio.resolution() != target_sample_raster.rio.resolution():
        rast_masked = rast_masked.rio.reproject_match(target_sample_raster)

    dtype = rast_masked.dtype

    # Apply dataset-specific transformations
    if args.dataset == "modis":
        # Values outside this range usually represent errors in the atmospheric
        # correction algorithm
        rast_masked = rast_masked.clip(0, 10000)
        dtype = "int16"

    if args.dataset == "soilgrids":
        dtype = "int16"
        # some soil properties have smaller ranges
        if (
            rast_masked.max() < np.iinfo(np.int8).max
            and rast_masked.min() >= np.iinfo(np.int8).min
        ):
            dtype = "int8"

    if args.dataset in ("canopy_height", "alos"):
        dtype = "uint8"

    if args.dataset in ("worldclim", "vodca"):
        dtype = "int16"
        rast_masked = pack_xr(rast_masked)

    if "long_name" not in rast_masked.attrs:
        rast_masked.attrs["long_name"] = filename.stem

    # Determine nodata value based on dtype
    match dtype:
        case "uint8":
            nodata = np.iinfo(np.uint8).max
        case "int8":
            nodata = np.iinfo(np.int8).min
        case "int16":
            nodata = np.iinfo(np.int16).min
        case _:
            nodata = np.nan

    # Write output
    log.info("Writing output: %s", str(out_path))
    xr_to_raster_rasterio(rast_masked, out_path, nodata=nodata)

    rast_masked.close()
    del rast_masked

    gc.collect()

    log.info("Completed processing: %s", filename.name)
    print(f"✓ Completed: {filename.name}")


if __name__ == "__main__":
    cli_args = cli()
    main(cli_args)
