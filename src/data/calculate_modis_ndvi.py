#!/usr/bin/env python
"""
Calculate MODIS NDVI from harmonized red and NIR bands.

This script calculates the Normalized Difference Vegetation Index (NDVI) from
MODIS satellite data after the individual bands have been harmonized.
"""

import argparse
import gc
import os
from pathlib import Path

from joblib import Parallel, delayed

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.raster_utils import open_raster, xr_to_raster


def cli() -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Calculate MODIS NDVI from harmonized red and NIR bands."
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


def process_month(month: int, modis_dir: Path, overwrite: bool) -> tuple[int, bool]:
    """
    Process NDVI for a single month.

    Args:
        month: Month number (1-12)
        modis_dir: Directory containing MODIS files
        overwrite: Whether to overwrite existing files

    Returns:
        Tuple of (month, success)
    """
    # Find red and NIR band files for this month
    fns = sorted(list(modis_dir.glob(f"*_m{month}_*.tif")))

    if len(fns) < 2:
        log.warning(
            "Expected 2 files for month %d, found %d. Skipping.", month, len(fns)
        )
        return month, False

    # Determine output path
    out_path = fns[0].parent / fns[0].name.replace("b01", "ndvi")

    # Check if output already exists
    if out_path.exists() and not overwrite:
        log.info(
            "NDVI file already exists for month %d, skipping: %s",
            month,
            out_path.name,
        )
        return month, True

    log.info("Processing month %d...", month)

    # Open red (b01) and NIR (b02) bands
    red = open_raster(fns[0]).sel(band=1)
    nir = open_raster(fns[1]).sel(band=1)

    # Calculate NDVI: (NIR - Red) / (NIR + Red)
    ndvi = (nir - red) / (nir + red)

    # Scale the values prior to int conversion
    ndvi = ndvi * 10000
    ndvi.attrs["long_name"] = out_path.stem

    # Write output
    log.info("Writing NDVI for month %d: %s", month, out_path.name)
    xr_to_raster(ndvi, out_path, dtype="int16")

    # Clean up
    for da in [red, nir, ndvi]:
        da.close()

    del red, nir, ndvi
    gc.collect()

    return month, True


def main(args: argparse.Namespace) -> None:
    """Main function."""
    log.info("Current working directory: %s", os.getcwd())

    # Load configuration
    params_path = Path(args.params).resolve() if args.params else None
    if params_path:
        log.info("Params path: %s", str(params_path))

    cfg = get_config(params_path=params_path)

    # Check if modis is in datasets
    if "modis" not in cfg.datasets:
        log.warning(
            "MODIS not found in datasets configuration. Skipping NDVI calculation."
        )
        print("⊘ MODIS not in datasets, skipping NDVI calculation")
        return

    # Get output directory
    out_dir = Path(cfg.interim.out_dir).resolve()
    modis_dir = out_dir / "modis"

    if not modis_dir.exists():
        log.error("MODIS output directory not found: %s", str(modis_dir))
        print(f"✗ MODIS output directory not found: {modis_dir}")
        return

    # Determine number of workers based on system
    system = detect_system()
    n_workers = getattr(cfg.get(system, {}), "calculate_modis_ndvi", {}).get(
        "n_workers", 12
    )

    log.info("Calculating MODIS NDVI for 12 months using %d workers...", n_workers)

    # Process all months in parallel
    results = Parallel(n_jobs=n_workers, backend="loky")(
        delayed(process_month)(month, modis_dir, args.overwrite)
        for month in range(1, 13)
    )

    # Count successful results
    processed = sum(1 for _, success in results if success)

    log.info("Completed NDVI calculation for %d months", processed)
    print(f"✓ Calculated NDVI for {processed} months")


if __name__ == "__main__":
    cli_args = cli()
    main(cli_args)
