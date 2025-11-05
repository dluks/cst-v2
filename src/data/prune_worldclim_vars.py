#!/usr/bin/env python
"""
Prune WorldClim variables and calculate derived variables.

This script processes harmonized WorldClim bioclimatic variables by:
1. Calculating derived variables (e.g., bio_13-14 = bio_13 - bio_14)
2. Deleting intermediate files not in the configured bio_vars list
"""

import argparse
import gc
import os
from pathlib import Path

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.raster_utils import open_raster, xr_to_raster


def cli() -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Prune WorldClim variables and calculate derived variables."
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

    # Load configuration
    params_path = Path(args.params).resolve() if args.params else None
    if params_path:
        log.info("Params path: %s", str(params_path))

    cfg = get_config(params_path=params_path)

    # Check if worldclim is in datasets
    if "worldclim" not in cfg.datasets:
        log.warning("WorldClim not found in datasets configuration. Skipping pruning.")
        print("⊘ WorldClim not in datasets, skipping pruning")
        return

    # Get bio_vars from configuration
    bio_vars = cfg.worldclim.bio_vars
    log.info("Bio variables to keep: %s", ", ".join(bio_vars))

    # Get output directory
    out_dir = Path(cfg.interim.out_dir).resolve()
    worldclim_dir = out_dir / "worldclim"

    if not worldclim_dir.exists():
        log.error("WorldClim output directory not found: %s", str(worldclim_dir))
        print(f"✗ WorldClim output directory not found: {worldclim_dir}")
        return

    # Get all WorldClim files
    fns = list(worldclim_dir.glob("*.tif"))
    log.info("Found %d WorldClim files", len(fns))

    # Process derived variables (e.g., "13-14" means bio_13 - bio_14)
    derived_count = 0
    for bio_var in bio_vars:
        if "-" in bio_var:
            start, end = bio_var.split("-")
            log.info("Calculating derived variable: bio_%s-%s", start, end)

            # Find source files
            start_files = [fn for fn in fns if f"bio_{start}.tif" in fn.name]
            end_files = [fn for fn in fns if f"bio_{end}.tif" in fn.name]

            if not start_files or not end_files:
                log.warning(
                    "Could not find source files for bio_%s-%s. Skipping.",
                    start,
                    end,
                )
                continue

            # Determine output path
            diff_name = f"wc2.1_30s_bio_{start}-{end}"
            out_path = worldclim_dir / f"{diff_name}.tif"

            # Check if output already exists
            if out_path.exists() and not args.overwrite:
                log.info("Derived variable already exists, skipping: %s", out_path.name)
                derived_count += 1
                continue

            # Calculate difference
            da1 = open_raster(start_files[0]).sel(band=1)
            da2 = open_raster(end_files[0]).sel(band=1)
            diff = da1 - da2
            diff.attrs["long_name"] = diff_name

            # Write output
            log.info("Writing derived variable: %s", out_path.name)
            xr_to_raster(diff, out_path)

            # Clean up
            for da in [da1, da2, diff]:
                da.close()

            del da1, da2, diff
            gc.collect()

            derived_count += 1

    log.info("Calculated %d derived variables", derived_count)

    # Delete files that don't contain a configured bio_var
    deleted_count = 0
    for fn in fns:
        # Check if this file corresponds to any configured bio_var
        keep = False
        for var in bio_vars:
            # Handle both simple vars (e.g., "1") and derived vars (e.g., "13-14")
            if "-" in var:
                # Derived variable - check if output file exists
                derived_name = f"bio_{var}.tif"
                if derived_name in fn.name:
                    keep = True
                    break
            else:
                # Simple variable
                if f"bio_{var}.tif" in fn.name:
                    keep = True
                    break

        if not keep:
            log.info("Deleting unused file: %s", fn.name)
            fn.unlink()
            deleted_count += 1

    log.info("Deleted %d unused WorldClim files", deleted_count)
    print(f"✓ Processed WorldClim: {derived_count} derived, {deleted_count} deleted")


if __name__ == "__main__":
    cli_args = cli()
    main(cli_args)
