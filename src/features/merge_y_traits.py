"""
Merge individual trait Y files into final Y.parquet.

This script reads all individual trait parquet files from the tmp directory,
merges them on spatial coordinates and source, and writes the final Y.parquet
file to the features directory.
"""

import argparse
import os
import shutil
from pathlib import Path

import dask.dataframe as dd
from dask.distributed import Client

from src.conf.conf import get_config
from src.conf.environment import detect_system, log


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge individual trait Y files into final Y.parquet."
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
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Main function to merge trait files."""
    args = cli() if args is None else args
    cfg = get_config(params_path=args.params)
    syscfg = cfg[detect_system()]["merge_y_traits"]

    # Set up paths
    proj_root = os.environ.get("PROJECT_ROOT")
    if proj_root is None:
        raise ValueError("PROJECT_ROOT environment variable is not set")
    tmp_dir = Path(proj_root) / cfg.tmp_dir / "y_traits"
    out_fn = Path(cfg.train.dir, cfg.product_code, cfg.train.Y.fn)

    # Check if output already exists
    if out_fn.exists() and not args.overwrite:
        log.info("Output file %s already exists. Use --overwrite to replace.", out_fn)
        return

    # Find all trait parquet files
    trait_files = sorted(list(tmp_dir.glob("*.parquet")))

    if not trait_files:
        log.error("No trait files found in %s", tmp_dir)
        return

    log.info("Found %d trait files to merge", len(trait_files))

    # Create output directory
    out_fn.parent.mkdir(parents=True, exist_ok=True)

    # Use dask for memory-efficient merging
    with Client(
        n_workers=syscfg.n_workers,
        threads_per_worker=syscfg.threads_per_worker,
        memory_limit=syscfg.memory_limit,
    ):
        log.info("Reading trait files...")

        # Read the first file to start
        y_df = dd.read_parquet(trait_files[0])
        log.info("Loaded base file: %s", trait_files[0].stem)

        # Merge remaining files
        for i, trait_file in enumerate(trait_files[1:], start=2):
            log.info("Merging file %d/%d: %s", i, len(trait_files), trait_file.stem)

            trait_df = dd.read_parquet(trait_file)

            # Merge on common spatial coordinates and source
            y_df = y_df.merge(trait_df, on=["x", "y", "source"], how="outer")

        log.info("All files merged. Writing to disk...")

        # Write to parquet with compression
        y_df.to_parquet(out_fn, compression="zstd", write_index=False)

        log.info("✓ Successfully wrote Y data to %s", out_fn)

    # Clean up temporary files
    log.info("Cleaning up temporary files...")
    try:
        shutil.rmtree(tmp_dir)
        log.info("✓ Temporary directory removed: %s", tmp_dir)
    except Exception as e:
        log.warning("Failed to remove temporary directory %s: %s", tmp_dir, e)

    log.info("Done!")


if __name__ == "__main__":
    main()
