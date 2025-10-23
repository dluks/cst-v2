"""
Process a single trait's Y data from GBIF and sPlot maps.

This script loads trait maps from both GBIF and sPlot datasets, extracts the
specified band (trait statistic), converts to dataframe format, and saves to
an intermediate parquet file for later merging.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.raster_utils import open_raster


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process a single trait's Y data from GBIF and sPlot maps."
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        help="Path to the parameters file.",
    )
    parser.add_argument(
        "-t",
        "--trait",
        type=str,
        required=True,
        help="Trait name to process (e.g., 'gsmax', 'P12').",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Main function to process a single trait."""
    args = cli() if args is None else args
    cfg = get_config(params_path=args.params)

    # Set up output directory
    proj_root = os.environ.get("PROJECT_ROOT")
    if proj_root is None:
        raise ValueError("PROJECT_ROOT environment variable is not set")

    out_dir = Path(proj_root) / cfg.tmp_dir / "y_traits"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir.absolute())

    out_fn = out_dir / f"{args.trait}.parquet"
    if out_fn.exists() and not args.overwrite:
        log.info("%s.parquet already exists. Skipping...", args.trait)
        return

    log.info("Processing trait: %s", args.trait)

    # Get the trait statistic band to extract
    try:
        trait_stat = cfg.traits.trait_map_layers.index(cfg.traits.target_trait_stat) + 1
    except ValueError:
        raise ValueError(
            f"Target trait stat {cfg.target_trait_stat} not found in trait map layers."
        )

    # Load GBIF map
    trait_fp = Path(cfg.gbif.maps.out_dir, cfg.product_code) / f"{args.trait}.tif"
    gbif_df = load_trait_map("gbif", trait_fp, trait_stat)

    # Load sPlot map
    trait_fp = Path(cfg.splot.maps.out_dir, cfg.product_code) / f"{args.trait}.tif"
    splot_df = load_trait_map("splot", trait_fp, trait_stat)

    # Combine dataframes
    dfs_to_concat = []
    if gbif_df is not None:
        dfs_to_concat.append(gbif_df)
    if splot_df is not None:
        dfs_to_concat.append(splot_df)

    if not dfs_to_concat:
        log.error("No data found for trait %s. Skipping...", args.trait)
        return

    # Concatenate all dataframes
    combined_df = pd.concat(dfs_to_concat, ignore_index=True)

    log.info("Total observations for trait %s: %d", args.trait, len(combined_df))

    # Write to parquet
    log.info("Writing to %s...", out_fn)
    combined_df.to_parquet(out_fn, compression="zstd", index=False)

    log.info("✓ Completed processing trait %s", args.trait)


def load_trait_map(y_set: str, trait_fn: Path, trait_stat: int) -> pd.DataFrame | None:
    """
    Load a trait map and convert to dataframe.

    Args:
        y_set: Dataset name ('gbif' or 'splot')
        trait: Trait name
        trait_stat: Band number to extract (1-indexed)
        cfg: Configuration object

    Returns:
        DataFrame with columns: x, y, {trait}, source
        None if file doesn't exist
    """
    if not trait_fn.exists():
        log.warning("Trait map not found: %s", trait_fn.absolute())
        return None

    log.info("Loading %s map: %s", y_set.upper(), trait_fn.absolute())

    # Load the raster and extract the specified band
    ds = open_raster(trait_fn)

    # Select the band corresponding to the trait statistic
    da = ds.sel(band=trait_stat)

    # Convert to dataframe
    df = da.to_dataframe(name=trait_fn.stem).reset_index()

    # Drop band and spatial_ref columns if they exist
    cols_to_drop = ["band", "spatial_ref"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Drop NaN values
    df = df.dropna(subset=[trait_fn.stem])

    # Add source column ('g' for GBIF, 's' for sPlot)
    source_code = "g" if y_set == "gbif" else "s"
    df["source"] = source_code

    log.info(
        "Loaded %d observations from %s for trait %s",
        len(df),
        y_set.upper(),
        trait_fn.stem,
    )

    return df


if __name__ == "__main__":
    main()
