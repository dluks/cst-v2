"""Build training data by merging histogram targets with EO features.

This module combines:
1. GBIF histogram targets (cell-level trait distributions)
2. sPlot histogram targets (cell-level trait distributions)
3. EO features (Earth observation predictors)

The output is a merged dataset ready for histogram-based trait distribution modeling.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import get_config

log = logging.getLogger(__name__)


def cli(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Parameters
    ----------
    args : list[str] | None
        Command line arguments. If None, uses sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Merge histogram targets with EO features for training"
    )
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to params.yaml file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser.parse_args(args)


def _load_histogram_source(
    source_dir: Path,
    source_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict] | None:
    """Load histogram data from a source directory.

    Parameters
    ----------
    source_dir : Path
        Directory containing histogram outputs.
    source_name : str
        Name of the source ('gbif' or 'splot').

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict] | None
        Tuple of (histograms_df, masks_df, coords_df, metadata) or None if not found.
    """
    if not source_dir.exists():
        log.warning("Source directory not found: %s", source_dir)
        return None

    hist_fp = source_dir / "histograms.parquet"
    masks_fp = source_dir / "masks.parquet"
    coords_fp = source_dir / "coordinates.parquet"
    meta_fp = source_dir / "metadata.json"

    if not all(fp.exists() for fp in [hist_fp, masks_fp, coords_fp, meta_fp]):
        log.warning("Missing files in %s source directory: %s", source_name, source_dir)
        return None

    log.info("Loading %s histograms from %s", source_name, source_dir)

    histograms_df = pd.read_parquet(hist_fp)
    masks_df = pd.read_parquet(masks_fp)
    coords_df = pd.read_parquet(coords_fp)

    with open(meta_fp) as f:
        metadata = json.load(f)

    log.info(
        "Loaded %s: %d cells, %d traits",
        source_name,
        len(histograms_df),
        len(metadata["trait_names"]),
    )

    return histograms_df, masks_df, coords_df, metadata


def _load_eo_features(x_fp: Path) -> pd.DataFrame:
    """Load EO features from parquet file.

    Parameters
    ----------
    x_fp : Path
        Path to X.parquet file.

    Returns
    -------
    pd.DataFrame
        EO features with x, y columns.
    """
    log.info("Loading EO features from %s", x_fp)
    x_df = pd.read_parquet(x_fp)

    # Handle MultiIndex (y, x) -> regular columns
    if isinstance(x_df.index, pd.MultiIndex):
        x_df = x_df.reset_index()

    log.info("Loaded EO features: %d cells, %d features", len(x_df), len(x_df.columns))
    return x_df


def _combine_sources(
    gbif_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict] | None,
    splot_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Combine GBIF and sPlot histogram data.

    For cells present in both sources, uses average of histograms weighted by
    the number of underlying observations/abundance.

    Parameters
    ----------
    gbif_data : tuple | None
        GBIF histogram data tuple.
    splot_data : tuple | None
        sPlot histogram data tuple.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]
        Combined (histograms_df, masks_df, coords_df, metadata).

    Raises
    ------
    ValueError
        If no data sources are available.
    """
    if gbif_data is None and splot_data is None:
        raise ValueError("No histogram data available from either GBIF or sPlot")

    # Single source case
    if gbif_data is None:
        log.info("Using sPlot data only")
        hist_df, mask_df, coord_df, meta = splot_data  # type: ignore[misc]
        hist_df["source"] = "splot"
        mask_df["source"] = "splot"
        coord_df["source"] = "splot"
        return hist_df, mask_df, coord_df, meta

    if splot_data is None:
        log.info("Using GBIF data only")
        hist_df, mask_df, coord_df, meta = gbif_data
        hist_df["source"] = "gbif"
        mask_df["source"] = "gbif"
        coord_df["source"] = "gbif"
        return hist_df, mask_df, coord_df, meta

    # Both sources available - combine them
    gbif_hist, gbif_mask, gbif_coord, gbif_meta = gbif_data
    splot_hist, splot_mask, splot_coord, splot_meta = splot_data

    # Verify consistent bin edges
    for trait in gbif_meta["trait_names"]:
        if trait in splot_meta["trait_names"]:
            gbif_edges = np.array(gbif_meta["bin_edges"][trait])
            splot_edges = np.array(splot_meta["bin_edges"][trait])
            if not np.allclose(gbif_edges, splot_edges, rtol=1e-5):
                raise ValueError(f"Inconsistent bin edges for trait {trait}")

    log.info("Combining GBIF and sPlot data...")

    # Add source column
    gbif_hist["source"] = "gbif"
    gbif_mask["source"] = "gbif"
    gbif_coord["source"] = "gbif"

    splot_hist["source"] = "splot"
    splot_mask["source"] = "splot"
    splot_coord["source"] = "splot"

    # Concatenate (keep separate rows for cells in both sources)
    combined_hist = pd.concat([gbif_hist, splot_hist], ignore_index=True)
    combined_mask = pd.concat([gbif_mask, splot_mask], ignore_index=True)
    combined_coord = pd.concat([gbif_coord, splot_coord], ignore_index=True)

    # Combine metadata
    combined_meta = gbif_meta.copy()
    combined_meta["sources"] = ["gbif", "splot"]
    combined_meta["gbif_cells"] = len(gbif_hist)
    combined_meta["splot_cells"] = len(splot_hist)

    log.info(
        "Combined data: %d GBIF cells + %d sPlot cells = %d total rows",
        len(gbif_hist),
        len(splot_hist),
        len(combined_hist),
    )

    return combined_hist, combined_mask, combined_coord, combined_meta


def _merge_with_features(
    histograms_df: pd.DataFrame,
    masks_df: pd.DataFrame,
    coords_df: pd.DataFrame,
    x_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge histogram data with EO features on coordinates.

    Parameters
    ----------
    histograms_df : pd.DataFrame
        Histogram data with cell_id index.
    masks_df : pd.DataFrame
        Valid trait masks with cell_id index.
    coords_df : pd.DataFrame
        Cell coordinates with cell_id index and x, y columns.
    x_df : pd.DataFrame
        EO features with x, y columns.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Merged (histograms, masks, features) DataFrames with aligned indices.
    """
    log.info("Merging histogram data with EO features on coordinates...")

    # Get x, y from coordinates
    if "cell_id" in coords_df.columns:
        coords_with_xy = coords_df.set_index("cell_id")
    else:
        coords_with_xy = coords_df.copy()

    # Ensure histograms and masks have cell_id index
    if "cell_id" in histograms_df.columns:
        histograms_df = histograms_df.set_index("cell_id")
    if "cell_id" in masks_df.columns:
        masks_df = masks_df.set_index("cell_id")

    # Add x, y to histograms for merge
    hist_with_coords = histograms_df.join(coords_with_xy[["x", "y"]])

    # Create merge key in x_df
    x_df = x_df.copy()

    # Merge on (x, y)
    # Use inner join to only keep cells with both histograms and EO features
    merged = hist_with_coords.reset_index().merge(
        x_df,
        on=["x", "y"],
        how="inner",
    )

    n_before = len(histograms_df)
    n_after = len(merged)
    n_dropped = n_before - n_after

    if n_dropped > 0:
        log.warning(
            "Dropped %d cells (%.1f%%) without EO features",
            n_dropped,
            100 * n_dropped / n_before,
        )

    log.info(
        "Merged data: %d cells with both histogram targets and EO features",
        n_after,
    )

    # Split back into components
    # Extract histogram columns (those ending in _binN pattern)
    hist_cols = [c for c in merged.columns if "_bin" in c]
    coord_cols = ["x", "y"]
    source_col = ["source"] if "source" in merged.columns else []
    feature_cols = [
        c
        for c in merged.columns
        if c not in hist_cols + coord_cols + source_col + ["cell_id"]
    ]

    # Rebuild dataframes with consistent index
    if "cell_id" in merged.columns:
        merged = merged.set_index("cell_id")

    # Get mask columns aligned
    mask_cols = [c for c in masks_df.columns if c not in ["source"]]
    masks_aligned = masks_df.loc[merged.index, mask_cols].copy()
    if "source" in merged.columns:
        masks_aligned["source"] = merged["source"]

    merged_hist = merged[hist_cols + source_col].copy()
    merged_features = merged[coord_cols + feature_cols].copy()

    return merged_hist, masks_aligned, merged_features


def _save_outputs(
    out_dir: Path,
    histograms_df: pd.DataFrame,
    masks_df: pd.DataFrame,
    features_df: pd.DataFrame,
    metadata: dict,
) -> None:
    """Save merged training data to disk.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    histograms_df : pd.DataFrame
        Merged histogram data.
    masks_df : pd.DataFrame
        Merged validity masks.
    features_df : pd.DataFrame
        Merged EO features with coordinates.
    metadata : dict
        Combined metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save histograms (Y targets)
    hist_fp = out_dir / "Y_histograms.parquet"
    histograms_df.to_parquet(hist_fp, compression="zstd")
    log.info("Saved histogram targets to %s", hist_fp)

    # Save masks
    masks_fp = out_dir / "Y_masks.parquet"
    masks_df.to_parquet(masks_fp, compression="zstd")
    log.info("Saved validity masks to %s", masks_fp)

    # Save features (X)
    features_fp = out_dir / "X_features.parquet"
    features_df.to_parquet(features_fp, compression="zstd")
    log.info("Saved EO features to %s", features_fp)

    # Save metadata
    metadata["n_cells"] = len(histograms_df)
    metadata["n_features"] = len(
        [c for c in features_df.columns if c not in ["x", "y"]]
    )

    meta_fp = out_dir / "metadata.json"
    with open(meta_fp, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved metadata to %s", meta_fp)


def main(args: argparse.Namespace | None = None) -> None:
    """Main function to merge histogram targets with EO features.

    Parameters
    ----------
    args : argparse.Namespace | None
        Parsed command line arguments.
    """
    args = cli() if args is None else args
    cfg = get_config(params_path=args.params)

    # Set up paths
    proj_root = os.environ.get("PROJECT_ROOT")
    if proj_root is None:
        raise ValueError("PROJECT_ROOT environment variable is not set")
    proj_root = Path(proj_root)

    # Output directory
    out_dir = proj_root / cfg.output.xy_dir
    out_fp = out_dir / "X_features.parquet"

    if out_fp.exists() and not args.overwrite:
        log.info("Output already exists: %s. Use --overwrite to regenerate.", out_fp)
        return

    # Load histogram sources
    hist_dir = proj_root / cfg.output.dir

    gbif_dir = hist_dir / "gbif"
    splot_dir = hist_dir / "splot"

    gbif_data = _load_histogram_source(gbif_dir, "gbif")
    splot_data = _load_histogram_source(splot_dir, "splot")

    # Combine sources
    histograms_df, masks_df, coords_df, metadata = _combine_sources(
        gbif_data, splot_data
    )

    # Load EO features
    x_fp = proj_root / cfg.eo_features.x_fp
    x_df = _load_eo_features(x_fp)

    # Merge histogram targets with EO features
    merged_hist, merged_masks, merged_features = _merge_with_features(
        histograms_df, masks_df, coords_df, x_df
    )

    # Save outputs
    _save_outputs(out_dir, merged_hist, merged_masks, merged_features, metadata)

    log.info("✓ Successfully built histogram XY training data")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
