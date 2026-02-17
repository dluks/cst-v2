"""Build training data by merging histogram targets with EO features.

This module combines:
1. GBIF histogram targets (Zarr: cell-level trait distributions)
2. sPlot histogram targets (Zarr: cell-level trait distributions)
3. EO features (parquet: Earth observation predictors)

The output is a single Zarr store ready for histogram-based trait
distribution modeling with PyTorch.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from src.conf.conf import get_config

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


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_histogram_source(
    source_dir: Path,
    source_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict] | None:
    """Load histogram data from a Zarr store.

    Parameters
    ----------
    source_dir : Path
        Directory containing ``histograms.zarr``.
    source_name : str
        Name of the source ('gbif' or 'splot').

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, dict] | None
        (histograms, masks, coords, attrs) or None if not found.
    """
    zarr_path = source_dir / "histograms.zarr"
    if not zarr_path.exists():
        log.warning("Zarr store not found: %s", zarr_path)
        return None

    log.info("Loading %s histograms from %s", source_name, zarr_path)
    root = zarr.open_group(zarr_path, mode="r")

    histograms = np.asarray(root["histograms"])
    masks = np.asarray(root["masks"])
    coords = np.asarray(root["coords"])
    attrs = dict(root.attrs)

    log.info(
        "Loaded %s: %d cells, %d traits, %d bins",
        source_name,
        histograms.shape[0],
        histograms.shape[1],
        histograms.shape[2],
    )

    return histograms, masks, coords, attrs


def _load_eo_features(x_fp: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load EO features from parquet file.

    Parameters
    ----------
    x_fp : Path
        Path to X.parquet file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        (feature_values (N, F), coords (N, 2), feature_names).
    """
    log.info("Loading EO features from %s", x_fp)
    x_df = pd.read_parquet(x_fp)

    # Handle MultiIndex (y, x) -> regular columns
    if isinstance(x_df.index, pd.MultiIndex):
        x_df = x_df.reset_index()

    feature_names = [c for c in x_df.columns if c not in ("x", "y")]
    x_coords = x_df[["x", "y"]].values.astype(np.float64)
    x_values = x_df[feature_names].values.astype(np.float32)

    log.info("Loaded EO features: %d cells, %d features", len(x_df), len(feature_names))
    return x_values, x_coords, feature_names


# ---------------------------------------------------------------------------
# Combining / merging
# ---------------------------------------------------------------------------

def _combine_sources(
    gbif_data: tuple[np.ndarray, np.ndarray, np.ndarray, dict] | None,
    splot_data: tuple[np.ndarray, np.ndarray, np.ndarray, dict] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Combine GBIF and sPlot histogram data.

    Parameters
    ----------
    gbif_data : tuple | None
        GBIF (histograms, masks, coords, attrs).
    splot_data : tuple | None
        sPlot (histograms, masks, coords, attrs).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]
        (histograms, masks, coords, source_ids, attrs).
        source_ids: 0 = gbif, 1 = splot.

    Raises
    ------
    ValueError
        If no data sources are available.
    """
    if gbif_data is None and splot_data is None:
        raise ValueError("No histogram data available from either GBIF or sPlot")

    if gbif_data is None:
        hist, mask, coord, attrs = splot_data  # type: ignore[misc]
        source_ids = np.ones(hist.shape[0], dtype=np.int8)
        return hist, mask, coord, source_ids, attrs

    if splot_data is None:
        hist, mask, coord, attrs = gbif_data
        source_ids = np.zeros(hist.shape[0], dtype=np.int8)
        return hist, mask, coord, source_ids, attrs

    # Both available
    g_hist, g_mask, g_coord, g_attrs = gbif_data
    s_hist, s_mask, s_coord, s_attrs = splot_data

    hist = np.concatenate([g_hist, s_hist], axis=0)
    mask = np.concatenate([g_mask, s_mask], axis=0)
    coord = np.concatenate([g_coord, s_coord], axis=0)
    source_ids = np.concatenate([
        np.zeros(g_hist.shape[0], dtype=np.int8),
        np.ones(s_hist.shape[0], dtype=np.int8),
    ])

    attrs = g_attrs.copy()
    attrs["sources"] = ["gbif", "splot"]
    attrs["gbif_cells"] = int(g_hist.shape[0])
    attrs["splot_cells"] = int(s_hist.shape[0])

    log.info(
        "Combined: %d GBIF + %d sPlot = %d total cells",
        g_hist.shape[0],
        s_hist.shape[0],
        hist.shape[0],
    )

    return hist, mask, coord, source_ids, attrs


def _merge_with_features(
    hist: np.ndarray,
    mask: np.ndarray,
    hist_coords: np.ndarray,
    source_ids: np.ndarray,
    x_values: np.ndarray,
    x_coords: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge histogram data with EO features on grid cell coordinates.

    Snaps EO feature pixel-center coordinates to the same grid used by
    the histogram cells, then matches on the resulting (cell_x, cell_y).
    Only cells present in both datasets are kept.

    Parameters
    ----------
    hist : np.ndarray
        Histogram data (N_hist, n_traits, n_bins).
    mask : np.ndarray
        Validity masks (N_hist, n_traits).
    hist_coords : np.ndarray
        Histogram cell coordinates (N_hist, 2), already grid-snapped.
    source_ids : np.ndarray
        Source indicator per cell (N_hist,).
    x_values : np.ndarray
        EO feature values (N_eo, n_features).
    x_coords : np.ndarray
        EO feature pixel-center coordinates (N_eo, 2).
    resolution : int
        Grid cell resolution in meters (e.g. 22000).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Aligned (histograms, masks, features, coords, source_ids).
    """
    log.info("Merging histogram targets with EO features...")

    # Snap EO coordinates to the same grid as histogram cells
    eo_cell_x = (x_coords[:, 0] // resolution) * resolution
    eo_cell_y = (x_coords[:, 1] // resolution) * resolution

    # Build lookup from snapped (cell_x, cell_y) -> EO row index
    eo_keys: dict[tuple[float, float], int] = {}
    for i in range(x_coords.shape[0]):
        eo_keys[(eo_cell_x[i], eo_cell_y[i])] = i

    # Find matching indices
    hist_idx = []
    eo_idx = []
    for i in range(hist_coords.shape[0]):
        key = (hist_coords[i, 0], hist_coords[i, 1])
        if key in eo_keys:
            hist_idx.append(i)
            eo_idx.append(eo_keys[key])

    hist_idx = np.array(hist_idx)
    eo_idx = np.array(eo_idx)

    n_before = hist.shape[0]
    n_after = len(hist_idx)
    n_dropped = n_before - n_after

    if n_dropped > 0:
        log.warning(
            "Dropped %d cells (%.1f%%) without EO features",
            n_dropped,
            100 * n_dropped / n_before,
        )

    log.info(
        "Merged: %d cells with both histogram targets and EO features",
        n_after,
    )

    return (
        hist[hist_idx],
        mask[hist_idx],
        x_values[eo_idx],
        hist_coords[hist_idx],
        source_ids[hist_idx],
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save_outputs(
    out_path: Path,
    hist: np.ndarray,
    mask: np.ndarray,
    features: np.ndarray,
    coords: np.ndarray,
    source_ids: np.ndarray,
    feature_names: list[str],
    attrs: dict,
) -> None:
    """Save merged training data as a Zarr store.

    Parameters
    ----------
    out_path : Path
        Path for the output Zarr store.
    hist : np.ndarray
        Histogram targets (N, n_traits, n_bins).
    mask : np.ndarray
        Validity masks (N, n_traits).
    features : np.ndarray
        EO features (N, n_features).
    coords : np.ndarray
        Cell coordinates (N, 2).
    source_ids : np.ndarray
        Source indicator per cell (N,).
    feature_names : list[str]
        EO feature column names.
    attrs : dict
        Metadata attributes.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(out_path, mode="w")

    root.create_array("Y_hist", data=hist)
    root.create_array("Y_mask", data=mask)
    root.create_array("X", data=features)
    root.create_array("coords", data=coords)
    root.create_array("source", data=source_ids)

    # Store metadata
    root.attrs["n_cells"] = int(hist.shape[0])
    root.attrs["n_traits"] = int(hist.shape[1])
    root.attrs["n_bins"] = int(hist.shape[2])
    root.attrs["n_features"] = int(features.shape[1])
    root.attrs["feature_names"] = feature_names
    root.attrs["trait_names"] = attrs.get("trait_names", [])
    root.attrs["crs"] = attrs.get("crs", "")
    root.attrs["target_resolution"] = attrs.get("target_resolution", 0)
    root.attrs["label_smoothing_epsilon"] = attrs.get(
        "label_smoothing_epsilon", 0.0
    )
    root.attrs["source_encoding"] = {"gbif": 0, "splot": 1}

    log.info("Saved training Zarr store to %s", out_path)
    log.info(
        "  Y_hist: %s %s | Y_mask: %s %s | X: %s %s | coords: %s %s",
        hist.shape, hist.dtype,
        mask.shape, mask.dtype,
        features.shape, features.dtype,
        coords.shape, coords.dtype,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace | None = None) -> None:
    """Main function to merge histogram targets with EO features."""
    args = cli() if args is None else args
    cfg = get_config(params_path=args.params)

    proj_root = os.environ.get("PROJECT_ROOT")
    if proj_root is None:
        raise ValueError("PROJECT_ROOT environment variable is not set")
    proj_root = Path(proj_root)

    # Output path
    out_path = proj_root / cfg.output.xy_dir / "train.zarr"

    if out_path.exists() and not args.overwrite:
        log.info("Output already exists: %s. Use --overwrite to regenerate.", out_path)
        return

    # Load histogram sources
    hist_dir = proj_root / cfg.output.dir

    gbif_data = _load_histogram_source(hist_dir / "gbif", "gbif")
    splot_data = _load_histogram_source(hist_dir / "splot", "splot")

    # Combine sources
    hist, mask, coords, source_ids, attrs = _combine_sources(
        gbif_data, splot_data
    )

    # Load EO features
    x_fp = proj_root / cfg.eo_features.x_fp
    x_values, x_coords, feature_names = _load_eo_features(x_fp)

    # Merge
    hist, mask, features, coords, source_ids = _merge_with_features(
        hist, mask, coords, source_ids, x_values, x_coords,
        resolution=cfg.target_resolution,
    )

    # Save
    _save_outputs(
        out_path, hist, mask, features, coords, source_ids, feature_names, attrs
    )

    log.info("Done — %d training cells written to %s", hist.shape[0], out_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
