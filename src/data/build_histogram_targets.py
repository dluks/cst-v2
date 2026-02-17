"""Build histogram targets from GBIF or sPlot observations.

This module constructs probability histograms for each trait at the grid cell level,
using either GBIF occurrence data or sPlot vegetation survey data.

The histograms are built with global bin edges (computed from the full species pool)
to ensure spatial comparability across all grid cells.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyproj
import zarr

from src.conf.conf import get_config

log = logging.getLogger(__name__)

# Column name mappings for different data sources
GBIF_SPECIES_COL = "specieskey"
GBIF_COORDS = ("decimallatitude", "decimallongitude")
SPLOT_SPECIES_COL = "speciesname"
SPLOT_COORDS = ("Latitude", "Longitude")
PFT_COL = "pft"


def cli() -> argparse.Namespace:
    """Command-line interface for histogram target construction."""
    parser = argparse.ArgumentParser(
        description="Build histogram targets from GBIF or sPlot observations."
    )
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to the params.yaml configuration file.",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["gbif", "splot"],
        required=True,
        help="Data source: 'gbif' or 'splot'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Main function to build histogram targets."""
    args = cli() if args is None else args
    cfg = get_config(params_path=args.params)

    # Get project root for resolving paths
    proj_root = os.environ.get("PROJECT_ROOT")
    if proj_root is None:
        raise ValueError("PROJECT_ROOT environment variable is not set")
    proj_root = Path(proj_root)

    source = args.source
    log.info(
        "=== Building histogram targets from %s data ===", source.upper()
    )
    log.info(
        "Config: n_bins=%d, epsilon=%.3f, resolution=%dm, CRS=%s",
        cfg.histogram.n_bins,
        cfg.histogram.label_smoothing_epsilon,
        cfg.target_resolution,
        cfg.crs,
    )

    # Output directory
    out_dir = proj_root / cfg.output.dir / source
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if outputs already exist
    zarr_fp = out_dir / "histograms.zarr"
    if zarr_fp.exists() and not args.overwrite:
        log.info("Output already exists at %s. Use --overwrite to replace.", zarr_fp)
        return

    # Load trait data (species-level, transformed)
    traits_fp = proj_root / cfg.traits.interim_out
    log.info("[1/4] Loading trait data from %s", traits_fp)
    traits_df = pd.read_parquet(traits_fp)
    trait_names = cfg.traits.names
    log.info("  Loaded %d species with %d traits", len(traits_df), len(trait_names))

    # Compute global bin edges for each trait
    log.info("[2/4] Computing global bin edges (%d bins)...", cfg.histogram.n_bins)
    bin_edges = _compute_bin_edges(traits_df, trait_names, cfg.histogram.n_bins)
    log.info(
        "  Bin edges computed for %d / %d traits", len(bin_edges), len(trait_names)
    )

    # Load and process data based on source
    log.info("[3/4] Processing %s observations...", source.upper())
    if source == "gbif":
        hist_arr, mask_arr, coords_arr, stats = _process_gbif(
            cfg, traits_df, trait_names, bin_edges, proj_root
        )
    else:
        hist_arr, mask_arr, coords_arr, stats = _process_splot(
            cfg, traits_df, trait_names, bin_edges, proj_root
        )

    # Save outputs
    log.info("[4/4] Saving outputs to %s", out_dir)
    _save_outputs(
        out_dir=out_dir,
        histograms=hist_arr,
        masks=mask_arr,
        coords=coords_arr,
        bin_edges=bin_edges,
        trait_names=trait_names,
        stats=stats,
        cfg=cfg,
        source=source,
    )

    # Generate sanity-check report
    from src.data.histogram_report import generate_histogram_report

    generate_histogram_report(zarr_fp, out_dir, params_path=Path(args.params))

    log.info(
        "=== Done: %d valid cells, histograms shape %s ===",
        stats["n_cells_valid"],
        hist_arr.shape,
    )


def _compute_bin_edges(
    traits_df: pd.DataFrame,
    trait_names: list[str],
    n_bins: int,
) -> dict[str, np.ndarray]:
    """Compute global bin edges for each trait from the full species pool.

    Parameters
    ----------
    traits_df : pd.DataFrame
        Species-level trait data with transformed values.
    trait_names : list[str]
        List of trait column names.
    n_bins : int
        Number of bins for each trait's histogram.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping trait names to bin edge arrays (length n_bins + 1).
    """
    bin_edges = {}
    for trait in trait_names:
        values = traits_df[trait].dropna()
        if len(values) < 2:
            log.warning("Trait %s has fewer than 2 non-NaN values, skipping", trait)
            continue
        edges = np.linspace(values.min(), values.max(), n_bins + 1)
        bin_edges[trait] = edges
        log.debug(
            "Trait %s: min=%.4f, max=%.4f, %d bins",
            trait,
            values.min(),
            values.max(),
            n_bins,
        )
    return bin_edges


def _process_gbif(
    cfg,
    traits_df: pd.DataFrame,
    trait_names: list[str],
    bin_edges: dict[str, np.ndarray],
    proj_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Process GBIF observations to construct histograms.

    Parameters
    ----------
    cfg : OmegaConf
        Configuration object.
    traits_df : pd.DataFrame
        Species-level trait data.
    trait_names : list[str]
        List of trait column names.
    bin_edges : dict[str, np.ndarray]
        Pre-computed bin edges for each trait.
    proj_root : Path
        Project root directory for resolving paths.

    Returns
    -------
    tuple
        (histograms_df, masks_df, coords_df, stats)
    """
    # Load GBIF data
    gbif_path = proj_root / Path(
        cfg.gbif.filtered.out_dir, cfg.trait_type, cfg.gbif.filtered.fp
    )
    log.info("Loading GBIF data from %s", gbif_path)
    gbif = dd.read_parquet(gbif_path)

    # Keep only necessary columns
    gbif_cols = [GBIF_SPECIES_COL, *GBIF_COORDS, "weight"]
    if cfg.PFT:
        gbif_cols.append(PFT_COL)
        gbif = gbif[gbif[PFT_COL].isin(cfg.PFT.split("_"))]
    gbif = gbif[gbif_cols].compute()
    log.info("Loaded %d GBIF observations", len(gbif))

    # Reproject and assign cells BEFORE trait merge to avoid 292M × 35col blowup
    log.info("Reprojecting coordinates to %s...", cfg.crs)
    gbif = _reproject(
        gbif,
        lat_col=GBIF_COORDS[0],
        lon_col=GBIF_COORDS[1],
        target_crs=cfg.crs,
    )

    log.info("Assigning grid cell IDs at %dm resolution...", cfg.target_resolution)
    gbif = _assign_cell_ids(gbif, cfg.target_resolution)

    # Pre-aggregate weights by (cell_id, species) — each species has a single
    # trait value, so summing weights is mathematically equivalent to the
    # per-observation histogram and reduces ~270M rows to a few million.
    log.info("Pre-aggregating weights by (cell_id, species)...")
    agg = (
        gbif.groupby(["cell_id", "cell_x", "cell_y", GBIF_SPECIES_COL])["weight"]
        .sum()
        .reset_index()
    )
    n_before, n_after = len(gbif), len(agg)
    log.info(
        "Pre-aggregated %d observations → %d (cell, species) pairs (%.1fx reduction)",
        n_before,
        n_after,
        n_before / n_after if n_after > 0 else 0,
    )
    del gbif

    # NOW merge with traits at the (cell, species) level
    log.info("Joining with trait data...")
    merged = agg.merge(
        traits_df[["GBIFKeyGBIF", *trait_names]],
        left_on=GBIF_SPECIES_COL,
        right_on="GBIFKeyGBIF",
        how="inner",
    ).drop(columns=["GBIFKeyGBIF"])
    del agg
    log.info("Merged data has %d (cell, species) pairs", len(merged))

    # Build histograms per cell
    log.info("Building histograms per grid cell...")
    histograms_df, masks_df, coords_df, stats = _build_cell_histograms(
        df=merged,
        trait_names=trait_names,
        bin_edges=bin_edges,
        n_bins=cfg.histogram.n_bins,
        min_observations=cfg.gbif.min_observations,
        min_unique_species=cfg.gbif.min_unique_species,
        min_bin_coverage=cfg.gbif.min_bin_coverage,
        epsilon=cfg.histogram.label_smoothing_epsilon,
        weight_col="weight",
        species_col=GBIF_SPECIES_COL,
    )

    return histograms_df, masks_df, coords_df, stats


def _process_splot(
    cfg,
    traits_df: pd.DataFrame,
    trait_names: list[str],
    bin_edges: dict[str, np.ndarray],
    proj_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Process sPlot surveys to construct histograms.

    Parameters
    ----------
    cfg : OmegaConf
        Configuration object.
    traits_df : pd.DataFrame
        Species-level trait data.
    trait_names : list[str]
        List of trait column names.
    bin_edges : dict[str, np.ndarray]
        Pre-computed bin edges for each trait.
    proj_root : Path
        Project root directory for resolving paths.

    Returns
    -------
    tuple
        (histograms_df, masks_df, coords_df, stats)
    """
    # Load sPlot data
    splot_path = proj_root / Path(
        cfg.splot.filtered.out_dir, cfg.trait_type, cfg.splot.filtered.fp
    )
    log.info("Loading sPlot data from %s", splot_path)
    splot = pd.read_parquet(splot_path)

    # Filter by PFT if specified
    if cfg.PFT:
        splot = splot[splot[PFT_COL].isin(cfg.PFT.split("_"))]

    log.info("Loaded %d sPlot observations", len(splot))

    # Join with traits
    log.info("Joining sPlot with trait data...")
    merged = splot.merge(
        traits_df[["nameOutWCVP", *trait_names]],
        left_on=SPLOT_SPECIES_COL,
        right_on="nameOutWCVP",
        how="inner",
    ).drop(columns=["nameOutWCVP"])
    log.info("Merged data has %d observations", len(merged))

    # Reproject coordinates
    log.info("Reprojecting coordinates to %s...", cfg.crs)
    merged = _reproject(
        merged,
        lat_col=SPLOT_COORDS[0],
        lon_col=SPLOT_COORDS[1],
        target_crs=cfg.crs,
    )

    # Assign grid cell IDs
    log.info("Assigning grid cell IDs at %dm resolution...", cfg.target_resolution)
    merged = _assign_cell_ids(merged, cfg.target_resolution)

    # Compute combined weights: abundance × resurvey_weight
    # For sPlot, the histogram weight is relative abundance times survey weight
    merged["combined_weight"] = merged["Rel_Abund_Plot"] * merged["weight"]

    # Build histograms per cell
    log.info("Building histograms per grid cell...")
    histograms_df, masks_df, coords_df, stats = _build_cell_histograms(
        df=merged,
        trait_names=trait_names,
        bin_edges=bin_edges,
        n_bins=cfg.histogram.n_bins,
        min_observations=None,  # sPlot uses abundance threshold instead
        min_unique_species=cfg.splot.min_unique_species,
        min_bin_coverage=cfg.splot.min_bin_coverage,
        epsilon=cfg.histogram.label_smoothing_epsilon,
        weight_col="combined_weight",
        species_col=SPLOT_SPECIES_COL,
        min_total_abundance=cfg.splot.min_total_abundance,
    )

    return histograms_df, masks_df, coords_df, stats


def _reproject(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    target_crs: str,
) -> pd.DataFrame:
    """Reproject coordinates from WGS84 to target CRS.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with lat/lon columns.
    lat_col : str
        Name of latitude column.
    lon_col : str
        Name of longitude column.
    target_crs : str
        Target CRS string (e.g., "EPSG:6933").

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'x' and 'y' columns in target CRS.
    """
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", target_crs, always_xy=True
    )
    x, y = transformer.transform(df[lon_col].values, df[lat_col].values)
    df["x"] = x
    df["y"] = y
    # Drop original lat/lon to free memory
    df = df.drop(columns=[lat_col, lon_col])
    return df


def _assign_cell_ids(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """Assign grid cell IDs based on coordinates and resolution.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'x' and 'y' columns.
    resolution : int
        Grid cell size in meters.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'cell_x', 'cell_y', and 'cell_id' columns.
    """
    # Compute integer cell indices (floor division)
    cx = (df["x"].values // resolution).astype(np.int64)
    cy = (df["y"].values // resolution).astype(np.int64)
    # Cell origin coordinates (integer multiples of resolution)
    df["cell_x"] = cx * resolution
    df["cell_y"] = cy * resolution
    # Integer cell ID for fast groupby (avoids expensive string creation)
    # Safe as long as |cy / resolution| < 10M, which holds for any Earth CRS
    df["cell_id"] = cx * 10_000_000 + cy
    return df


def _build_cell_histograms(
    df: pd.DataFrame,
    trait_names: list[str],
    bin_edges: dict[str, np.ndarray],
    n_bins: int,
    min_observations: int | None,
    min_unique_species: int,
    min_bin_coverage: float,
    epsilon: float,
    weight_col: str,
    species_col: str,
    min_total_abundance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build histograms for each grid cell.

    Parameters
    ----------
    df : pd.DataFrame
        Merged observation data with traits, coordinates, and weights.
    trait_names : list[str]
        List of trait column names.
    bin_edges : dict[str, np.ndarray]
        Pre-computed bin edges for each trait.
    n_bins : int
        Number of bins per histogram.
    min_observations : int | None
        Minimum weighted observations per cell (GBIF).
    min_unique_species : int
        Minimum unique species per cell.
    min_bin_coverage : float
        Minimum fraction of bins that must be non-empty.
    epsilon : float
        Label smoothing parameter.
    weight_col : str
        Name of weight column.
    species_col : str
        Name of species column.
    min_total_abundance : float | None
        Minimum total abundance per cell (sPlot).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, dict]
        (histograms (N, n_traits, n_bins), masks (N, n_traits),
         coords (N, 2), stats)
    """
    # ------------------------------------------------------------------
    # Step 1: Cell-level filtering (single groupby, no Python loop)
    # ------------------------------------------------------------------
    agg_spec: dict = {
        "cell_x": ("cell_x", "first"),
        "cell_y": ("cell_y", "first"),
        "total_weight": (weight_col, "sum"),
        "n_species": (species_col, "nunique"),
    }
    if min_total_abundance is not None:
        agg_spec["total_abundance"] = ("Rel_Abund_Plot", "sum")
        agg_spec["n_plots"] = ("PlotObservationID", "nunique")

    cell_stats = df.groupby("cell_id").agg(**agg_spec)
    n_cells_total = len(cell_stats)
    log.info("Processing %d unique grid cells...", n_cells_total)

    # Apply filters in the same order as the original (obs → abundance → species)
    # so that filter statistics are mutually exclusive.
    remaining = np.ones(n_cells_total, dtype=bool)

    if min_observations is not None:
        fail_obs = cell_stats["total_weight"] < min_observations
        cells_filtered_obs = int((remaining & fail_obs).sum())
        remaining &= ~fail_obs
    else:
        cells_filtered_obs = 0

    if min_total_abundance is not None:
        avg_abund = (
            cell_stats["total_abundance"]
            / cell_stats["n_plots"].clip(lower=1)
        )
        fail_abund = avg_abund < min_total_abundance
        cells_filtered_abundance = int((remaining & fail_abund).sum())
        remaining &= ~fail_abund
    else:
        cells_filtered_abundance = 0

    fail_species = cell_stats["n_species"] < min_unique_species
    cells_filtered_species = int((remaining & fail_species).sum())
    remaining &= ~fail_species

    valid_cells = cell_stats[remaining]
    log.info(
        "Cell filtering: %d passed (obs: -%d, abundance: -%d, species: -%d)",
        len(valid_cells),
        cells_filtered_obs,
        cells_filtered_abundance,
        cells_filtered_species,
    )

    if len(valid_cells) == 0:
        n_traits = len(trait_names)
        return (
            np.empty((0, n_traits, n_bins), dtype=np.float32),
            np.empty((0, n_traits), dtype=bool),
            np.empty((0, 2), dtype=np.float64),
            {
                "n_cells_total": n_cells_total,
                "n_cells_valid": 0,
                "cells_filtered_observations": cells_filtered_obs,
                "cells_filtered_species": cells_filtered_species,
                "cells_filtered_abundance": cells_filtered_abundance,
                "trait_valid_counts": {t: 0 for t in trait_names},
            },
        )

    # ------------------------------------------------------------------
    # Step 2: Vectorized histogram construction per trait
    # ------------------------------------------------------------------
    # Restrict to valid cells and encode cell_id as integer codes
    log.info("Filtering observations to %d valid cells...", len(valid_cells))
    df_valid = df.loc[df["cell_id"].isin(valid_cells.index)]
    log.info("  %d observations in valid cells", len(df_valid))

    cell_id_cat = pd.Categorical(df_valid["cell_id"], categories=valid_cells.index)
    cell_codes = cell_id_cat.codes  # int array aligned with df_valid rows
    n_cells = len(valid_cells)
    n_traits = len(trait_names)

    uniform = np.float32(1.0 / n_bins)
    hist_arr = np.full((n_cells, n_traits, n_bins), uniform, dtype=np.float32)
    mask_arr = np.zeros((n_cells, n_traits), dtype=bool)
    trait_valid_counts: dict[str, int] = {}

    log.info("Building histograms for %d traits across %d cells...", n_traits, n_cells)
    for j, trait in enumerate(trait_names):
        if trait not in bin_edges:
            trait_valid_counts[trait] = 0
            log.debug("  [%2d/%d] %s — skipped (no bin edges)", j + 1, n_traits, trait)
            continue

        edges = bin_edges[trait]

        # Drop rows with NaN for this trait
        not_null = df_valid[trait].notna().values
        n_valid_obs = int(not_null.sum())
        if not not_null.any():
            trait_valid_counts[trait] = 0
            log.debug("  [%2d/%d] %s — skipped (all NaN)", j + 1, n_traits, trait)
            continue

        t_codes = cell_codes[not_null]
        t_values = df_valid[trait].values[not_null]
        t_weights = df_valid[weight_col].values[not_null]

        # Digitize all values at once → bin indices
        bin_idx = np.digitize(t_values, edges) - 1
        np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)

        # Scatter-add weighted counts into (n_cells, n_bins) matrix
        flat_idx = t_codes * n_bins + bin_idx
        counts_flat = np.zeros(n_cells * n_bins, dtype=np.float64)
        np.add.at(counts_flat, flat_idx, t_weights)
        counts_2d = counts_flat.reshape(n_cells, n_bins)

        # Per-cell observation count for this trait (non-NaN rows)
        obs_per_cell = np.zeros(n_cells, dtype=np.int64)
        np.add.at(obs_per_cell, t_codes, 1)

        # Per-trait validity: enough observations and bin coverage
        non_empty = (counts_2d > 0).sum(axis=1)
        coverage = non_empty / n_bins
        trait_ok = (obs_per_cell >= min_unique_species) & (coverage >= min_bin_coverage)

        # Label smoothing + normalize (vectorized over valid cells)
        if trait_ok.any():
            smoothed = counts_2d[trait_ok] + epsilon
            hist_arr[trait_ok, j, :] = (
                smoothed / smoothed.sum(axis=1, keepdims=True)
            ).astype(np.float32)
            mask_arr[trait_ok, j] = True

        n_ok = int(trait_ok.sum())
        trait_valid_counts[trait] = n_ok
        log.info(
            "  [%2d/%d] %-6s — %d obs, %d / %d cells valid (%.0f%%)",
            j + 1, n_traits, trait, n_valid_obs, n_ok, n_cells,
            100 * n_ok / n_cells if n_cells > 0 else 0,
        )

    # ------------------------------------------------------------------
    # Step 3: Remove cells where no trait passed
    # ------------------------------------------------------------------
    any_valid = mask_arr.any(axis=1)
    n_no_trait = int((~any_valid).sum())
    if n_no_trait > 0:
        log.info("Removing %d cells with no valid traits", n_no_trait)
    hist_arr = hist_arr[any_valid]
    mask_arr = mask_arr[any_valid]
    coords_arr = valid_cells[["cell_x", "cell_y"]].values[any_valid].astype(
        np.float64
    )

    n_cells_valid = int(hist_arr.shape[0])
    log.info(
        "Valid cells: %d / %d (%.1f%%)",
        n_cells_valid,
        n_cells_total,
        100 * n_cells_valid / n_cells_total if n_cells_total > 0 else 0,
    )

    stats = {
        "n_cells_total": n_cells_total,
        "n_cells_valid": n_cells_valid,
        "cells_filtered_observations": cells_filtered_obs,
        "cells_filtered_species": cells_filtered_species,
        "cells_filtered_abundance": cells_filtered_abundance,
        "trait_valid_counts": trait_valid_counts,
    }

    return hist_arr, mask_arr, coords_arr, stats


def _save_outputs(
    out_dir: Path,
    histograms: np.ndarray,
    masks: np.ndarray,
    coords: np.ndarray,
    bin_edges: dict[str, np.ndarray],
    trait_names: list[str],
    stats: dict,
    cfg,
    source: str,
) -> None:
    """Save histogram targets as a Zarr store with metadata.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    histograms : np.ndarray
        Histogram data, shape (N, n_traits, n_bins).
    masks : np.ndarray
        Valid trait indicators, shape (N, n_traits).
    coords : np.ndarray
        Cell coordinates, shape (N, 2) as [x, y].
    bin_edges : dict[str, np.ndarray]
        Bin edges for each trait.
    trait_names : list[str]
        Trait names.
    stats : dict
        Processing statistics.
    cfg : OmegaConf
        Configuration object.
    source : str
        Data source ('gbif' or 'splot').
    """
    zarr_path = out_dir / "histograms.zarr"
    root = zarr.open_group(zarr_path, mode="w")

    # Arrays — chunk along the cell (row) dimension
    root.create_array("histograms", data=histograms)
    root.create_array("masks", data=masks)
    root.create_array("coords", data=coords)

    # Store bin edges as (n_traits, n_bins+1)
    edges_arr = np.array(
        [bin_edges[t] for t in trait_names], dtype=np.float64
    )
    root.create_array("bin_edges", data=edges_arr)

    # Metadata as group attributes
    root.attrs["source"] = source
    root.attrs["n_bins"] = int(cfg.histogram.n_bins)
    root.attrs["label_smoothing_epsilon"] = float(
        cfg.histogram.label_smoothing_epsilon
    )
    root.attrs["target_resolution"] = int(cfg.target_resolution)
    root.attrs["crs"] = cfg.crs
    root.attrs["trait_names"] = list(trait_names)
    root.attrs["statistics"] = stats

    if source == "gbif":
        root.attrs["min_observations"] = int(cfg.gbif.min_observations)
        root.attrs["min_unique_species"] = int(cfg.gbif.min_unique_species)
        root.attrs["min_bin_coverage"] = float(cfg.gbif.min_bin_coverage)
    else:
        root.attrs["min_total_abundance"] = float(cfg.splot.min_total_abundance)
        root.attrs["min_unique_species"] = int(cfg.splot.min_unique_species)
        root.attrs["min_bin_coverage"] = float(cfg.splot.min_bin_coverage)

    log.info("Saved Zarr store to %s", zarr_path)
    log.info(
        "  histograms: %s %s, masks: %s %s, coords: %s %s",
        histograms.shape, histograms.dtype,
        masks.shape, masks.dtype,
        coords.shape, coords.dtype,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
