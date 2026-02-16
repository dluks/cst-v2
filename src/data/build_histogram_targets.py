"""Build histogram targets from GBIF or sPlot observations.

This module constructs probability histograms for each trait at the grid cell level,
using either GBIF occurrence data or sPlot vegetation survey data.

The histograms are built with global bin edges (computed from the full species pool)
to ensure spatial comparability across all grid cells.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyproj

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
    log.info("Building histogram targets from %s data", source.upper())

    # Output directory
    out_dir = proj_root / cfg.output.dir / source
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if outputs already exist
    hist_fp = out_dir / "histograms.parquet"
    if hist_fp.exists() and not args.overwrite:
        log.info("Output already exists at %s. Use --overwrite to replace.", hist_fp)
        return

    # Load trait data (species-level, transformed)
    traits_fp = proj_root / cfg.traits.interim_out
    log.info("Loading trait data from %s", traits_fp)
    traits_df = pd.read_parquet(traits_fp)
    trait_names = cfg.traits.names
    log.info("Loaded %d species with %d traits", len(traits_df), len(trait_names))

    # Compute global bin edges for each trait
    log.info("Computing global bin edges...")
    bin_edges = _compute_bin_edges(traits_df, trait_names, cfg.histogram.n_bins)

    # Load and process data based on source
    if source == "gbif":
        histograms_df, masks_df, coords_df, stats = _process_gbif(
            cfg, traits_df, trait_names, bin_edges, proj_root
        )
    else:
        histograms_df, masks_df, coords_df, stats = _process_splot(
            cfg, traits_df, trait_names, bin_edges, proj_root
        )

    # Save outputs
    _save_outputs(
        out_dir=out_dir,
        histograms_df=histograms_df,
        masks_df=masks_df,
        coords_df=coords_df,
        bin_edges=bin_edges,
        trait_names=trait_names,
        stats=stats,
        cfg=cfg,
        source=source,
    )

    log.info("Histogram target construction complete!")


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
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
    # Compute cell indices (floor division to get cell origin)
    df["cell_x"] = (df["x"] // resolution) * resolution
    df["cell_y"] = (df["y"] // resolution) * resolution
    # Create unique cell ID string for grouping
    df["cell_id"] = df["cell_x"].astype(str) + "_" + df["cell_y"].astype(str)
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
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
    tuple
        (histograms_df, masks_df, coords_df, stats)
    """
    # Group by cell
    grouped = df.groupby("cell_id")
    n_cells_total = len(grouped)
    log.info("Processing %d unique grid cells...", n_cells_total)

    # Initialize output containers
    cell_ids = []
    cell_coords = []
    cell_histograms = []
    cell_masks = []

    # Statistics
    cells_filtered_obs = 0
    cells_filtered_species = 0
    cells_filtered_abundance = 0
    trait_valid_counts = {t: 0 for t in trait_names}

    for cell_id, cell_df in grouped:
        # Get cell coordinates (use first observation's cell coords)
        cell_x = cell_df["cell_x"].iloc[0]
        cell_y = cell_df["cell_y"].iloc[0]

        # Check minimum observations (GBIF)
        if min_observations is not None:
            weighted_obs = cell_df[weight_col].sum()
            if weighted_obs < min_observations:
                cells_filtered_obs += 1
                continue

        # Check minimum total abundance (sPlot)
        if min_total_abundance is not None:
            total_abundance = cell_df["Rel_Abund_Plot"].sum()
            # Normalize by number of unique plots
            n_plots = cell_df["PlotObservationID"].nunique()
            if n_plots > 0:
                avg_abundance = total_abundance / n_plots
                if avg_abundance < min_total_abundance:
                    cells_filtered_abundance += 1
                    continue

        # Check minimum unique species
        n_species = cell_df[species_col].nunique()
        if n_species < min_unique_species:
            cells_filtered_species += 1
            continue

        # Build histogram for each trait
        trait_histograms = {}
        trait_masks = {}

        for trait in trait_names:
            if trait not in bin_edges:
                trait_histograms[trait] = np.full(n_bins, 1.0 / n_bins)
                trait_masks[trait] = False
                continue

            # Get trait values and weights
            trait_df = cell_df[[trait, weight_col]].dropna()
            values = trait_df[trait].values
            weights = trait_df[weight_col].values

            if len(values) < min_unique_species:
                # Not enough observations for this trait in this cell
                trait_histograms[trait] = np.full(n_bins, 1.0 / n_bins)
                trait_masks[trait] = False
                continue

            # Compute weighted histogram
            counts, _ = np.histogram(values, bins=bin_edges[trait], weights=weights)

            # Check bin coverage
            non_empty_bins = np.sum(counts > 0)
            bin_coverage = non_empty_bins / n_bins
            if bin_coverage < min_bin_coverage:
                trait_histograms[trait] = np.full(n_bins, 1.0 / n_bins)
                trait_masks[trait] = False
                continue

            # Apply label smoothing
            counts_smoothed = counts + epsilon

            # Normalize to probability distribution
            histogram = counts_smoothed / counts_smoothed.sum()

            trait_histograms[trait] = histogram
            trait_masks[trait] = True
            trait_valid_counts[trait] += 1

        # Check if at least one trait has a valid histogram
        if not any(trait_masks.values()):
            continue

        # Store results
        cell_ids.append(cell_id)
        cell_coords.append((cell_x, cell_y))
        cell_histograms.append(trait_histograms)
        cell_masks.append(trait_masks)

    n_cells_valid = len(cell_ids)
    log.info(
        "Valid cells: %d / %d (%.1f%%)",
        n_cells_valid,
        n_cells_total,
        100 * n_cells_valid / n_cells_total if n_cells_total > 0 else 0,
    )
    log.info("Filtered - observations: %d, species: %d, abundance: %d",
             cells_filtered_obs, cells_filtered_species, cells_filtered_abundance)

    # Convert to DataFrames
    # Histograms: flatten to (n_cells, n_traits * n_bins)
    hist_columns = []
    for trait in trait_names:
        for b in range(n_bins):
            hist_columns.append(f"{trait}_bin{b}")

    hist_data = []
    for cell_hist in cell_histograms:
        row = []
        for trait in trait_names:
            row.extend(cell_hist.get(trait, np.full(n_bins, 1.0 / n_bins)))
        hist_data.append(row)

    histograms_df = pd.DataFrame(hist_data, columns=hist_columns, index=cell_ids)

    # Masks: (n_cells, n_traits)
    mask_data = []
    for cell_mask in cell_masks:
        row = [cell_mask.get(trait, False) for trait in trait_names]
        mask_data.append(row)

    masks_df = pd.DataFrame(mask_data, columns=trait_names, index=cell_ids)

    # Coordinates: (n_cells, 2)
    coords_df = pd.DataFrame(cell_coords, columns=["x", "y"], index=cell_ids)

    # Statistics
    stats = {
        "n_cells_total": n_cells_total,
        "n_cells_valid": n_cells_valid,
        "cells_filtered_observations": cells_filtered_obs,
        "cells_filtered_species": cells_filtered_species,
        "cells_filtered_abundance": cells_filtered_abundance,
        "trait_valid_counts": trait_valid_counts,
    }

    return histograms_df, masks_df, coords_df, stats


def _save_outputs(
    out_dir: Path,
    histograms_df: pd.DataFrame,
    masks_df: pd.DataFrame,
    coords_df: pd.DataFrame,
    bin_edges: dict[str, np.ndarray],
    trait_names: list[str],
    stats: dict,
    cfg,
    source: str,
) -> None:
    """Save histogram targets and metadata to disk.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    histograms_df : pd.DataFrame
        Histogram data.
    masks_df : pd.DataFrame
        Valid trait indicators.
    coords_df : pd.DataFrame
        Cell coordinates.
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
    # Save histograms
    hist_fp = out_dir / "histograms.parquet"
    histograms_df.to_parquet(hist_fp)
    log.info("Saved histograms to %s", hist_fp)

    # Save masks
    masks_fp = out_dir / "masks.parquet"
    masks_df.to_parquet(masks_fp)
    log.info("Saved masks to %s", masks_fp)

    # Save coordinates
    coords_fp = out_dir / "coordinates.parquet"
    coords_df.to_parquet(coords_fp)
    log.info("Saved coordinates to %s", coords_fp)

    # Save metadata
    metadata = {
        "source": source,
        "n_bins": cfg.histogram.n_bins,
        "label_smoothing_epsilon": cfg.histogram.label_smoothing_epsilon,
        "target_resolution": cfg.target_resolution,
        "crs": cfg.crs,
        "trait_names": trait_names,
        "bin_edges": {k: v.tolist() for k, v in bin_edges.items()},
        "statistics": stats,
    }

    if source == "gbif":
        metadata["min_observations"] = cfg.gbif.min_observations
        metadata["min_unique_species"] = cfg.gbif.min_unique_species
        metadata["min_bin_coverage"] = cfg.gbif.min_bin_coverage
    else:
        metadata["min_total_abundance"] = cfg.splot.min_total_abundance
        metadata["min_unique_species"] = cfg.splot.min_unique_species
        metadata["min_bin_coverage"] = cfg.splot.min_bin_coverage

    meta_fp = out_dir / "metadata.json"
    with open(meta_fp, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved metadata to %s", meta_fp)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
