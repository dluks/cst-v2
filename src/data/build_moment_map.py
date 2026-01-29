"""
Build statistical moment maps (mean, variance, skewness, kurtosis) per grid cell.

This module computes the 4 statistical moments from pooled observations within
each grid cell, following the constraints for distribution modeling:

- Pool observations within grid cells before computing moments (Jensen's inequality)
- Use Fisher's bias corrections for skewness/kurtosis
- Use raw kurtosis (β₂, normal=3) for Pearson system compatibility
- Normalize abundances per plot so each plot contributes equally (sPlot only)

For GBIF: Pool observations directly within grid cells.
For sPlot: Expand plot abundances to pseudo-observations, then pool within grid cells.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.df_utils import rasterize_points, reproject_geo_to_xy
from src.utils.raster_utils import xr_to_raster
from src.utils.trait_utils import filter_pft


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build statistical moment maps (mean, variance, skewness, kurtosis) "
        "per grid cell from GBIF or sPlot data."
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        required=True,
        help="Path to the parameters file.",
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        required=True,
        choices=["gbif", "splot"],
        help="Data source: 'gbif' or 'splot'.",
    )
    parser.add_argument(
        "-t",
        "--trait",
        type=str,
        required=True,
        help="Trait name to process (e.g. 'X4', 'X6', 'gsmax').",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Main function."""
    args = cli() if args is None else args
    cfg = get_config(params_path=args.params)

    # Enable GDAL multi-threading based on allocated CPUs
    n_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    os.environ["GDAL_NUM_THREADS"] = n_cpus
    log.info("Setting GDAL_NUM_THREADS=%s", n_cpus)

    # Determine output directory based on source
    if args.source == "gbif":
        out_dir = Path(cfg.gbif.maps.out_dir, cfg.product_code)
    else:
        out_dir = Path(cfg.splot.maps.out_dir, cfg.product_code)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_fn = out_dir / f"{args.trait}.tif"

    if out_fn.exists() and not args.overwrite:
        log.info("%s.tif already exists. Skipping...", args.trait)
        return

    # Process based on source
    if args.source == "gbif":
        raster = _process_gbif(cfg, args.trait)
    else:
        raster = _process_splot(cfg, args.trait)

    if raster is None:
        log.warning("No data to process for trait %s. Skipping...", args.trait)
        return

    log.info("Writing to disk: %s", out_fn)
    xr_to_raster(raster, out_fn, num_threads=int(n_cpus))
    log.info("Wrote %s.tif.", args.trait)
    log.info("Done!")


def _process_gbif(cfg, trait: str):
    """Process GBIF data to compute moment maps.

    For GBIF, each observation is a single species occurrence. We join with
    trait data to get the median trait value for that species, then pool all
    observations within each grid cell to compute the 4 moments.
    """
    log.info("Processing GBIF data for trait %s...", trait)

    # Load filtered GBIF data
    gbif_fp = Path(cfg.gbif.filtered.out_dir, cfg.trait_type, cfg.gbif.filtered.fp)
    log.info("Loading GBIF data from %s...", gbif_fp)
    gbif = pd.read_parquet(gbif_fp).pipe(filter_pft, cfg.PFT).drop(columns=["pft"])

    # Load trait data
    log.info("Loading trait data...")
    trait_df = pd.read_parquet(
        cfg.traits.interim_out, columns=["GBIFKeyGBIF", trait]
    ).rename(columns={"GBIFKeyGBIF": "specieskey"})

    # Join GBIF with traits
    log.info("Joining GBIF with trait data...")
    gbif_traits = gbif.merge(trait_df, on="specieskey", how="inner")

    if len(gbif_traits) == 0:
        log.error("No data after joining GBIF with trait %s.", trait)
        return None

    log.info("Joined %d observations with trait values.", len(gbif_traits))

    # Reproject coordinates
    log.info("Reprojecting coordinates...")
    gbif_traits = _reproject_gbif(cfg.crs, gbif_traits)

    # Rasterize with moment functions
    log.info("Computing moments per grid cell...")
    n_min = cfg.moments.get("n_min", 30)
    min_unique = cfg.moments.get("min_unique_species", 3)

    raster = rasterize_points(
        gbif_traits[["x", "y", trait, "weight", "specieskey"]],
        data_cols=trait,
        res=cfg.target_resolution,
        crs=cfg.crs,
        agg=True,
        funcs=[
            "mean", "variance", "skewness", "kurtosis", "n_eff", "count", "n_species"
        ],
        n_min=n_min,
        n_max=cfg.gbif.maps.get("max_count", 500),
        weights="weight",
        unique_col="specieskey",
        min_unique=min_unique,
    )

    return raster


def _process_splot(cfg, trait: str):
    """Process sPlot data to compute moment maps.

    For sPlot, each plot contains multiple species with relative abundances.
    We need to:
    1. Join with trait data
    2. Filter plots with < min_abundance after trait matching
    3. Expand abundances to pseudo-observations (preserving original proportions)
    4. Pool all pseudo-observations within each grid cell
    5. Compute the 4 moments from pooled data

    Note: We do NOT normalize abundances per plot. If a plot has 80% of its
    original abundance matched to traits, it contributes ~80 pseudo-observations
    (with multiplier=100), not 100. This preserves the relative contribution
    of plots based on their trait coverage.
    """
    log.info("Processing sPlot data for trait %s...", trait)

    # Load filtered sPlot data
    splot_fp = Path(cfg.splot.filtered.out_dir, cfg.trait_type, cfg.splot.filtered.fp)
    log.info("Loading sPlot data from %s...", splot_fp)
    splot = pd.read_parquet(splot_fp).pipe(filter_pft, cfg.PFT).drop(columns=["pft"])

    # Load trait data
    log.info("Loading trait data...")
    trait_df = pd.read_parquet(cfg.traits.interim_out, columns=["nameOutWCVP", trait])

    # Join sPlot with traits
    log.info("Joining sPlot with trait data...")
    splot_traits = splot.merge(
        trait_df, left_on="speciesname", right_on="nameOutWCVP", how="inner"
    ).drop(columns=["nameOutWCVP"])

    if len(splot_traits) == 0:
        log.error("No data after joining sPlot with trait %s.", trait)
        return None

    log.info("Joined %d observations with trait values.", len(splot_traits))

    # Get moment parameters
    min_abundance = cfg.moments.get("splot_min_abundance", 0.75)
    multiplier = cfg.moments.get("splot_abundance_multiplier", 100)
    # Note: n_min is NOT applied to sPlot. The abundance filter (splot_min_abundance)
    # already ensures data quality. Pseudo-observation count is an artifact of the
    # expansion multiplier, not a measure of sample quality.

    # Filter plots by minimum abundance threshold
    log.info("Filtering plots by minimum abundance (%.2f)...", min_abundance)
    splot_traits = _filter_plots_by_abundance(
        splot_traits,
        min_abundance=min_abundance,
        plot_id_col="PlotObservationID",
        abundance_col="Rel_Abund_Plot",
    )

    if len(splot_traits) == 0:
        log.error("No plots remaining after abundance filtering.")
        return None

    n_plots = splot_traits["PlotObservationID"].nunique()
    log.info("Retained %d plots after abundance filtering.", n_plots)

    # Expand abundances to pseudo-observations
    # Note: We do NOT normalize - plots contribute proportional to their trait coverage
    log.info("Expanding abundances to pseudo-observations (multiplier=%d)...", multiplier)
    pseudo_obs = _expand_abundances(
        splot_traits,
        trait_col=trait,
        abundance_col="Rel_Abund_Plot",
        multiplier=multiplier,
    )

    log.info("Created %d pseudo-observations from %d plots.", len(pseudo_obs), n_plots)

    # Reproject coordinates
    log.info("Reprojecting coordinates...")
    pseudo_obs = _reproject_splot(cfg.crs, pseudo_obs)

    # Rasterize with moment functions
    # Note: n_min=1 (default) for sPlot - abundance filter handles quality control
    # min_unique_species=3 ensures enough species for meaningful distribution estimation
    log.info("Computing moments per grid cell...")
    min_unique = cfg.moments.get("min_unique_species", 3)

    raster = rasterize_points(
        pseudo_obs[["x", "y", trait, "weight", "speciesname"]],
        data_cols=trait,
        res=cfg.target_resolution,
        crs=cfg.crs,
        agg=True,
        funcs=[
            "mean", "variance", "skewness", "kurtosis", "n_eff", "count", "n_species"
        ],
        weights="weight",
        unique_col="speciesname",
        min_unique=min_unique,
    )

    return raster


def _filter_plots_by_abundance(
    df: pd.DataFrame,
    min_abundance: float,
    plot_id_col: str,
    abundance_col: str,
) -> pd.DataFrame:
    """Filter plots where cumulative abundance is below threshold.

    After joining with trait data, some species are dropped because they don't
    have trait values. This function filters out plots where the remaining
    species don't represent at least `min_abundance` of the original plot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with plot observations.
    min_abundance : float
        Minimum cumulative abundance required (e.g., 0.75 for 75%).
    plot_id_col : str
        Column name for plot identifier.
    abundance_col : str
        Column name for species abundance.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only plots meeting the abundance threshold.
    """
    # Calculate cumulative abundance per plot
    plot_abundance = df.groupby(plot_id_col)[abundance_col].sum()

    # Filter plots meeting the threshold
    valid_plots = plot_abundance[plot_abundance >= min_abundance].index

    # Filter the DataFrame
    filtered_df = df[df[plot_id_col].isin(valid_plots)].copy()

    return filtered_df




def _expand_abundances(
    df: pd.DataFrame,
    trait_col: str,
    abundance_col: str,
    multiplier: int = 100,
) -> pd.DataFrame:
    """Expand fractional abundances to pseudo-observations.

    Each species in a plot is replicated proportional to its abundance.
    E.g., species with abundance=0.4 gets 40 pseudo-observations (multiplier=100).

    The resurvey weight from the filtered sPlot data is preserved - if a plot
    has weight=0.5 (from resurvey detection), all its pseudo-observations
    retain that weight.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with plot observations. Must have columns:
        - trait_col: trait values
        - abundance_col: normalized abundances (sum to 1 per plot)
        - Latitude, Longitude: coordinates
        - weight: observation weight (from resurvey detection)
    trait_col : str
        Column name for trait values.
    abundance_col : str
        Column name for abundances.
    multiplier : int
        Factor to convert fractions to counts. Default 100.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per pseudo-observation. The weight column
        is preserved from the input (resurvey weights are retained).
    """
    # Calculate number of copies for each row
    n_copies = np.maximum(1, np.round(df[abundance_col] * multiplier).astype(int))

    # Repeat rows according to n_copies
    # The weight column is preserved (resurvey weights from filter_splot.py)
    expanded = df.loc[df.index.repeat(n_copies)].copy()

    return expanded.reset_index(drop=True)


def _reproject_gbif(crs: str, df: pd.DataFrame) -> pd.DataFrame:
    """Reproject GBIF coordinates."""
    if crs != "EPSG:4326":
        if crs != "EPSG:6933":
            raise ValueError(f"Unsupported CRS: {crs}")

        df = reproject_geo_to_xy(
            df,
            to_crs=crs,
            x="decimallongitude",
            y="decimallatitude",
        ).drop(columns=["decimallatitude", "decimallongitude"])
    else:
        df = df.rename(columns={"decimallongitude": "x", "decimallatitude": "y"})

    return df


def _reproject_splot(crs: str, df: pd.DataFrame) -> pd.DataFrame:
    """Reproject sPlot coordinates."""
    if crs != "EPSG:4326":
        if crs != "EPSG:6933":
            raise ValueError(f"Unsupported CRS: {crs}")

        df = reproject_geo_to_xy(
            df,
            to_crs=crs,
            x="Longitude",
            y="Latitude",
        ).drop(columns=["Latitude", "Longitude"])
    else:
        df = df.rename(columns={"Longitude": "x", "Latitude": "y"})

    return df


if __name__ == "__main__":
    main()
