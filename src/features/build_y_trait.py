"""
Process a single trait's Y data from GBIF and sPlot maps.

This script loads trait maps from both GBIF and sPlot datasets, extracts the
specified band (trait statistic), converts to dataframe format, and saves to
an intermediate parquet file for later merging.

Optionally applies power transformation (Yeo-Johnson) to trait values and
adds reliability weighting based on count_weighted values.
"""

import argparse
import gc
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

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
        trait_stat_band = (
            cfg.traits.trait_map_layers.index(cfg.traits.target_trait_stat) + 1
        )
        log.info(
            "Trait statistic band: %d (%s)",
            trait_stat_band,
            cfg.traits.target_trait_stat,
        )
    except ValueError:
        raise ValueError(
            f"Target trait stat {cfg.traits.target_trait_stat} not found in trait map layers."
        )

    # Get count_weighted band if reliability weighting or min_count filtering is needed
    count_weighted_band = None

    gbif_min_count = cfg.gbif.min_count

    use_reliability_weights = cfg.traits.get("use_reliability_weights", False)

    if use_reliability_weights or gbif_min_count is not None:
        try:
            count_weighted_band = (
                cfg.traits.trait_map_layers.index("count_weighted") + 1
            )
            log.info("Count_weighted band: %d", count_weighted_band)
            if gbif_min_count is not None:
                log.info("GBIF minimum count threshold: %.1f", gbif_min_count)
        except ValueError:
            log.warning(
                "count_weighted not found in trait map layers, "
                "reliability weighting and min_count filtering disabled"
            )
            gbif_min_count = None

    # Load GBIF map (with optional min_count filtering)
    trait_fp = Path(cfg.gbif.maps_dir, f"{args.trait}.tif")
    gbif_df = load_trait_map(
        "gbif",
        trait_fp,
        args.trait,
        trait_stat_band,
        count_weighted_band,
        min_count=gbif_min_count,
    )

    # Load sPlot map (no min_count filtering for sPlot)
    trait_fp = Path(cfg.splot.maps_dir, f"{args.trait}.tif")
    splot_df = load_trait_map(
        "splot", trait_fp, args.trait, trait_stat_band, count_weighted_band
    )

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

    # Explicitly delete source dataframes and force garbage collection
    del dfs_to_concat, gbif_df, splot_df
    import gc

    gc.collect()

    log.info("Total observations for trait %s: %d", args.trait, len(combined_df))

    # Apply power transformation if enabled
    if cfg.traits.get("power_transform", False):
        combined_df = apply_power_transform(combined_df, args.trait, cfg)
        # Force garbage collection after transformation
        gc.collect()

    # Calculate reliability weights if enabled
    if count_weighted_band is not None and "count_weighted" in combined_df.columns:
        combined_df = calculate_reliability_weights(combined_df, args.trait, cfg)
        # Force garbage collection after reliability calculation
        gc.collect()

    # Write to parquet
    log.info("Writing to %s...", out_fn)
    combined_df.to_parquet(out_fn, compression="zstd", index=False)

    # Final cleanup
    del combined_df
    gc.collect()

    log.info("✓ Completed processing trait %s", args.trait)


def load_trait_map(
    y_set: str,
    trait_fn: Path,
    trait_name: str,
    trait_stat_band: int,
    count_weighted_band: int | None = None,
    min_count: float | None = None,
) -> pd.DataFrame | None:
    """
    Load a trait map and convert to dataframe.

    Args:
        y_set: Dataset name ('gbif' or 'splot')
        trait_fn: Path to trait map file
        trait_name: Trait name
        trait_stat_band: Band number for trait statistic (1-indexed)
        count_weighted_band: Band number for count_weighted (1-indexed), optional
        min_count: Minimum count_weighted value required to include a cell.
            Only applied when count_weighted_band is provided. Cells with
            count_weighted below this threshold are excluded.

    Returns:
        DataFrame with columns: x, y, {trait}, source, count_weighted (optional)
        None if file doesn't exist
    """
    if not trait_fn.exists():
        log.warning("Trait map not found: %s", trait_fn.absolute())
        return None

    log.info("Loading %s map: %s", y_set.upper(), trait_fn.absolute())

    # Load the raster
    ds = open_raster(trait_fn)

    # Extract the trait statistic band
    da_trait = ds.sel(band=trait_stat_band)

    # Extract count_weighted band if requested and join efficiently
    if count_weighted_band is not None:
        # Load both bands and convert to dataframes
        da_count = ds.sel(band=count_weighted_band)
        df_trait = da_trait.rename(trait_name).to_dask_dataframe()
        df_count = da_count.rename("count_weighted").to_dask_dataframe()

        # Use join instead of merge (more efficient for aligned indices)
        # This avoids creating a copy during merge
        df = df_trait.merge(
            df_count[["x", "y", "count_weighted"]], how="left", on=["x", "y"]
        )

        # Clean up intermediate dataframes immediately
        del df_trait, df_count, da_count
    else:
        # Convert to dataframe
        df = da_trait.rename(trait_name).to_dask_dataframe()

    # Add source column ('g' for GBIF, 's' for sPlot)
    source_code = "g" if y_set == "gbif" else "s"

    # Drop band and spatial_ref columns immediately if they exist
    cols_to_drop = ["band", "spatial_ref"]
    df = (
        df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        .dropna(subset=[trait_name])
        .assign(source=source_code)
    ).compute()

    # Apply minimum count filter if specified and count_weighted is available
    if min_count is not None and "count_weighted" in df.columns:
        n_before = len(df)
        df = df[df["count_weighted"] >= min_count]
        n_filtered = n_before - len(df)
        if n_filtered > 0:
            log.info(
                "Filtered %d cells with count_weighted < %.1f (%.1f%% removed)",
                n_filtered,
                min_count,
                100 * n_filtered / n_before,
            )

    # Clean up xarray objects
    log.info("Cleaning up xarray objects...")
    del da_trait, ds
    gc.collect()

    log.info(
        "Loaded %d observations from %s for trait %s",
        len(df),
        y_set.upper(),
        trait_name,
    )

    return df


def apply_power_transform(df: pd.DataFrame, trait_name: str, cfg) -> pd.DataFrame:
    """
    Apply Yeo-Johnson power transformation to trait values.

    Args:
        df: DataFrame with trait values
        trait_name: Name of trait column to transform
        cfg: Configuration object

    Returns:
        DataFrame with transformed trait values
    """
    log.info("Applying power transformation to trait %s...", trait_name)

    # Get transformation method
    method = cfg.traits.get("transform_method", "yeo-johnson")

    # Store original statistics for logging (use describe() for efficiency)
    orig_stats = df[trait_name].describe()
    orig_mean = orig_stats["mean"]
    orig_std = orig_stats["std"]
    orig_min = orig_stats["min"]
    orig_max = orig_stats["max"]

    # Calculate coefficient of variation
    cv = orig_std / abs(orig_mean) if orig_mean != 0 else float("inf")

    log.info(
        "Original distribution - Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f, CV: %.4f",
        orig_mean,
        orig_std,
        orig_min,
        orig_max,
        cv,
    )

    # Skip transformation if coefficient of variation is too low
    # Low CV indicates narrow relative variation that may not benefit from
    # transformation
    if cv < 0.15:
        log.info(
            "Skipping power transformation for trait %s due to low coefficient of "
            "variation (CV=%.4f < 0.15). Trait has narrow relative variation.",
            trait_name,
            cv,
        )

        # Save metadata indicating no transformation was applied
        proj_root = os.environ.get("PROJECT_ROOT")
        if proj_root is not None:
            transformer_dir = Path(proj_root) / cfg.traits.transformer_dir
            transformer_dir.mkdir(parents=True, exist_ok=True)

            metadata = {
                "trait": trait_name,
                "transformed": False,
                "reason": "low_cv",
                "cv": float(cv),
                "cv_threshold": 0.15,
                "orig_mean": float(orig_mean),
                "orig_std": float(orig_std),
                "orig_min": float(orig_min),
                "orig_max": float(orig_max),
            }

            metadata_path = transformer_dir / f"{trait_name}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            log.info("Saved no-transform metadata to: %s", metadata_path)

        return df

    # Fit and transform
    transformer = PowerTransformer(method=method, standardize=False)
    transformed_values = transformer.fit_transform(df[[trait_name]])

    # Update dataframe in-place
    df[trait_name] = transformed_values

    # Log transformed statistics (use describe() for efficiency)
    trans_stats = df[trait_name].describe()
    trans_mean = trans_stats["mean"]
    trans_std = trans_stats["std"]
    trans_min = trans_stats["min"]
    trans_max = trans_stats["max"]

    log.info(
        "Transformed distribution - Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f",
        trans_mean,
        trans_std,
        trans_min,
        trans_max,
    )

    # Save transformer to disk
    proj_root = os.environ.get("PROJECT_ROOT")
    if proj_root is None:
        raise ValueError("PROJECT_ROOT environment variable is not set")

    transformer_dir = Path(proj_root) / cfg.traits.transformer_dir
    transformer_dir.mkdir(parents=True, exist_ok=True)

    transformer_path = transformer_dir / f"{trait_name}_transformer.pkl"
    with open(transformer_path, "wb") as f:
        pickle.dump(transformer, f)

    log.info("Saved transformer to: %s", transformer_path)

    # Save transformation metadata
    metadata = {
        "trait": trait_name,
        "transformed": True,
        "method": method,
        "cv": float(cv),
        "orig_mean": float(orig_mean),
        "orig_std": float(orig_std),
        "orig_min": float(orig_min),
        "orig_max": float(orig_max),
        "trans_mean": float(trans_mean),
        "trans_std": float(trans_std),
        "trans_min": float(trans_min),
        "trans_max": float(trans_max),
        "lambda": float(transformer.lambdas_[0]),
    }

    metadata_path = transformer_dir / f"{trait_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Saved transformation metadata to: %s", metadata_path)

    return df


def calculate_reliability_weights(
    df: pd.DataFrame, trait_name: str, cfg
) -> pd.DataFrame:
    """
    Calculate reliability weights based on count_weighted values.

    Args:
        df: DataFrame with count_weighted column
        trait_name: Trait name (for naming the reliability column)
        cfg: Configuration object

    Returns:
        DataFrame with added reliability column
    """
    log.info("Calculating reliability weights for trait %s...", trait_name)

    # Get transformation method and normalization bounds
    transform_method = cfg.traits.get("reliability_transform", "sqrt")
    min_weight = cfg.traits.get("reliability_min", 0.1)
    max_weight = cfg.traits.get("reliability_max", 1.0)

    # Apply transformation to count_weighted in-place
    # Work directly with the values to minimize memory allocations
    count_vals = df["count_weighted"].values

    if transform_method == "sqrt":
        transformed = np.sqrt(count_vals)
    elif transform_method == "log":
        transformed = np.log1p(count_vals)
    elif transform_method == "none":
        transformed = count_vals
    else:
        raise ValueError(f"Unknown reliability transform: {transform_method}")

    # Normalize to [min_weight, max_weight] range in-place
    raw_min = transformed.min()
    raw_max = transformed.max()

    if raw_max > raw_min:
        # Compute normalized values directly without intermediate Series
        normalized = ((transformed - raw_min) / (raw_max - raw_min)) * (
            max_weight - min_weight
        ) + min_weight
    else:
        # If all values are the same, use max_weight
        normalized = np.full(len(transformed), max_weight)

    # Add as new column with trait-specific name
    df[f"{trait_name}_reliability"] = normalized

    # Drop the original count_weighted column
    df = df.drop(columns=["count_weighted"])

    # Calculate stats from the normalized array (avoid re-scanning dataframe)
    log.info(
        "Reliability weights - Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f",
        normalized.mean(),
        normalized.std(),
        normalized.min(),
        normalized.max(),
    )

    return df


if __name__ == "__main__":
    main()
