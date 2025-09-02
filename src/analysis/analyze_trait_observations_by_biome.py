#!/usr/bin/env python3
"""
Script to analyze trait observations by source and biome using Dask for large-scale processing.

This script:
1. Reads the Y.parquet dataset from data/features/Shrub_Tree_Grass/1km/
2. Assigns biomes to each observation point using the biome raster
3. Creates a summary dataframe showing the number of observations for each trait_id, source, and biome combination.
"""

import json
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import rasterio
from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import get_biome_map_fn

cfg = get_config()


def load_biome_names() -> dict:
    """
    Load biome names from the biomes.json file.

    Returns:
        Dictionary mapping biome numbers (as strings) to biome names
    """
    biome_file = Path("reference/biomes.json")
    with open(biome_file) as f:
        biome_names = json.load(f)
    return biome_names


def assign_biome_to_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign biome values to dataframe points based on x, y coordinates.

    Args:
        df: DataFrame with 'x' and 'y' columns

    Returns:
        DataFrame with added 'biome' column
    """
    df = df.copy().sort_values(by=["x", "y"])

    with rasterio.open(get_biome_map_fn()) as src:
        # Read the entire band once
        biome_band = src.read(1)

        # Get raster bounds for diagnostic info
        bounds = src.bounds
        log.info(f"Biome raster bounds: {bounds}")
        log.info(f"Biome raster shape: {biome_band.shape}")
        log.info(f"Unique biome values in raster: {sorted(np.unique(biome_band))}")

        # Check coordinate ranges
        x_range = (df["x"].min(), df["x"].max())
        y_range = (df["y"].min(), df["y"].max())
        log.info(f"Observation coordinate ranges - X: {x_range}, Y: {y_range}")

        # Compute row,col indices vectorized
        row_cols = [src.index(x, y) for x, y in zip(df["x"], df["y"])]

        # Check for out-of-bounds indices
        valid_indices = []
        biome_values = []

        for i, (r, c) in enumerate(row_cols):
            # Check if the index is within bounds
            if 0 <= r < biome_band.shape[0] and 0 <= c < biome_band.shape[1]:
                biome_val = biome_band[r, c]
                valid_indices.append(i)
                biome_values.append(biome_val)
            else:
                log.warning(
                    f"Point {i} at coordinates ({df.iloc[i]['x']}, {df.iloc[i]['y']}) is outside raster bounds"
                )
                biome_values.append(0)  # Assign 0 for out-of-bounds points

        # Assign biome values
        df["biome"] = biome_values

        # Diagnostic information
        biome_counts = df["biome"].value_counts().sort_index()
        log.info(f"Biome assignment results:")
        for biome, count in biome_counts.items():
            if biome == 0:
                log.info(f"  Biome 0 (no data/out of bounds): {count:,} observations")
            else:
                log.info(f"  Biome {biome}: {count:,} observations")

        # Check for any NaN values in biome column
        nan_count = df["biome"].isna().sum()
        if nan_count > 0:
            log.warning(f"Found {nan_count} NaN values in biome column")

    return df


def get_trait_columns(df: dd.DataFrame) -> list[str]:
    """
    Extract trait columns (columns starting with 'X' and ending with '_mean').

    Args:
        df: Input Dask dataframe

    Returns:
        List of trait column names
    """
    trait_cols = [
        col for col in df.columns if col.startswith("X") and col.endswith("_mean")
    ]
    return trait_cols


def main():
    """Main function to execute the analysis."""

    log.info("Starting trait observations by biome analysis with Dask...")

    # Initialize Dask
    client, cluster = init_dask(
        dashboard_address=cfg.dask_dashboard
        if hasattr(cfg, "dask_dashboard")
        else ":8787",
        n_workers=8,
        threads_per_worker=2,
    )

    try:
        # Define paths
        y_parquet_path = Path("data/features/Shrub_Tree_Grass/1km/Y.parquet")
        output_path = Path("results/trait_observations_by_biome.parquet")

        log.info(f"Reading data from {y_parquet_path}")

        # Read the partitioned parquet dataset with Dask
        try:
            ddf = dd.read_parquet(y_parquet_path)
            log.info(
                f"Successfully loaded Dask DataFrame with {ddf.npartitions} partitions"
            )
            log.info(f"Columns: {list(ddf.columns)}")

            # Get basic info about the dataset
            n_rows = len(ddf)
            log.info(f"Total rows: {n_rows:,}")

        except Exception as e:
            log.error(f"Error reading parquet data: {e}")
            return

        # Check data structure
        trait_cols = get_trait_columns(ddf)
        log.info(f"Found {len(trait_cols)} trait columns")

        # Get unique sources
        unique_sources = ddf["source"].unique().compute()
        log.info(f"Unique sources: {unique_sources}")

        # Convert to pandas for biome assignment (this is the bottleneck operation)
        log.info("Assigning biomes to observation points...")
        try:
            log.info("Converting to pandas for biome assignment...")
            df_pandas = ddf.compute()
            log.info(f"Converted to pandas: {len(df_pandas):,} rows")

            # Assign biomes
            df_with_biomes = assign_biome_to_points(df_pandas)
            log.info(
                f"Assigned biomes. Unique biomes: {sorted(df_with_biomes['biome'].unique())}"
            )

            # Check how many observations have biome 0
            biome_0_count = (df_with_biomes["biome"] == 0).sum()
            total_obs = len(df_with_biomes)
            log.info(
                f"Observations with biome 0: {biome_0_count:,} out of {total_obs:,} ({biome_0_count / total_obs * 100:.1f}%)"
            )

            # Optionally filter out biome 0 observations
            if biome_0_count > 0:
                log.info("Filtering out biome 0 observations for analysis...")
                df_with_biomes = df_with_biomes[df_with_biomes["biome"] != 0].copy()
                log.info(
                    f"Remaining observations after filtering: {len(df_with_biomes):,}"
                )

            log.info(
                f"Final unique biomes after filtering: {sorted(df_with_biomes['biome'].unique())}"
            )

        except Exception as e:
            log.error(f"Error assigning biomes: {e}")
            return

        # Create summary using pandas operations (more reliable)
        log.info("Creating trait observation summary...")
        try:
            summary_records = []

            for i, trait_col in enumerate(trait_cols):
                log.info(f"Processing trait {i + 1}/{len(trait_cols)}: {trait_col}")

                # Extract trait ID
                trait_id = trait_col.replace("_mean", "")

                # Select only needed columns and filter non-null values using pandas
                trait_data = df_with_biomes[["source", "biome", trait_col]].dropna()

                # Group and count using pandas (more reliable than Dask for this operation)
                counts = (
                    trait_data.groupby(["source", "biome"])
                    .size()
                    .reset_index(name="n_observations")
                )

                # Add trait_id
                counts["trait_id"] = trait_id
                counts = counts[["trait_id", "source", "biome", "n_observations"]]

                summary_records.append(counts)

            # Combine all results
            summary_df = pd.concat(summary_records, ignore_index=True)
            log.info(f"Created raw summary with {len(summary_df):,} records")

            # Compute summary statistics for each biome-source combination
            log.info("Computing summary statistics per biome and source...")

            # For each biome and source, calculate min/max/mean/median across all traits
            biome_stats = (
                summary_df.groupby(["biome", "source"])["n_observations"]
                .agg(["min", "max", "mean", "median", "std"])
                .round(2)
            )

            log.info(
                f"Computed statistics for {len(biome_stats)} biome-source combinations"
            )

            # Pivot to get separate columns for each source's statistics
            # This will create columns like: ('min', 'G'), ('max', 'G'), ('mean', 'G'), etc.
            pivoted_stats = biome_stats.unstack("source")

            # Flatten column names and rename to requested format
            pivoted_stats.columns = [
                f"{stat}_{source.lower()}" for stat, source in pivoted_stats.columns
            ]

            # Fill any missing values with 0 (in case some biomes don't have both sources)
            pivoted_stats = pivoted_stats.fillna(0)

            # Rename index to just 'biome'
            pivoted_stats.index.name = "biome"

            # Load biome names and add them to the dataframe
            log.info("Adding biome names to the output...")
            biome_names = load_biome_names()

            # Add biome name column
            pivoted_stats["biome_name"] = pivoted_stats.index.map(
                lambda x: biome_names.get(str(x), f"Unknown Biome {x}")
            )

            # Reorder columns to put biome_name first
            cols = ["biome_name"] + [
                col for col in pivoted_stats.columns if col != "biome_name"
            ]
            pivoted_stats = pivoted_stats[cols]

            log.info(f"Final data shape: {pivoted_stats.shape}")
            log.info(f"Columns: {list(pivoted_stats.columns)}")

            # Display some basic statistics
            log.info("\nSummary statistics:")
            log.info(f"Number of biomes: {len(pivoted_stats)}")

            # Show statistics for each source
            for source in ["g", "s"]:
                source_name = "GBIF" if source == "g" else "sPlot"
                if f"mean_{source}" in pivoted_stats.columns:
                    log.info(f"\n{source_name} statistics across biomes:")
                    log.info(
                        f"  Average min observations per trait: {pivoted_stats[f'min_{source}'].mean():.1f}"
                    )
                    log.info(
                        f"  Average max observations per trait: {pivoted_stats[f'max_{source}'].mean():.1f}"
                    )
                    log.info(
                        f"  Average mean observations per trait: {pivoted_stats[f'mean_{source}'].mean():.1f}"
                    )
                    log.info(
                        f"  Average median observations per trait: {pivoted_stats[f'median_{source}'].mean():.1f}"
                    )
                    log.info(
                        f"  Average std observations per trait: {pivoted_stats[f'std_{source}'].mean():.1f}"
                    )

            # Show top biomes by mean GBIF observations
            if "mean_g" in pivoted_stats.columns:
                top_biomes_gbif = pivoted_stats.sort_values("mean_g", ascending=False)
                log.info("\nTop 10 biomes by mean GBIF observations per trait:")
                for biome, row in top_biomes_gbif.head(10).iterrows():
                    biome_name = row["biome_name"]
                    log.info(
                        f"  Biome {biome} ({biome_name}): {row['mean_g']:.1f} mean observations per trait"
                    )

            # Show top biomes by mean sPlot observations
            if "mean_s" in pivoted_stats.columns:
                top_biomes_splot = pivoted_stats.sort_values("mean_s", ascending=False)
                log.info("\nTop 10 biomes by mean sPlot observations per trait:")
                for biome, row in top_biomes_splot.head(10).iterrows():
                    biome_name = row["biome_name"]
                    log.info(
                        f"  Biome {biome} ({biome_name}): {row['mean_s']:.1f} mean observations per trait"
                    )

        except Exception as e:
            log.error(f"Error creating summary: {e}")
            return

        # Save results in the requested format
        log.info(f"Saving results to {output_path}")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pivoted_stats.to_parquet(output_path)
            log.info("Results saved successfully!")

            # Also save as CSV for easy inspection
            csv_path = output_path.with_suffix(".csv")
            pivoted_stats.to_csv(csv_path)
            log.info(f"Also saved as CSV: {csv_path}")

        except Exception as e:
            log.error(f"Error saving results: {e}")
            return

        log.info("Analysis completed successfully!")

        return pivoted_stats

    finally:
        # Always close Dask client
        close_dask(client)


if __name__ == "__main__":
    summary_df = main()
