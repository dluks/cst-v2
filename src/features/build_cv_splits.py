"""Assign spatial k-fold cross-validation splits to a single trait."""

import argparse
import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from box import ConfigBox
from dask import compute, delayed
from scipy.stats import ks_2samp

from src.conf.conf import get_config
from src.conf.environment import activate_env, log
from src.utils.df_utils import reproject_xy_to_geo
from src.utils.log_utils import get_loggers_starting_with
from src.utils.spatial_utils import acr_to_h3_res, assign_hexagons

# Import cartopy for map features
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    log.warning("Cartopy not available. Fold maps will not include geographic features.")


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Assign spatial k-fold cross-validation splits to a single trait."
    )
    parser.add_argument(
        "-t", "--trait", type=str, required=True, help="Trait name to process"
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        default=None,
        help="Path to parameters file",
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite existing splits"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def main() -> None:
    """Main entry point for assigning CV splits to a single trait."""
    args = cli()
    activate_env()

    # Ignore warnings
    warnings.simplefilter(action="ignore", category=UserWarning)

    if args.debug:
        log.setLevel(logging.DEBUG)

    # Load configuration
    cfg = get_config(Path(args.params).resolve() if args.params else None)

    # Process the trait
    assign_trait_cv_splits(args.trait, cfg, args.overwrite)

    log.info("Done! \U00002705")


def calculate_kg_p_value(
    df: pd.DataFrame, data_col: str, fold_i: int, fold_j: int
) -> float:
    """
    Calculate the p-value using the Kolmogorov-Smirnov test for two folds in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        data_col (str): The column name of the data to compare.
        fold_i (int): The index of the first fold.
        fold_j (int): The index of the second fold.

    Returns:
        float: The p-value calculated using the Kolmogorov-Smirnov test.
               Returns 0.0 if either fold is empty (worst similarity score).
    """
    folds_df = df[df["fold"].isin([fold_i, fold_j])]
    folds_values = folds_df[data_col]
    mask = folds_df["fold"] == fold_i
    fold_i_values = folds_values[mask]
    fold_j_values = folds_values[~mask]

    # Handle empty folds - return 0.0 (worst similarity) to discourage this split
    if len(fold_i_values) == 0 or len(fold_j_values) == 0:
        log.warning(
            "Empty fold detected (fold %d: %d samples, fold %d: %d samples). "
            "Returning p-value of 0.0.",
            fold_i, len(fold_i_values), fold_j, len(fold_j_values)
        )
        return 0.0

    _, p_value = ks_2samp(fold_i_values, fold_j_values)
    return p_value  # pyright: ignore[reportReturnType]


def calculate_similarity_kg(folds: Sequence, df: pd.DataFrame, data_col: str) -> float:
    """
    Calculate the similarity between folds using the Kolmogorov-Smirnov test.

    Parameters:
    - folds (Sequence): A sequence of folds.
    - df (pd.DataFrame): The DataFrame containing the data.
    - data_col (str): The name of the column containing the data.

    Returns:
    - float: The similarity between the folds based on the Kolmogorov-Smirnov test.
    """

    # Calculate the pairwise comparisons
    p_values = [
        calculate_kg_p_value(df, data_col, folds[i], folds[j])
        for i in range(len(folds))
        for j in range(i + 1, len(folds))
    ]

    # Return the mean p-value as the similarity score
    return float(np.mean(p_values))


def assign_folds_iteration(
    df: pd.DataFrame, n_folds: int, data_col: str, hexagons: npt.NDArray
) -> tuple[float, pd.Series]:
    """
    Assigns folds to the hexagons in the given dataframe based on the number of folds
    specified.

    Parameters:
    - df: The input dataframe containing the hexagon data.
    - n_folds: The number of folds to assign.
    - data_col: The column name in the dataframe containing the data.
    - hexagons: The array of hexagons to assign folds to.

    Returns:
    - A tuple containing the similarity score and a copy of the fold assignments.
    """
    np.random.shuffle(hexagons)
    folds = np.array_split(hexagons, n_folds)
    hexagon_to_fold = {hexagon: i for i, fold in enumerate(folds) for hexagon in fold}
    df["fold"] = df["hex_id"].map(hexagon_to_fold)

    similarity = calculate_similarity_kg(range(n_folds), df, data_col)
    return similarity, df["fold"].copy()


def visualize_folds(
    trait_df: pd.DataFrame,
    trait_col: str,
    splits_dir: Path,
    cfg: ConfigBox,
) -> None:
    """
    Create a map visualization of the fold assignments.

    Args:
        trait_df: DataFrame with x, y, and fold columns
        trait_col: Name of the trait
        splits_dir: Directory to save the visualization
        cfg: Configuration object
    """
    log.info("Creating fold visualization map...")

    # Get data bounds
    x_min, x_max = trait_df['x'].min(), trait_df['x'].max()
    y_min, y_max = trait_df['y'].min(), trait_df['y'].max()

    log.info("Data extent: x=[%.2f, %.2f], y=[%.2f, %.2f]", x_min, x_max, y_min, y_max)
    log.info("Number of points: %d", len(trait_df))

    # Set style
    sns.set_style("white")

    # Determine figure size based on aspect ratio
    x_range = x_max - x_min
    y_range = y_max - y_min
    aspect = x_range / y_range if y_range > 0 else 1.5

    # Use landscape orientation with appropriate aspect ratio
    if aspect > 1.5:
        fig_width = 16
        fig_height = 10
    else:
        fig_width = 14
        fig_height = int(14 / aspect)

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)

    if HAS_CARTOPY:
        # Use EqualEarth projection (similar to EPSG:6933 Equal Area Cylindrical)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())

        # Add map features with better styling
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', alpha=0.5, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#333333', zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':', edgecolor='#666666', zorder=3)
        ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', alpha=0.5, zorder=0)

        # Create CRS from config (EPSG:6933 = Equal Area Cylindrical)
        # The data will be plotted in its native projection coordinates
        data_crs = ccrs.epsg(6933)

        # Set global extent for EqualEarth projection
        ax.set_global()

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3, linestyle='--', color='gray')
        gl.top_labels = False
        gl.right_labels = False

    else:
        # Fallback to simple matplotlib plot
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Longitude (degrees)', fontsize=12)
        ax.set_ylabel('Latitude (degrees)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(x_min - 5, x_max + 5)
        ax.set_ylim(y_min - 5, y_max + 5)

    # Create color palette for folds
    n_folds = trait_df['fold'].nunique()
    palette = sns.color_palette("Set2", n_folds)

    # Calculate appropriate marker size based on number of points
    n_points = len(trait_df)
    if n_points < 100:
        marker_size = 50
    elif n_points < 500:
        marker_size = 30
    elif n_points < 2000:
        marker_size = 20
    else:
        marker_size = 10

    log.info("Using marker size: %d for %d points", marker_size, n_points)

    # Plot each fold with different color
    for fold_id in sorted(trait_df['fold'].unique()):
        fold_data = trait_df[trait_df['fold'] == fold_id]
        log.info("Plotting fold %d with %d points", fold_id, len(fold_data))

        if HAS_CARTOPY:
            ax.scatter(
                fold_data['x'].values,
                fold_data['y'].values,
                c=[palette[fold_id]],
                label=f'Fold {fold_id}',
                s=marker_size,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.3,
                transform=data_crs,
                zorder=10
            )
        else:
            ax.scatter(
                fold_data['x'].values,
                fold_data['y'].values,
                c=[palette[fold_id]],
                label=f'Fold {fold_id}',
                s=marker_size,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.3,
                zorder=10
            )

    # Add title and legend
    plt.title(f'{trait_col} Spatial Folds', fontsize=16, fontweight='bold', pad=20)

    # Place legend based on data location
    if x_max < 0:  # Data mostly in western hemisphere
        legend_loc = 'upper right'
    elif x_min > 0:  # Data mostly in eastern hemisphere
        legend_loc = 'upper left'
    else:
        legend_loc = 'upper right'

    plt.legend(
        loc=legend_loc,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=1 if n_folds <= 5 else 2,
        fontsize=10,
        markerscale=1.5
    )

    # Save figure
    output_path = splits_dir / f"{trait_col}_folds_map.pdf"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()

    log.info("Saved fold visualization to %s", output_path)


def assign_folds(
    df: pd.DataFrame, n_folds: int, n_iterations: int, data_col: str
) -> pd.DataFrame:
    """
    Assigns folds to the given DataFrame based on similarity scores.

    Args:
        df (pd.DataFrame): The DataFrame to assign folds to.
        n_folds (int): The number of folds to assign.
        n_iterations (int): The number of iterations to perform.
        data_col (str): The column name in the DataFrame containing the data.

    Returns:
        pd.DataFrame: The DataFrame with the folds assigned.

    """
    hexagons = df["hex_id"].unique()

    results = compute(
        *[
            delayed(assign_folds_iteration)(df, n_folds, data_col, hexagons)
            for _ in range(n_iterations)
        ]
    )

    def _compute_best_similarity(
        assignments: list[tuple[float, pd.Series]],
    ) -> tuple[pd.Series, float]:
        best_similarity = None
        best_assignment = pd.Series(dtype=int)

        for similarity, assignment in assignments:
            log.info("Similarity: %e. Current best: %e", similarity, best_similarity)
            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
                best_assignment = assignment

        if best_similarity is None:
            raise ValueError("No best similarity found.")

        return best_assignment, best_similarity

    best_assignment, best_similarity = _compute_best_similarity(results)
    log.info("Best similarity: %e", best_similarity)
    df["fold"] = best_assignment.astype(int)

    return df


def assign_trait_cv_splits(
    trait_col: str,
    cfg: ConfigBox,
    overwrite: bool = False,
) -> None:
    """
    Assign spatial k-fold cross-validation splits to a single trait.

    Args:
        trait_col: Name of the trait column to process
        cfg: Configuration object
        overwrite: Whether to overwrite existing splits

    Outputs:
        Saves splits to {cv_splits_dir}/{trait_col}.parquet with columns [x, y, fold]
    """
    # Get paths from config
    splits_dir = Path(cfg.train.cv_splits.dir_fp)
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_fn = splits_dir / f"{trait_col}.parquet"

    if splits_fn.exists() and not overwrite:
        log.info("Splits for trait %s already exist. Skipping...", trait_col)
        return

    log.info("Processing trait: %s", trait_col)

    # Ensure dask loggers don't interfere with the main logger
    dask_loggers = get_loggers_starting_with("distributed")
    for logger in dask_loggers:
        logging.getLogger(logger).setLevel(logging.WARNING)

    # Load Y data
    y_fp = Path(cfg.train.Y.fp)
    traits_df = dd.read_parquet(y_fp, columns=[trait_col, "x", "y"]).repartition(
        npartitions=100
    )
    trait_df = traits_df.dropna()

    # Compute to get row count for diagnostics
    if isinstance(trait_df, dd.DataFrame):
        trait_df_computed = trait_df.compute()
        n_samples = len(trait_df_computed)
        trait_df = dd.from_pandas(trait_df_computed, npartitions=100)
    else:
        n_samples = len(trait_df)

    log.info("Found %d non-null samples for trait %s", n_samples, trait_col)

    if n_samples < cfg.train.cv_splits.n_splits:
        log.error(
            "Trait %s has only %d samples, which is less than n_splits=%d. "
            "Cannot create CV splits.",
            trait_col, n_samples, cfg.train.cv_splits.n_splits
        )
        raise ValueError(
            f"Insufficient data for trait {trait_col}: {n_samples} samples < "
            f"{cfg.train.cv_splits.n_splits} folds"
        )

    # Load spatial autocorrelation ranges
    ranges_fp = Path(cfg.spatial_autocorr.ranges_fp).resolve()
    ranges = pd.read_parquet(
        ranges_fp,
        columns=["trait", cfg.train.cv_splits.range_stat],
    )

    trait_range = ranges[ranges["trait"] == trait_col][
        cfg.train.cv_splits.range_stat
    ].values[0]

    log.info("Using autocorrelation range of %.2f m for trait %s", trait_range, trait_col)

    # Handle coordinate system-specific range adjustments
    if cfg.crs == "EPSG:4326":
        trait_range_deg = trait_range / 111320
        if trait_range_deg <= cfg.target_resolution:
            log.warning(
                "Trait range of %.2f m is less than or equal to the existing map"
                "resolution of %.2f m. "
                "Using the map resolution for hexagon assignment...",
                trait_range,
                cfg.target_resolution,
            )
            trait_range = cfg.target_resolution * 111320

    if cfg.crs == "EPSG:6933":
        if trait_range <= cfg.target_resolution:
            log.warning(
                "Trait range of %.2f m is less than or equal to the existing map"
                "resolution of %.2f m. "
                "Using the map resolution for hexagon assignment...",
                trait_range,
                cfg.target_resolution,
            )
            trait_range = cfg.target_resolution

        log.info("Reprojecting coordinates to WGS84 for hexagon assignment...")
        # Convert coordinates to EPSG:4326 to get hexagon assignments
        meta = {
            trait_col: "float64",
            "x": "float64",
            "y": "float64",
            "lon": "float64",
            "lat": "float64",
        }
        trait_df = trait_df.map_partitions(
            reproject_xy_to_geo, from_crs=cfg.crs, meta=meta
        )
        trait_df = trait_df.rename(
            columns={"x": "x_old", "y": "y_old", "lat": "y", "lon": "x"}
        )

    # Convert autocorrelation range to H3 resolution
    h3_res = acr_to_h3_res(trait_range)
    log.info("Using H3 resolution %d for hexagon assignment", h3_res)

    # Assign hexagon IDs
    trait_df = assign_hexagons(trait_df, h3_res, dask=True).reset_index(drop=True)

    if cfg.crs == "EPSG:6933":
        # Revert back to the original coordinates
        trait_df = trait_df.drop(columns=["x", "y"]).rename(
            columns={"x_old": "x", "y_old": "y"}
        )

    if isinstance(trait_df, dd.DataFrame):
        log.info("Computing trait dask DataFrame...")
        trait_df = trait_df.compute()  # pyright: ignore[reportCallIssue]

    # Check number of unique hexagons
    n_hexagons = trait_df["hex_id"].nunique()
    log.info(
        "Trait %s: %d samples assigned to %d unique hexagons (%.2f samples/hexagon)",
        trait_col, len(trait_df), n_hexagons, len(trait_df) / n_hexagons
    )

    if n_hexagons < cfg.train.cv_splits.n_splits:
        log.warning(
            "Trait %s has only %d unique hexagons, which is less than n_splits=%d. "
            "Some folds may be empty or very small.",
            trait_col, n_hexagons, cfg.train.cv_splits.n_splits
        )

    # Assign folds based on similarity optimization
    log.info("Assigning the best folds...")
    trait_df = assign_folds(
        trait_df,
        cfg.train.cv_splits.n_splits,
        cfg.train.cv_splits.n_sims,
        trait_col,
    )

    # Save splits
    splits_data = trait_df[["x", "y", "fold"]].drop_duplicates(subset=["x", "y"]).reset_index(
        drop=True
    )
    splits_data.to_parquet(splits_fn, compression="zstd")

    log.info("Saved splits for trait %s to %s", trait_col, splits_fn)

    # Create visualization
    try:
        visualize_folds(splits_data, trait_col, splits_dir, cfg)
    except Exception as e:
        log.warning("Failed to create fold visualization: %s", e)
        log.warning("Continuing without visualization...")


if __name__ == "__main__":
    main()
