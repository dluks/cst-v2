"""Featurize EO data for prediction and AoA calculation."""

import argparse
import math
from pathlib import Path

import dask.dataframe as dd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr
from dask import config

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import (
    compute_partitions,
    load_rasters_parallel,
)


def cli() -> argparse.Namespace:
    """Command line interface for featurizing EO data for prediction and AoA
    calculation."""
    parser = argparse.ArgumentParser(
        description="Featurize EO data for prediction and AoA calculation."
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        help="Path to a params.yaml.",
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite existing files."
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Main function for featurizing EO data for prediction and AoA calculation."""
    params_path = Path(args.params).absolute()
    log.info("Loading config from %s", params_path)
    cfg = get_config(params_path)
    syscfg = cfg[detect_system()]["build_predict"]

    if args.debug:
        log.info("Running in debug mode...")

    out_fp = Path(cfg.x_dir) / cfg.x_fn
    # Check if output already exists and handle overwrite
    if out_fp.exists() and not args.overwrite:
        log.info("Output file already exists: %s", out_fp)
        log.info("Use --overwrite flag to overwrite existing files.")
        return
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    log.info("Initializing Dask client...")
    client, _ = init_dask(
        dashboard_address=cfg.dask_dashboard,
        n_workers=syscfg.n_workers,
        memory_limit=syscfg.memory_limit,
    )
    config.set({"array.slicing.split_large_chunks": False})

    log.info("Getting filenames...")

    ds_fns = {}

    for dataset in cfg.datasets:
        ds_fns[dataset] = list(
            (Path(cfg.interim.out_dir).resolve() / dataset).glob("*.tif")
        )

    eo_fns = [fn for ds_fns in ds_fns.values() for fn in ds_fns]

    if args.debug:
        eo_fns = eo_fns[:2]

    log.info("Loading rasters...")
    ds = load_rasters_parallel(eo_fns, nchunks=syscfg.n_chunks)

    log.info("Converting to Dask DataFrame...")
    ddf = eo_ds_to_ddf(ds, thresh=cfg.missing_val_thresh)

    log.info("Computing partitions...")
    df = compute_partitions(ddf).reset_index(drop=True).set_index(["y", "x"])

    log.info("Closing Dask client...")
    close_dask(client)

    log.info("Saving DataFrame to disk...")
    df.to_parquet(out_fp, compression="zstd", compression_level=19)

    log.info("Generating report...")
    report_fp = out_fp.parent / "report.md"
    # Also check for report overwrite
    if report_fp.exists() and not args.overwrite:
        log.info("Report file already exists: %s", report_fp)
        log.info("Use --overwrite flag to overwrite existing files.")
        return

    _build_report(df, report_fp)

    log.info("Done!")


def eo_ds_to_ddf(ds: xr.Dataset, thresh: float, sample: float = 1.0) -> dd.DataFrame:
    """
    Convert an EO dataset to a Dask DataFrame.

    Parameters:
        ds (xr.Dataset): The input EO dataset.
        dtypes (dict[str, str]): A dictionary mapping variable names to their data
            types.

    Returns:
        dd.DataFrame: The converted Dask DataFrame.
    """

    return (
        ds.to_dask_dataframe()
        .sample(frac=sample)
        .drop(columns=["band", "spatial_ref"])
        .dropna(
            thresh=math.ceil(len(ds.data_vars) * (1 - thresh)),
            subset=list(ds.data_vars),
        )
    )


def _generate_figure(df: pd.DataFrame, figure_fp: Path) -> None:
    """
    Generate a figure with distribution plots for each feature.

    Args:
        df (pd.DataFrame): Input dataframe containing features
        figure_fp (Path): Path where to save the figure
    """
    # Set matplotlib and seaborn style
    plt.style.use("default")
    sns.set_palette("husl")

    # Exclude coordinate columns (y, x) from plotting
    feature_cols = [col for col in df.columns if col not in ["y", "x"]]

    # Calculate number of rows needed (5 columns)
    n_features = len(feature_cols)
    n_cols = 5
    n_rows = math.ceil(n_features / n_cols)

    # Set up the figure with appropriate size
    fig_width = 20  # 4 inches per column
    fig_height = 4 * n_rows  # 4 inches per row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Flatten axes for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    # Plot each feature
    for i, feature in enumerate(feature_cols):
        ax = axes_flat[i]

        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[feature]):
            ax.text(
                0.5,
                0.5,
                f"Non-numeric\n{feature}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(feature)
            continue

        # Create KDE plot
        sns.kdeplot(data=df, x=feature, ax=ax, fill=True, alpha=0.6)

        # Set title and labels
        ax.set_title(f"{feature}", fontsize=10)
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)

        # Format tick labels
        ax.tick_params(axis="both", which="major", labelsize=8)

    # Hide unused subplots
    for i in range(n_features, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Apply tight layout
    plt.tight_layout()

    # Save figure at 300 DPI
    plt.savefig(figure_fp, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_report(df: pd.DataFrame, report_fp: Path) -> None:
    """
    Generate a markdown report with feature statistics and distribution plots.

    Args:
        df (pd.DataFrame): Input dataframe containing features
        report_fp (Path): Path where to save the report
    """
    # Define figure path (next to report)
    figure_fp = report_fp.parent / "feature_distributions.png"

    # Generate the figure
    _generate_figure(df, figure_fp)

    # Calculate statistics for each feature (excluding coordinate columns)
    feature_cols = [col for col in df.columns if col not in ["y", "x"]]

    # Calculate statistics
    stats_list = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = {
                "Feature": col,
                "Count": df[col].count(),
                "Mean": df[col].mean(),
                "Std": df[col].std(),
                "Median": df[col].median(),
                "Min": df[col].min(),
                "Max": df[col].max(),
                "Missing": df[col].isna().sum(),
            }
        else:
            # For non-numeric columns
            stats = {
                "Feature": col,
                "Count": df[col].count(),
                "Mean": None,
                "Std": None,
                "Median": None,
                "Min": None,
                "Max": None,
                "Missing": df[col].isna().sum(),
            }
        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)

    # Format numeric values for display
    for col in ["Mean", "Std", "Median", "Min", "Max", "Missing"]:
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].apply(
                lambda x: f"{x:.2f}"
                if pd.notnull(x) and isinstance(x, (int, float))
                else str(x)
            )

    # Create markdown report
    report_content = f"""# Feature Statistics and Distributions Report

Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Overview

- **Total samples**: {len(df):,}
- **Total features**: {len(feature_cols):,}
- **Coordinate columns**: y, x (excluded from statistics)

## Feature Statistics

The table below shows descriptive statistics for each feature:

| Feature | Count | Mean | Std | Median | Min | Max | Missing |
|---------|-------|------|-----|--------|-----|-----|---------|
"""

    # Add each row to the table
    for _, row in stats_df.iterrows():
        report_content += f"| {row['Feature']} | {row['Count']:,} | {row['Mean']} | {row['Std']} | {row['Median']} | {row['Min']} | {row['Max']} | {row['Missing']} |\n"  # noqa: E501

    # Add figure section
    report_content += f"""

## Feature Distributions

The figure below shows the distribution of values for each feature using kernel density estimation (KDE) plots.

![Feature Distributions]({figure_fp.name})

*Figure: Distribution plots for all features. Each subplot shows the kernel density estimate of the feature values.*

## Notes

- Statistics are calculated only for numeric features
- Non-numeric features show count information only
- KDE plots are only generated for numeric features
- The figure is saved as a high-resolution PNG (300 DPI) for clear visualization
"""  # noqa: E501

    # Write report to file
    with open(report_fp, "w") as f:
        f.write(report_content)


if __name__ == "__main__":
    args = cli()
    main(args)
