"""
Merge individual trait Y files into final Y.parquet.

This script reads all individual trait parquet files from the tmp directory,
merges them on spatial coordinates and source, and writes the final Y.parquet
file to the features directory.
"""

import argparse
import os
import shutil
from math import ceil
from pathlib import Path

import dask.dataframe as dd
import matplotlib
import pandas as pd
from dask.distributed import Client

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.conf.conf import get_config
from src.conf.environment import detect_system, log


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge individual trait Y files into final Y.parquet."
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        help="Path to the parameters file.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    return parser.parse_args()


def _generate_figure(df: pd.DataFrame, trait_cols: list[str], figure_fp: Path) -> None:
    """
    Generate a figure with KDE distribution plots for each trait.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing trait data
    trait_cols : list[str]
        List of trait column names to plot
    figure_fp : Path
        Path to save the figure
    """
    log.info("Generating trait distribution plots...")

    # Set plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Calculate grid dimensions (5 columns)
    n_traits = len(trait_cols)
    n_cols = 5
    n_rows = ceil(n_traits / n_cols)

    # Create figure with subplots
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(20, 4 * n_rows), constrained_layout=True
    )

    # Flatten axes array for easier iteration
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Plot each trait
    for idx, trait in enumerate(trait_cols):
        ax = axes[idx]

        # Check if trait data is valid
        trait_data = df[trait].dropna()

        if len(trait_data) == 0:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(trait)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Check if data is numeric
        if not pd.api.types.is_numeric_dtype(trait_data):
            ax.text(
                0.5,
                0.5,
                "Non-numeric data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(trait)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Plot KDE
        try:
            sns.kdeplot(data=df, x=trait, ax=ax, fill=True, alpha=0.6)
            ax.set_title(trait)
            ax.set_xlabel("")
        except Exception as e:
            log.warning("Failed to plot trait '%s': %s", trait, e)
            ax.text(
                0.5,
                0.5,
                f"Plot failed:\n{str(e)[:50]}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(trait)
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide unused subplots
    for idx in range(n_traits, len(axes)):
        axes[idx].axis("off")

    # Save figure
    plt.savefig(figure_fp, dpi=300, bbox_inches="tight")
    plt.close()

    log.info("✓ Trait distribution plots saved to %s", figure_fp)


def _build_report(
    df: pd.DataFrame, trait_cols: list[str], report_fp: Path, figure_fp: Path
) -> None:
    """
    Build a markdown report with trait statistics and distribution plots.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing trait data
    trait_cols : list[str]
        List of trait column names
    report_fp : Path
        Path to save the report
    figure_fp : Path
        Path to the distribution figure
    """
    log.info("Building trait statistics report...")

    # Generate figure first
    _generate_figure(df, trait_cols, figure_fp)

    # Calculate statistics for each trait
    stats_rows = []
    for trait in trait_cols:
        trait_data = df[trait].dropna()
        n_total = len(df)
        n_valid = len(trait_data)
        n_missing = n_total - n_valid

        if n_valid > 0 and pd.api.types.is_numeric_dtype(trait_data):
            stats_rows.append(
                {
                    "Trait": trait,
                    "Count": n_valid,
                    "Mean": trait_data.mean(),
                    "Std": trait_data.std(),
                    "Median": trait_data.median(),
                    "Min": trait_data.min(),
                    "Max": trait_data.max(),
                    "Missing": n_missing,
                }
            )
        else:
            stats_rows.append(
                {
                    "Trait": trait,
                    "Count": n_valid,
                    "Mean": None,
                    "Std": None,
                    "Median": None,
                    "Min": None,
                    "Max": None,
                    "Missing": n_missing,
                }
            )

    stats_df = pd.DataFrame(stats_rows)

    # Build markdown report
    report_lines = [
        "# Y Data Report",
        "",
        "## Dataset Overview",
        "",
        f"- **Total samples**: {len(df):,}",
        f"- **Total traits**: {len(trait_cols)}",
        f"- **Trait list**: {', '.join(trait_cols)}",
        "",
        "## Trait Statistics",
        "",
    ]

    # Format statistics table
    report_lines.append("| Trait | Count | Mean | Std | Median | Min | Max | Missing |")
    report_lines.append("|-------|-------|------|-----|--------|-----|-----|---------|")

    for _, row in stats_df.iterrows():
        trait = row["Trait"]
        count = f"{row['Count']:,}" if pd.notna(row["Count"]) else "N/A"

        # Format numeric values with 4 decimal places
        if pd.notna(row["Mean"]):
            mean = f"{row['Mean']:.4f}"
            std = f"{row['Std']:.4f}"
            median = f"{row['Median']:.4f}"
            min_val = f"{row['Min']:.4f}"
            max_val = f"{row['Max']:.4f}"
        else:
            mean = std = median = min_val = max_val = "N/A"

        missing = f"{row['Missing']:,}" if pd.notna(row["Missing"]) else "N/A"

        report_lines.append(
            f"| {trait} | {count} | {mean} | {std} | {median} | "
            f"{min_val} | {max_val} | {missing} |"
        )

    report_lines.extend(
        [
            "",
            "## Trait Distributions",
            "",
            f"![Trait Distributions]({figure_fp.name})",
            "",
            "*Figure: Kernel Density Estimation (KDE) plots showing the distribution "
            "of each trait. Each subplot represents the probability density function "
            "of trait values across all samples.*",
            "",
            "## Notes",
            "",
            "- Statistics are calculated from all non-missing values for each trait.",
            "- Missing values are counted separately and do not affect the statistics.",
            "- The coordinate columns (x, y) and source column are excluded from "
            "this report.",
            "- Reliability weight columns (ending in `_reliability`) are preserved "
            "in the dataset but excluded from trait statistics.",
            "- For very large datasets, statistics may be calculated from a random "
            "sample for performance reasons.",
            "",
        ]
    )

    # Write report
    with open(report_fp, "w") as f:
        f.write("\n".join(report_lines))

    log.info("✓ Report saved to %s", report_fp)


def main(args: argparse.Namespace | None = None) -> None:
    """Main function to merge trait files."""
    args = cli() if args is None else args
    cfg = get_config(params_path=args.params)
    syscfg = cfg[detect_system()]["merge_y_traits"]

    # Set up paths
    proj_root = os.environ.get("PROJECT_ROOT")
    if proj_root is None:
        raise ValueError("PROJECT_ROOT environment variable is not set")
    tmp_dir = Path(proj_root) / cfg.tmp_dir / "y_traits"
    out_fn = Path(cfg.train.dir, cfg.product_code, cfg.train.Y.fn)
    report_fp = out_fn.parent / "report.md"
    figure_fp = out_fn.parent / "trait_distributions.png"

    # Smart overwrite logic: check what needs to be generated
    y_exists = out_fn.exists()
    report_exists = report_fp.exists() and figure_fp.exists()

    # Determine what to generate
    generate_y = args.overwrite or not y_exists
    generate_report = args.overwrite or not report_exists

    # Early exit if nothing needs to be done
    if not generate_y and not generate_report:
        log.info("All outputs already exist. Use --overwrite to regenerate.")
        log.info("  Y data: %s", out_fn)
        log.info("  Report: %s", report_fp)
        log.info("  Figure: %s", figure_fp)
        return

    # If we need to generate Y data
    if generate_y:
        log.info("Generating Y data...")

        # Find all trait parquet files
        trait_files = sorted(list(tmp_dir.glob("*.parquet")))

        if not trait_files:
            log.error("No trait files found in %s", tmp_dir)
            return

        log.info("Found %d trait files to merge", len(trait_files))

        # Create output directory
        out_fn.parent.mkdir(parents=True, exist_ok=True)

        # Use dask for memory-efficient merging
        with Client(
            n_workers=syscfg.n_workers,
            threads_per_worker=syscfg.threads_per_worker,
            memory_limit=syscfg.memory_limit,
        ):
            log.info("Reading trait files...")

            # Read the first file to start
            y_df = dd.read_parquet(trait_files[0])
            log.info("Loaded base file: %s", trait_files[0].stem)

            # Merge remaining files
            for i, trait_file in enumerate(trait_files[1:], start=2):
                log.info("Merging file %d/%d: %s", i, len(trait_files), trait_file.stem)

                trait_df = dd.read_parquet(trait_file)

                # Merge on common spatial coordinates and source
                y_df = y_df.merge(trait_df, on=["x", "y", "source"], how="outer")

            log.info("All files merged. Writing to disk...")

            # Write to parquet with compression
            y_df.to_parquet(out_fn, compression="zstd", write_index=False)

            log.info("✓ Successfully wrote Y data to %s", out_fn)

        # Clean up temporary files
        log.info("Cleaning up temporary files...")
        try:
            shutil.rmtree(tmp_dir)
            log.info("✓ Temporary directory removed: %s", tmp_dir)
        except Exception as e:
            log.warning("Failed to remove temporary directory %s: %s", tmp_dir, e)
    else:
        log.info("Y data already exists, skipping generation: %s", out_fn)

    # Generate report if needed
    if generate_report:
        log.info("Generating trait statistics report and distribution plots...")

        # Load Y data for reporting
        log.info("Loading Y data for report generation...")
        y_df = dd.read_parquet(out_fn)

        # Sample if dataset is very large for performance
        REPORT_SAMPLE_SIZE = 50000
        df_len = len(y_df)

        if df_len > REPORT_SAMPLE_SIZE:
            log.info(
                "Dataset has %s rows. Sampling %s rows for report generation...",
                f"{df_len:,}",
                f"{REPORT_SAMPLE_SIZE:,}",
            )
            sample_frac = REPORT_SAMPLE_SIZE / df_len
            y_df_report = y_df.sample(frac=sample_frac, random_state=42).compute()
        else:
            log.info("Converting to pandas for report generation...")
            y_df_report = y_df.compute()

        # Get trait columns (exclude x, y, source, and reliability columns)
        trait_cols = [
            col for col in y_df_report.columns
            if col not in ["x", "y", "source"] and not col.endswith("_reliability")
        ]

        if not trait_cols:
            log.warning("No trait columns found in Y data. Cannot generate report.")
        else:
            log.info(
                "Found %d traits for report: %s", len(trait_cols), ", ".join(trait_cols)
            )

            # Generate report and plots
            _build_report(y_df_report, trait_cols, report_fp, figure_fp)
    else:
        log.info("Report and figure already exist, skipping generation.")
        log.info("  Report: %s", report_fp)
        log.info("  Figure: %s", figure_fp)

    log.info("Done!")


if __name__ == "__main__":
    main()
