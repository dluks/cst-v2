"""Build TRY6 trait dataset from gapfilled TRY data.

This script loads the TRY6 gapfilled trait data from a nested zip file,
standardizes trait IDs, cleans and formats species names, aggregates to
species-level means, applies optional transformation, and saves both the
transformed dataset and the fitted transformer for downstream use.
"""

from __future__ import annotations

import argparse
import math
import pickle
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PowerTransformer

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.match_pfts import match_pfts


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build TRY6 traits.")
    parser.add_argument(
        "-p", "--params", type=str, default=None, help="Path to params.yaml"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Load, clean, aggregate, transform, and save TRY6 traits."""
    cfg = (
        get_config(params_path=args.params) if args.params is not None else get_config()
    )

    out_fp = Path(cfg.traits.interim_out)

    # Load from nested zip structure
    df = _load_try6_zip(
        Path(cfg.traits.raw), cfg.traits.zip, cfg.traits.zipfile_csv, cfg.traits.names
    )

    # Standardize trait IDs (X.X4. -> X4)
    df = _standardize_trait_ids(df)

    # Clean species names: drop non-binomial, lowercase
    df = _clean_species_names(df)

    # Filter outliers if specified
    if hasattr(cfg.traits, "quantile_range") and cfg.traits.quantile_range is not None:
        df = _filter_outliers(df, cfg.traits.names, cfg.traits.quantile_range)

    # Aggregate to species-level summary (mean or median)
    aggregation = (
        cfg.traits.aggregation if hasattr(cfg.traits, "aggregation") else "mean"
    )
    df = _aggregate_species_traits(df, cfg.traits.names, aggregation)

    # Track species count before PFT matching
    n_species_before_pft = df.shape[0]

    # Match with growth forms (coarse PFTs)
    df = match_pfts(
        df,
        cfg.traits.pfts,
        cfg.traits.harmonization_fp,
        cfg.traits.pfts_threshold,
        harmonization_col="groNameIn",  # TRY6 uses TRY growth form names
    )

    # Calculate species dropped during PFT matching
    n_species_after_pft = df.shape[0]
    n_species_dropped = n_species_before_pft - n_species_after_pft
    pct_dropped = (
        100 * n_species_dropped / n_species_before_pft
        if n_species_before_pft > 0
        else 0.0
    )

    if cfg.traits.transform == "power":
        # Transform traits
        df, transformer = _power_transform(df, cfg.traits.names)
        _save_outputs(df=df, out_fp=out_fp, transformer=transformer)
    else:
        _save_outputs(df=df, out_fp=out_fp)

    # Generate report
    log.info("Generating trait statistics report...")
    report_fp = out_fp.parent / "report.md"
    _build_report(
        df,
        cfg.traits.names,
        report_fp,
        n_species_before_pft=n_species_before_pft,
        n_species_dropped=n_species_dropped,
        pct_dropped=pct_dropped,
        aggregation=aggregation,
    )


def _load_try6_zip(
    fp: Path, zip_fn: str, zipfile_csv: str, traits: list[str]
) -> pd.DataFrame:
    """Load the TRY6 nested zip file and return a minimally cleaned DataFrame.

    TRY6 has a nested zip structure: outer zip contains an inner zip that
    contains the CSV file.

    Returns a DataFrame with columns:
    - Species (original species names)
    - X.X4., X.X6., etc. (trait columns in TRY6 format)
    """
    zip_path = fp / zip_fn

    if not zip_path.exists():
        raise FileNotFoundError(
            f"TRY6 zip file not found at {zip_path}. Place file at this path."
        )

    log.info("Loading TRY6 traits from %s", zip_path)

    # TRY6 column format: X.X4., X.X6., etc.
    trait_cols_try6 = [f"X.{t}." for t in traits]
    required_cols = ["Species"] + trait_cols_try6

    log.info("Extracting from nested zip structure: %s", zipfile_csv)

    with (
        zipfile.ZipFile(zip_path, "r") as zip_ref,
        zip_ref.open(zipfile_csv) as nested_zip_file,
        zipfile.ZipFile(nested_zip_file, "r") as nested_zip_ref,
    ):
        # Get the CSV file name (should be only one file in nested zip)
        csv_filename = nested_zip_ref.namelist()[0]
        log.info("Reading CSV file: %s", csv_filename)

        with nested_zip_ref.open(csv_filename) as csvfile:
            df = pd.read_csv(csvfile, encoding="latin-1", usecols=required_cols)

    log.info("Loaded %d rows with %d trait columns", df.shape[0], len(trait_cols_try6))

    # Cast dtypes
    df = df.astype(
        {"Species": "string[pyarrow]", **{t: "float32" for t in trait_cols_try6}}
    )

    return df


def _standardize_trait_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize trait IDs from TRY6 format (X.X4.) to standard format (X4).

    TRY6 uses column names like 'X.X4.', 'X.X6.', etc.
    We standardize to 'X4', 'X6', etc. by extracting the middle segment.
    """
    old_trait_cols = df.columns[df.columns.str.startswith("X.")]
    log.info("Standardizing %d trait column IDs", len(old_trait_cols))

    # Extract middle segment: X.X4. -> X4
    new_trait_cols = [col.split(".")[1] for col in old_trait_cols]

    df = df.rename(columns=dict(zip(old_trait_cols, new_trait_cols)))
    log.info("Standardized trait IDs: %s", new_trait_cols[:5] + ["..."])

    return df


def _clean_species_names(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-binomial taxa and standardize species names.

    Keeps only rows where the species string contains exactly two words
    (binomial nomenclature), then converts to lowercase and renames to
    'speciesname'.
    """
    if "Species" not in df.columns:
        raise ValueError("'Species' column not found during cleaning stage")

    before = df.shape[0]
    df = (
        df.copy()
        .query("Species.str.split().str.len() == 2")
        .assign(speciesname=lambda d: d["Species"].str.lower())
        .drop(columns=["Species"])
    )
    after = df.shape[0]
    log.info(
        "Filtered non-binomial taxa: kept %d of %d rows (%.2f%%)",
        after,
        before,
        100 * after / max(1, before),
    )
    return df


def _filter_outliers(
    df: pd.DataFrame, trait_cols: list[str], quantile_range: tuple[float, float]
) -> pd.DataFrame:
    """Filter out outliers based on quantile range on a per-trait basis.

    Sets outlier values to NaN for each trait independently, rather than dropping
    entire rows. This is appropriate for trait data where we want to remove
    extreme values per trait without losing all data for a species/observation.

    Args:
        df: DataFrame with trait data
        trait_cols: List of trait column names to filter
        quantile_range: Tuple of (lower, upper) quantiles (e.g., (0.01, 0.99))

    Returns:
        DataFrame with outliers set to NaN per trait
    """
    df_filtered = df.copy()
    lower_q, upper_q = quantile_range

    log.info(
        "Filtering outliers per-trait with quantile range: %s on %d traits",
        quantile_range,
        len(trait_cols),
    )

    total_values_filtered = 0

    for col in trait_cols:
        if col not in df.columns:
            continue

        # Calculate quantile bounds for this trait
        lower_bound = df[col].quantile(lower_q)
        upper_bound = df[col].quantile(upper_q)

        # Count outliers
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        n_outliers = outliers.sum()
        total_values_filtered += n_outliers

        # Set outliers to NaN
        df_filtered.loc[outliers, col] = pd.NA

        if n_outliers > 0:
            log.debug(
                "Trait %s: set %d values (%.1f%%) to NaN [bounds: %.4f, %.4f]",
                col,
                n_outliers,
                100 * n_outliers / df[col].notna().sum(),
                lower_bound,
                upper_bound,
            )

    total_values = sum(df[col].notna().sum() for col in trait_cols if col in df.columns)
    pct_filtered = 100 * total_values_filtered / total_values if total_values > 0 else 0

    log.info(
        "Filtered %d outlier values (%.2f%% of all trait values)",
        total_values_filtered,
        pct_filtered,
    )

    return df_filtered


def _aggregate_species_traits(
    df: pd.DataFrame, trait_cols: list[str], aggregation: str = "mean"
) -> pd.DataFrame:
    """Aggregate trait values to species-level summary statistic.

    TRY6 data contains multiple observations per species. We compute the
    specified summary statistic (mean or median) for each species.

    Args:
        df: DataFrame with trait data
        trait_cols: List of trait column names
        aggregation: Aggregation method - either "mean" or "median"

    Returns:
        DataFrame with one row per species
    """
    if aggregation not in ["mean", "median"]:
        raise ValueError(f"Aggregation must be 'mean' or 'median', got '{aggregation}'")

    log.info(
        "Aggregating to species-level %ss for %d traits", aggregation, len(trait_cols)
    )

    before = df.shape[0]

    if aggregation == "mean":
        df_agg = df.groupby("speciesname").mean().reset_index()
    else:  # median
        df_agg = df.groupby("speciesname").median().reset_index()

    after = df_agg.shape[0]

    log.info(
        "Aggregated %d observations to %d species (%.1f observations per species)",
        before,
        after,
        before / max(1, after),
    )

    return df_agg


def _power_transform(
    df: pd.DataFrame, traits: list[str]
) -> tuple[pd.DataFrame, PowerTransformer]:
    """Apply Yeo-Johnson power transform to traits and return the result.

    The transformer is fit jointly across the trait columns and applied to
    those columns only. Returns the transformed DataFrame (same columns) and the
    fitted transformer.
    """
    for col in traits:
        if col not in df.columns:
            raise ValueError(f"Expected trait column '{col}' not found")

    # Separate speciesname and harmonization columns from traits
    trait_values = df[traits]
    log.info("Applying Yeo-Johnson to %d traits", len(traits))
    pt = PowerTransformer(method="yeo-johnson")
    transformed = pt.fit_transform(trait_values)

    df_t = df.copy()
    df_t[traits] = transformed

    log.info("Transformation lambdas: %s", pt.lambdas_[:5].tolist() + ["..."])

    return df_t, pt


def _save_outputs(
    df: pd.DataFrame, out_fp: Path, transformer: PowerTransformer | None = None
) -> None:
    """Save transformed traits as parquet and transformer as pickle to out_dir."""
    out_dir = out_fp.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if transformer is not None and isinstance(transformer, PowerTransformer):
        transformer_fp = out_fp.with_suffix(".pkl")
        with open(transformer_fp, "wb") as f:
            pickle.dump(transformer, f)
        log.info("Saved PowerTransformer to %s", transformer_fp)

    # Save data (speciesname + transformed traits + harmonization columns)
    df.to_parquet(out_fp, index=False, compression="zstd")
    log.info(
        "Saved %s TRY6 traits to %s (%d species, %d columns)",
        "transformed" if transformer is not None else "untransformed",
        out_fp,
        df.shape[0],
        df.shape[1],
    )


def _generate_figure(df: pd.DataFrame, trait_cols: list[str], figure_fp: Path) -> None:
    """Generate a figure with distribution plots for each trait.

    Args:
        df: Input dataframe containing traits
        trait_cols: List of trait column names to plot
        figure_fp: Path where to save the figure
    """
    # Set matplotlib and seaborn style
    plt.style.use("default")
    sns.set_palette("husl")

    # Calculate number of rows needed (5 columns)
    n_traits = len(trait_cols)
    n_cols = 5
    n_rows = math.ceil(n_traits / n_cols)

    # Set up the figure with appropriate size
    fig_width = 20  # 4 inches per column
    fig_height = 4 * n_rows  # 4 inches per row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Flatten axes for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    # Plot each trait
    for i, trait in enumerate(trait_cols):
        ax = axes_flat[i]

        # Skip if trait not in dataframe
        if trait not in df.columns:
            ax.text(
                0.5,
                0.5,
                f"Missing\n{trait}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(trait)
            continue

        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[trait]):
            ax.text(
                0.5,
                0.5,
                f"Non-numeric\n{trait}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(trait)
            continue

        # Create KDE plot
        sns.kdeplot(data=df, x=trait, ax=ax, fill=True, alpha=0.6)

        # Set title and labels
        ax.set_title(f"{trait}", fontsize=10)
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)

        # Format tick labels
        ax.tick_params(axis="both", which="major", labelsize=8)

    # Hide unused subplots
    for i in range(n_traits, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Apply tight layout
    plt.tight_layout()

    # Save figure at 300 DPI
    plt.savefig(figure_fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved trait distribution figure to %s", figure_fp)


def _build_report(
    df: pd.DataFrame,
    trait_cols: list[str],
    report_fp: Path,
    n_species_before_pft: int,
    n_species_dropped: int,
    pct_dropped: float,
    aggregation: str = "mean",
) -> None:
    """Generate a markdown report with trait statistics and distribution plots.

    Args:
        df: Input dataframe containing traits
        trait_cols: List of trait column names to report on
        report_fp: Path where to save the report
        n_species_before_pft: Number of species before PFT matching
        n_species_dropped: Number of species dropped during PFT matching
        pct_dropped: Percentage of species dropped during PFT matching
        aggregation: Aggregation method used (mean or median)
    """
    # Define figure path (next to report)
    figure_fp = report_fp.parent / "trait_distributions.png"

    # Generate the figure
    _generate_figure(df, trait_cols, figure_fp)

    # Get PFT distribution if pft column exists
    pft_info = ""
    if "pft" in df.columns:
        pft_counts = df["pft"].value_counts()
        pft_info = "\n### PFT Distribution\n\n"
        pft_info += "| PFT | Count | Percentage |\n"
        pft_info += "|-----|-------|------------|\n"
        for pft, count in pft_counts.items():
            pct = 100 * count / len(df)
            pft_info += f"| {pft} | {count:,} | {pct:.1f}% |\n"

    # Calculate statistics for each trait
    stats_list = []
    for col in trait_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            stats = {
                "Trait": col,
                "Count": df[col].count(),
                "Mean": df[col].mean(),
                "Std": df[col].std(),
                "Median": df[col].median(),
                "Min": df[col].min(),
                "Max": df[col].max(),
                "Missing": df[col].isna().sum(),
            }
        else:
            # For missing or non-numeric traits
            stats = {
                "Trait": col,
                "Count": df[col].count() if col in df.columns else 0,
                "Mean": None,
                "Std": None,
                "Median": None,
                "Min": None,
                "Max": None,
                "Missing": df[col].isna().sum() if col in df.columns else len(df),
            }
        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)

    # Format numeric values for display
    for col in ["Mean", "Std", "Median", "Min", "Max"]:
        if col in stats_df.columns:
            stats_df[col] = stats_df[col].apply(
                lambda x: f"{x:.4f}"
                if pd.notnull(x) and isinstance(x, (int, float))
                else str(x)
            )

    # Create markdown report
    report_content = f"""# TRY6 Trait Statistics Report

Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Overview

- **Aggregation method**: {aggregation.capitalize()} (per species)
- **Species before PFT matching**: {n_species_before_pft:,}
- **Species after PFT matching**: {len(df):,}
- **Species dropped during PFT matching**: {n_species_dropped:,} ({pct_dropped:.1f}%)
- **Total traits**: {len(trait_cols):,}
- **Additional columns**: speciesname, GBIFKeyGBIF, nameOutWCVP, nameOutWFO, pft
{pft_info}
## Trait Statistics

The table below shows descriptive statistics for each trait:

| Trait | Count | Mean | Std | Median | Min | Max | Missing |
|-------|-------|------|-----|--------|-----|-----|---------|
"""

    # Add each row to the table
    for _, row in stats_df.iterrows():
        report_content += f"| {row['Trait']} | {row['Count']:,} | {row['Mean']} | {row['Std']} | {row['Median']} | {row['Min']} | {row['Max']} | {row['Missing']} |\n"  # noqa: E501

    # Add figure section
    report_content += f"""

## Trait Distributions

The figure below shows the distribution of values for each trait using kernel density estimation (KDE) plots.

![Trait Distributions]({figure_fp.name})

*Figure: Distribution plots for all traits. Each subplot shows the kernel density estimate of the trait values.*

## Notes

- Statistics are calculated only for numeric traits
- Missing traits or non-numeric traits show count information only
- KDE plots are only generated for numeric traits present in the dataset
- The figure is saved as a high-resolution PNG (300 DPI) for clear visualization
- Trait values shown are after species-level aggregation ({aggregation} per species)
"""  # noqa: E501

    # Write report to file
    with open(report_fp, "w") as f:
        f.write(report_content)

    log.info("Saved trait statistics report to %s", report_fp)


if __name__ == "__main__":
    log.info("Starting TRY6 trait processing")
    main(cli())
    log.info("TRY6 trait processing completed successfully!")
