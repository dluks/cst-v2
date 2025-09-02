"""Dot plot visualization of fold-wise Pearson R across traits and trait sets.

This script creates a landscape-oriented dot plot showing the fold-wise mean Pearson R
and standard deviation for each trait across trait sets (SCI, COMB, CIT) for 1km
resolution models that have been power transformed.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats
from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.plotting_utils import add_human_readables, set_font

# Color scheme matching other figures
TRAIT_SET_ORDER = ["SCI", "COMB", "CIT"]
tricolor_palette = sns.color_palette(["#b0b257", "#66a9aa", "#b95fa1"])
CFG = get_config()


def cli() -> argparse.Namespace:
    """Command line interface for the script."""
    parser = argparse.ArgumentParser(
        description="Create dot plot of fold-wise Pearson R across traits and trait sets."
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="./results/figures/trait-pearsonr-dotplot.png",
        help="Output file path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output figure.",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="20,8",
        help="Figure size (width, height) in inches as 'width,height'.",
    )
    parser.add_argument(
        "--error_type",
        type=str,
        choices=["std", "ci95"],
        default="std",
        help="Type of error bars: 'std' for standard deviation, 'ci95' for 95%% confidence interval.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Main function to create the dot plot visualization."""
    set_font("FreeSans")

    log.info("Loading and processing results data...")
    results_df = load_and_process_data()

    # Parse figsize if provided
    if args and args.figsize:
        try:
            width, height = map(float, args.figsize.split(","))
            figsize = (width, height)
        except (ValueError, AttributeError):
            log.warning("Invalid figsize format, using default (20, 8)")
            figsize = (20, 8)
    else:
        figsize = (20, 8)

    log.info("Creating dot plot visualization...")
    error_type = args.error_type if args else "std"
    with sns.plotting_context("paper", 1.5):
        fig = create_trait_pearsonr_dotplot(
            results_df, figsize=figsize, error_type=error_type
        )

    if args is not None:
        log.info("Saving figure to %s...", args.out_path)
        plt.savefig(
            args.out_path,
            dpi=args.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

    plt.show()


def load_and_process_data() -> pd.DataFrame:
    """Load and process the results data for 1km power-transformed models."""
    # Only keep traits listed in params
    keep_traits = [f"X{t}" for t in CFG.datasets.Y.traits]

    try:
        results_df = (
            pd.read_parquet("results/all_results.parquet")
            .assign(base_trait_id=lambda df: df.trait_id.str.split("_").str[0])
            .query("base_trait_id in @keep_traits")
            .query("resolution == '1km'")
            .query("transform == 'power'")
            .pipe(add_human_readables)
            .drop(columns=["base_trait_id"])
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "results/all_results.parquet not found. Make sure you're running from the project root."
        )

    # Check if we have the required columns
    required_columns = ["trait_name", "trait_set_abbr", "pearsonr_mean", "pearsonr_std"]
    missing_columns = [col for col in required_columns if col not in results_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Select only the columns we need and remove rows with missing data
    results_df = results_df[required_columns].dropna()

    if results_df.empty:
        raise ValueError("No data found for 1km resolution power-transformed models")

    log.info(
        "Loaded data for %d traits across %d trait sets",
        results_df.trait_name.nunique(),
        results_df.trait_set_abbr.nunique(),
    )
    log.info("Trait sets found: %s", list(results_df.trait_set_abbr.unique()))

    return results_df


def calculate_error_bars(
    mean: float, std: float, error_type: str, n_folds: int = 5
) -> float:
    """Calculate error bar values based on the specified error type.

    Args:
        mean: Mean value
        std: Standard deviation
        error_type: Type of error bars ('std' or 'ci95')
        n_folds: Number of cross-validation folds (default: 5)

    Returns:
        Error bar value
    """
    if error_type == "std":
        return std
    elif error_type == "ci95":
        # Calculate 95% confidence interval using t-distribution
        # For n=5 folds, df=4, t-critical value ≈ 2.776
        degrees_of_freedom = n_folds - 1
        t_critical = stats.t.ppf(
            0.975, degrees_of_freedom
        )  # 97.5th percentile for 95% CI
        standard_error = std / np.sqrt(n_folds)
        return t_critical * standard_error
    else:
        raise ValueError(f"Unknown error_type: {error_type}. Must be 'std' or 'ci95'")


def create_trait_pearsonr_dotplot(
    df: pd.DataFrame, figsize: tuple[float, float] = (20, 8), error_type: str = "std"
) -> Figure:
    """Create the dot plot showing Pearson R across traits and trait sets.

    Args:
        df: DataFrame with columns trait_name, trait_set_abbr, pearsonr_mean, pearsonr_std
        figsize: Figure size (width, height) in inches
        error_type: Type of error bars ('std' for standard deviation, 'ci95' for 95% CI)

    Returns:
        Figure: The matplotlib figure object
    """
    # Sort traits by average Pearson R for better visual presentation
    trait_order = (
        df.groupby("trait_name")["pearsonr_mean"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create trait positions on x-axis
    trait_positions = np.arange(len(trait_order))

    # Create offset positions for each trait set within each trait
    n_trait_sets = len(TRAIT_SET_ORDER)
    offset_width = 0.25
    offsets = np.linspace(-offset_width, offset_width, n_trait_sets)

    # Plot dots with error bars for each trait set
    for i, trait_set in enumerate(TRAIT_SET_ORDER):
        trait_set_data = df[df.trait_set_abbr == trait_set]

        # Get data for each trait in the correct order
        x_positions = []
        y_values = []
        y_errors = []

        for j, trait in enumerate(trait_order):
            trait_data = trait_set_data[trait_set_data.trait_name == trait]
            if not trait_data.empty:
                mean_val = trait_data.pearsonr_mean.iloc[0]
                std_val = trait_data.pearsonr_std.iloc[0]
                error_val = calculate_error_bars(mean_val, std_val, error_type)

                x_positions.append(trait_positions[j] + offsets[i])
                y_values.append(mean_val)
                y_errors.append(error_val)

        # Plot the dots with error bars
        ax.errorbar(
            x_positions,
            y_values,
            yerr=y_errors,
            fmt="o",
            color=tricolor_palette[i],
            label=trait_set,
            markersize=8,
            capsize=3,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.8,
        )

    # Customize the plot
    error_label = "SD" if error_type == "std" else "95% CI"
    ax.set_xlabel("Trait", fontweight="bold", fontsize=14)
    ax.set_ylabel(
        f"Pearson's $r$ (fold-wise mean ± {error_label})",
        fontweight="bold",
        fontsize=14,
    )
    ax.set_title(
        "Fold-wise Pearson Correlation by Trait and Trait Set\n"
        "(1 km resolution, power transformed)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Set x-axis ticks and labels with line breaks for long names
    ax.set_xticks(trait_positions)
    formatted_trait_names = [split_trait_name(trait) for trait in trait_order]
    ax.set_xticklabels(formatted_trait_names, rotation=45, ha="right", fontsize=12)

    # Add legend
    ax.legend(
        title="Trait Set",
        title_fontsize=13,
        fontsize=12,
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Customize grid and spines
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    sns.despine()

    # Set y-axis limits to start from 0 and add some padding at the top
    y_max = df.pearsonr_mean.max() + df.pearsonr_std.max()
    ax.set_ylim(0, y_max * 1.1)

    # Adjust layout for landscape orientation
    plt.tight_layout()

    return fig


def split_trait_name(trait_name: str, max_length: int = 12) -> str:
    """Split a trait name to two lines if it is too long."""
    if len(trait_name) > max_length:
        # If >= 3 words, split on the second space, otherwise split on the first space
        words = trait_name.split()
        if len(words) >= 3:
            split_idx = trait_name.find(" ", trait_name.find(" ") + 1)
        else:
            split_idx = trait_name.find(" ")

        if split_idx > 0:
            trait_name = trait_name[:split_idx] + "\n" + trait_name[split_idx + 1 :]

    return trait_name


if __name__ == "__main__":
    main(cli())
