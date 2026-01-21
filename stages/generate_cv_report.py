#!/usr/bin/env python
"""
Generate a cross-validation performance report for trained models.

This script creates a summary report with bar charts and performance categorization
for all traits in a given product/run. It can be run after model training completes.

Usage:
    python stages/generate_cv_report.py --params pipeline/products/<product>/params.yaml [--run-id <run_id>]
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.conf.conf import get_config
from src.conf.environment import log
from src.models.run_utils import get_latest_run_id
from src.pipeline.entrypoint_utils import setup_environment

project_root = setup_environment()


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate CV performance report for trained models."
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        required=True,
        help="Path to the params.yaml file for the product.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run ID to report on. Defaults to latest run.",
    )
    parser.add_argument(
        "--trait-set",
        type=str,
        default=None,
        help="Trait set to report on. Defaults to first in config.",
    )
    return parser.parse_args()


def parse_trait_descriptions(params_path: Path) -> dict[str, str]:
    """
    Parse trait names and their descriptions from params.yaml comments.

    Args:
        params_path: Path to params.yaml file

    Returns:
        Dictionary mapping trait ID to description
    """
    trait_descriptions = {}
    with open(params_path) as f:
        content = f.read()

    # Find lines with trait definitions like "- X4 # description"
    pattern = r"^\s*-\s*(X\d+)\s*#\s*(.+)$"
    for match in re.finditer(pattern, content, re.MULTILINE):
        trait_id = match.group(1)
        description = match.group(2).strip()
        trait_descriptions[trait_id] = description

    return trait_descriptions


def load_evaluation_results(
    models_dir: Path,
    trait_name: str,
    arch: str,
    run_id: str,
    trait_set: str,
) -> pd.DataFrame | None:
    """
    Load evaluation results CSV for a specific trait.

    Args:
        models_dir: Base models directory
        trait_name: Name of the trait (e.g., "X50")
        arch: Model architecture (e.g., "autogluon")
        run_id: Run ID
        trait_set: Trait set (e.g., "splot_gbif")

    Returns:
        DataFrame with evaluation results or None if not found
    """
    eval_path = models_dir / trait_name / arch / run_id / trait_set / "evaluation_results.csv"
    if not eval_path.exists():
        log.warning("Evaluation results not found: %s", eval_path)
        return None

    df = pd.read_csv(eval_path)
    df["trait"] = trait_name
    return df


def collect_all_results(
    models_dir: Path,
    arch: str,
    run_id: str,
    trait_set: str,
) -> pd.DataFrame:
    """
    Collect evaluation results from all traits in a run.

    Args:
        models_dir: Base models directory
        arch: Model architecture
        run_id: Run ID
        trait_set: Trait set

    Returns:
        Combined DataFrame with results from all traits
    """
    all_results = []

    # Find all trait directories (any subdirectory is considered a trait)
    trait_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir()])

    for trait_dir in trait_dirs:
        trait_name = trait_dir.name
        df = load_evaluation_results(models_dir, trait_name, arch, run_id, trait_set)
        if df is not None:
            all_results.append(df)

    if not all_results:
        raise ValueError(f"No evaluation results found in {models_dir} for run {run_id}")

    return pd.concat(all_results, ignore_index=True)


def create_bar_chart(
    df: pd.DataFrame,
    metric: str,
    scale: str,
    trait_descriptions: dict[str, str],
    output_path: Path,
    title: str | None = None,
) -> None:
    """
    Create a bar chart for a given metric.

    Args:
        df: DataFrame with evaluation results (filtered to single scale)
        metric: Metric to plot (e.g., "pearsonr_mean")
        scale: Scale name for labeling
        trait_descriptions: Mapping of trait ID to description
        output_path: Path to save the chart
        title: Optional custom title
    """
    # Sort by metric value descending
    df_sorted = df.sort_values(metric, ascending=False).copy()

    # Create labels with trait ID and short description
    labels = []
    for trait in df_sorted["trait"]:
        desc = trait_descriptions.get(trait, "")
        # Truncate long descriptions
        if len(desc) > 30:
            desc = desc[:27] + "..."
        labels.append(f"{trait}\n{desc}" if desc else trait)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Get mean and std columns
    mean_col = metric
    std_col = metric.replace("_mean", "_std") if "_mean" in metric else None

    values = df_sorted[mean_col].values
    errors = df_sorted[std_col].values if std_col and std_col in df_sorted.columns else None

    # Color bars based on performance
    n_traits = len(values)
    colors = []
    for i in range(n_traits):
        if i < n_traits * 0.3:
            colors.append("#2ecc71")  # Green for best
        elif i < n_traits * 0.7:
            colors.append("#f39c12")  # Orange for middling
        else:
            colors.append("#e74c3c")  # Red for worst

    bars = ax.bar(range(n_traits), values, yerr=errors, capsize=3, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(n_traits))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    # Format metric name for display
    metric_display = metric.replace("_mean", "").replace("_", " ").title()
    scale_display = "Original" if scale == "original" else "Transformed"

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title(f"{metric_display} by Trait ({scale_display} Scale)", fontsize=14, fontweight="bold")

    ax.set_xlabel("Trait", fontsize=12)
    ax.set_ylabel(metric_display, fontsize=12)

    # Add horizontal line at 0 for reference
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="Best (~30%)"),
        Patch(facecolor="#f39c12", edgecolor="black", label="Middling (~40%)"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Worst (~30%)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    # Also save PNG for markdown embedding
    png_path = output_path.with_suffix(".png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    log.info("Saved chart: %s (and %s)", output_path, png_path)


def categorize_traits(
    df: pd.DataFrame,
    metric: str = "pearsonr_mean",
    trait_descriptions: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """
    Categorize traits into performance tiers.

    Args:
        df: DataFrame with evaluation results (single scale)
        metric: Metric to use for categorization
        trait_descriptions: Optional trait descriptions

    Returns:
        Dictionary with "best", "middling", and "worst" trait lists
    """
    df_sorted = df.sort_values(metric, ascending=False)
    traits = df_sorted["trait"].tolist()
    n = len(traits)

    best_cutoff = int(n * 0.3)
    worst_cutoff = int(n * 0.7)

    def format_trait(t):
        desc = trait_descriptions.get(t, "") if trait_descriptions else ""
        return f"{t} ({desc})" if desc else t

    return {
        "best": [format_trait(t) for t in traits[:best_cutoff]],
        "middling": [format_trait(t) for t in traits[best_cutoff:worst_cutoff]],
        "worst": [format_trait(t) for t in traits[worst_cutoff:]],
    }


def generate_markdown_report(
    df_all: pd.DataFrame,
    trait_descriptions: dict[str, str],
    output_dir: Path,
    product_code: str,
    run_id: str,
    trait_set: str,
) -> None:
    """
    Generate a markdown report summarizing model performance.

    Args:
        df_all: DataFrame with all evaluation results
        trait_descriptions: Mapping of trait ID to description
        output_dir: Directory to save the report
        product_code: Product code
        run_id: Run ID
        trait_set: Trait set name
    """
    report_path = output_dir / "cv_performance_report.md"

    # Separate by scale
    df_original = df_all[df_all["scale"] == "original"].copy()
    df_transformed = df_all[df_all["scale"] == "model"].copy()

    # Categorize based on original scale
    categories = categorize_traits(df_original, "pearsonr_mean", trait_descriptions)

    with open(report_path, "w") as f:
        f.write(f"# Cross-Validation Performance Report\n\n")
        f.write(f"**Product:** {product_code}  \n")
        f.write(f"**Run ID:** {run_id}  \n")
        f.write(f"**Trait Set:** {trait_set}  \n")
        f.write(f"**Number of Traits:** {len(df_original)}  \n\n")

        f.write("---\n\n")

        # Performance summary
        f.write("## Performance Summary\n\n")
        f.write("Traits are categorized based on Pearson's r (original scale):\n\n")

        f.write("### Best Performing Traits (~30%)\n\n")
        for trait in categories["best"]:
            f.write(f"- {trait}\n")
        f.write("\n")

        f.write("### Middling Performance (~40%)\n\n")
        for trait in categories["middling"]:
            f.write(f"- {trait}\n")
        f.write("\n")

        f.write("### Worst Performing Traits (~30%)\n\n")
        for trait in categories["worst"]:
            f.write(f"- {trait}\n")
        f.write("\n")

        f.write("---\n\n")

        # Detailed metrics tables
        f.write("## Detailed Metrics\n\n")

        # Original Scale - Pooled Statistics
        f.write("### Original Scale - Pooled Statistics\n\n")
        f.write("*Statistics computed from all CV predictions pooled together.*\n\n")
        f.write("| Trait | R² | Pearson r | RMSE | nRMSE |\n")
        f.write("|-------|-----|-----------|------|-------|\n")
        df_orig_sorted = df_original.sort_values("pearsonr", ascending=False)
        for _, row in df_orig_sorted.iterrows():
            r2 = f"{row['r2']:.3f}"
            pearsonr = f"{row['pearsonr']:.3f}"
            rmse = f"{row['root_mean_squared_error']:.3f}"
            nrmse = f"{row['norm_root_mean_squared_error']:.3f}"
            f.write(f"| {row['trait']} | {r2} | {pearsonr} | {rmse} | {nrmse} |\n")
        f.write("\n")

        # Original Scale - Fold-wise Statistics
        f.write("### Original Scale - Fold-wise Statistics\n\n")
        f.write("*Mean ± standard deviation across CV folds.*\n\n")
        f.write("| Trait | R² (mean ± std) | Pearson r (mean ± std) | RMSE (mean ± std) |\n")
        f.write("|-------|-----------------|------------------------|-------------------|\n")
        df_orig_sorted = df_original.sort_values("pearsonr_mean", ascending=False)
        for _, row in df_orig_sorted.iterrows():
            r2 = f"{row['r2_mean']:.3f} ± {row['r2_std']:.3f}"
            pearsonr = f"{row['pearsonr_mean']:.3f} ± {row['pearsonr_std']:.3f}"
            rmse = f"{row['root_mean_squared_error_mean']:.3f} ± {row['root_mean_squared_error_std']:.3f}"
            f.write(f"| {row['trait']} | {r2} | {pearsonr} | {rmse} |\n")
        f.write("\n")

        if not df_transformed.empty:
            # Transformed Scale - Pooled Statistics
            f.write("### Transformed Scale - Pooled Statistics\n\n")
            f.write("*Statistics computed from all CV predictions pooled together (in transformed space).*\n\n")
            f.write("| Trait | R² | Pearson r | RMSE | nRMSE |\n")
            f.write("|-------|-----|-----------|------|-------|\n")
            df_trans_sorted = df_transformed.sort_values("pearsonr", ascending=False)
            for _, row in df_trans_sorted.iterrows():
                r2 = f"{row['r2']:.3f}"
                pearsonr = f"{row['pearsonr']:.3f}"
                rmse = f"{row['root_mean_squared_error']:.3f}"
                nrmse = f"{row['norm_root_mean_squared_error']:.3f}"
                f.write(f"| {row['trait']} | {r2} | {pearsonr} | {rmse} | {nrmse} |\n")
            f.write("\n")

            # Transformed Scale - Fold-wise Statistics
            f.write("### Transformed Scale - Fold-wise Statistics\n\n")
            f.write("*Mean ± standard deviation across CV folds (in transformed space).*\n\n")
            f.write("| Trait | R² (mean ± std) | Pearson r (mean ± std) | RMSE (mean ± std) |\n")
            f.write("|-------|-----------------|------------------------|-------------------|\n")
            df_trans_sorted = df_transformed.sort_values("pearsonr_mean", ascending=False)
            for _, row in df_trans_sorted.iterrows():
                r2 = f"{row['r2_mean']:.3f} ± {row['r2_std']:.3f}"
                pearsonr = f"{row['pearsonr_mean']:.3f} ± {row['pearsonr_std']:.3f}"
                rmse = f"{row['root_mean_squared_error_mean']:.3f} ± {row['root_mean_squared_error_std']:.3f}"
                f.write(f"| {row['trait']} | {r2} | {pearsonr} | {rmse} |\n")
            f.write("\n")

        f.write("---\n\n")

        # Charts - embedded images
        f.write("## Charts\n\n")

        f.write("### Original Scale - Pooled\n\n")
        f.write("*Statistics computed from all CV predictions pooled together.*\n\n")
        f.write("#### R² by Trait\n\n")
        f.write("![R² by Trait (Original Scale - Pooled)](figures/r2_by_trait_original_pooled.png)\n\n")
        f.write("#### Pearson r by Trait\n\n")
        f.write("![Pearson r by Trait (Original Scale - Pooled)](figures/pearsonr_by_trait_original_pooled.png)\n\n")

        f.write("### Original Scale - Fold-wise\n\n")
        f.write("*Mean ± standard deviation across CV folds.*\n\n")
        f.write("#### R² by Trait\n\n")
        f.write("![R² by Trait (Original Scale - Fold-wise)](figures/r2_by_trait_original_foldwise.png)\n\n")
        f.write("#### Pearson r by Trait\n\n")
        f.write("![Pearson r by Trait (Original Scale - Fold-wise)](figures/pearsonr_by_trait_original_foldwise.png)\n\n")

        if not df_transformed.empty:
            f.write("### Transformed Scale - Pooled\n\n")
            f.write("*Statistics computed from all CV predictions pooled together (in transformed space).*\n\n")
            f.write("#### R² by Trait\n\n")
            f.write("![R² by Trait (Transformed Scale - Pooled)](figures/r2_by_trait_transformed_pooled.png)\n\n")
            f.write("#### Pearson r by Trait\n\n")
            f.write("![Pearson r by Trait (Transformed Scale - Pooled)](figures/pearsonr_by_trait_transformed_pooled.png)\n\n")

            f.write("### Transformed Scale - Fold-wise\n\n")
            f.write("*Mean ± standard deviation across CV folds (in transformed space).*\n\n")
            f.write("#### R² by Trait\n\n")
            f.write("![R² by Trait (Transformed Scale - Fold-wise)](figures/r2_by_trait_transformed_foldwise.png)\n\n")
            f.write("#### Pearson r by Trait\n\n")
            f.write("![Pearson r by Trait (Transformed Scale - Fold-wise)](figures/pearsonr_by_trait_transformed_foldwise.png)\n\n")

        # PDF links for download
        f.write("---\n\n")
        f.write("## Download Charts (PDF)\n\n")
        f.write("- [R² Original Pooled](figures/r2_by_trait_original_pooled.pdf) | ")
        f.write("[Pearson r Original Pooled](figures/pearsonr_by_trait_original_pooled.pdf)\n")
        f.write("- [R² Original Fold-wise](figures/r2_by_trait_original_foldwise.pdf) | ")
        f.write("[Pearson r Original Fold-wise](figures/pearsonr_by_trait_original_foldwise.pdf)\n")
        if not df_transformed.empty:
            f.write("- [R² Transformed Pooled](figures/r2_by_trait_transformed_pooled.pdf) | ")
            f.write("[Pearson r Transformed Pooled](figures/pearsonr_by_trait_transformed_pooled.pdf)\n")
            f.write("- [R² Transformed Fold-wise](figures/r2_by_trait_transformed_foldwise.pdf) | ")
            f.write("[Pearson r Transformed Fold-wise](figures/pearsonr_by_trait_transformed_foldwise.pdf)\n")

    log.info("Saved report: %s", report_path)


def main():
    """Main entry point."""
    args = cli()

    # Load configuration
    cfg = get_config(args.params)

    # Get product code and models directory
    product_code = cfg.product_code
    models_dir = Path(project_root) / cfg.models.dir_fp
    arch = cfg.train.arch

    # Determine trait set
    trait_set = args.trait_set or cfg.train.trait_sets[0]

    # Determine run ID
    if args.run_id:
        run_id = args.run_id
    else:
        # Find latest run from any trait (any subdirectory is considered a trait)
        trait_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        if not trait_dirs:
            raise ValueError(f"No trait directories found in {models_dir}")

        base_dir = trait_dirs[0] / arch
        run_id = get_latest_run_id(base_dir)
        if run_id is None:
            raise ValueError(f"No runs found in {base_dir}")

    log.info("Generating report for product=%s, run=%s, trait_set=%s", product_code, run_id, trait_set)

    # Create output directories
    output_dir = Path(project_root) / "results" / product_code / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Parse trait descriptions from params file
    trait_descriptions = parse_trait_descriptions(Path(args.params))

    # Collect all results
    df_all = collect_all_results(models_dir, arch, run_id, trait_set)

    # Save combined CSV
    csv_path = output_dir / "cv_metrics_summary.csv"
    df_all.to_csv(csv_path, index=False)
    log.info("Saved metrics summary: %s", csv_path)

    # Separate by scale
    df_original = df_all[df_all["scale"] == "original"].copy()
    df_transformed = df_all[df_all["scale"] == "model"].copy()

    # Generate bar charts - Original scale (Pooled)
    create_bar_chart(
        df_original,
        "r2",
        "original",
        trait_descriptions,
        figures_dir / "r2_by_trait_original_pooled.pdf",
        title="R² by Trait (Original Scale - Pooled)",
    )
    create_bar_chart(
        df_original,
        "pearsonr",
        "original",
        trait_descriptions,
        figures_dir / "pearsonr_by_trait_original_pooled.pdf",
        title="Pearson r by Trait (Original Scale - Pooled)",
    )

    # Generate bar charts - Original scale (Fold-wise with error bars)
    create_bar_chart(
        df_original,
        "r2_mean",
        "original",
        trait_descriptions,
        figures_dir / "r2_by_trait_original_foldwise.pdf",
        title="R² by Trait (Original Scale - Fold-wise Mean ± Std)",
    )
    create_bar_chart(
        df_original,
        "pearsonr_mean",
        "original",
        trait_descriptions,
        figures_dir / "pearsonr_by_trait_original_foldwise.pdf",
        title="Pearson r by Trait (Original Scale - Fold-wise Mean ± Std)",
    )

    # Generate bar charts - Transformed scale (if available)
    if not df_transformed.empty:
        # Pooled
        create_bar_chart(
            df_transformed,
            "r2",
            "model",
            trait_descriptions,
            figures_dir / "r2_by_trait_transformed_pooled.pdf",
            title="R² by Trait (Transformed Scale - Pooled)",
        )
        create_bar_chart(
            df_transformed,
            "pearsonr",
            "model",
            trait_descriptions,
            figures_dir / "pearsonr_by_trait_transformed_pooled.pdf",
            title="Pearson r by Trait (Transformed Scale - Pooled)",
        )

        # Fold-wise
        create_bar_chart(
            df_transformed,
            "r2_mean",
            "model",
            trait_descriptions,
            figures_dir / "r2_by_trait_transformed_foldwise.pdf",
            title="R² by Trait (Transformed Scale - Fold-wise Mean ± Std)",
        )
        create_bar_chart(
            df_transformed,
            "pearsonr_mean",
            "model",
            trait_descriptions,
            figures_dir / "pearsonr_by_trait_transformed_foldwise.pdf",
            title="Pearson r by Trait (Transformed Scale - Fold-wise Mean ± Std)",
        )

    # Generate markdown report
    generate_markdown_report(
        df_all,
        trait_descriptions,
        output_dir,
        product_code,
        run_id,
        trait_set,
    )

    log.info("Report generation complete. Output directory: %s", output_dir)


if __name__ == "__main__":
    main()
