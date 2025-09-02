import zipfile
from pathlib import Path

import pandas as pd
from src.conf.conf import get_config
from src.conf.environment import log
from src.data.build_try_traits import standardize_trait_ids
from src.utils.trait_utils import clean_species_name, get_trait_name_from_id


def _generate_latex_table(df: pd.DataFrame, output_path: Path, caption: str) -> None:
    """Generate a LaTeX table with alternating row colors and bold headers."""

    # Create column specification with custom widths
    # First column (trait name) is 3.5cm left-aligned, others are 2.5cm centered
    n_cols = len(df.columns)
    if n_cols > 0:
        col_spec = "p{3.5cm}" + ">{\\centering\\arraybackslash}p{2.5cm}" * (n_cols - 1)
    else:
        col_spec = ""

    # Start LaTeX table
    latex_content = []
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append(f"\\caption{{{caption}}}")
    latex_content.append("\\small")  # Set font size to small
    latex_content.append("\\color{blue}")  # Set text color to blue
    latex_content.append("\\begin{tabular}{" + col_spec + "}")
    latex_content.append("\\toprule")

    # Add bold headers
    headers = " & ".join(
        [f"\\textbf{{{col.replace('_', ' ').title()}}}" for col in df.columns]
    )
    latex_content.append(headers + " \\\\")
    latex_content.append("\\midrule")

    # Add data rows with alternating colors
    for idx, (i, row) in enumerate(df.iterrows()):
        if idx % 2 == 0:
            # 0% gray (white background)
            row_color = ""
        else:
            # 20% gray background
            row_color = "\\rowcolor{gray!20}"

        # Escape special LaTeX characters in data
        row_data = []
        for value in row:
            str_value = str(value)
            # Escape common LaTeX special characters
            str_value = str_value.replace("&", "\\&")
            str_value = str_value.replace("%", "\\%")
            str_value = str_value.replace("_", "\\_")
            str_value = str_value.replace("#", "\\#")
            str_value = str_value.replace("$", "\\$")
            row_data.append(str_value)

        latex_content.append(row_color + " & ".join(row_data) + " \\\\")

    # End table
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_content))


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main function that calculates trait statistics and returns two DataFrames: final stats and PFT comparison."""
    cfg = get_config()

    # Load raw TRY data to get individual observations (not aggregated by species)
    try_params = f"try{cfg.try_version}"
    try_raw_dir = Path(cfg.raw_dir, cfg.trydb.raw[try_params].dir)

    # Get trait columns for the configured TRY version
    if cfg.try_version == 6:
        trait_cols = [f"X.X{t}." for t in cfg.datasets.Y.traits]
    else:
        trait_cols = [f"X{t}" for t in cfg.datasets.Y.traits]

    trait_cols = ["Species"] + trait_cols

    log.info(
        f"Loading raw TRY data for {len(trait_cols) - 1} traits from version {cfg.try_version}"
    )

    # Load raw TRY data (before aggregation)
    with zipfile.ZipFile(try_raw_dir / cfg.trydb.raw[try_params].zip, "r") as zip_ref:
        with zip_ref.open(cfg.trydb.raw[try_params].zipfile_csv) as zf:
            if cfg.try_version == 6:
                with zipfile.ZipFile(zf, "r") as nested_zip_ref:
                    with nested_zip_ref.open(nested_zip_ref.namelist()[0]) as csvfile:
                        traits_df = pd.read_csv(
                            csvfile,
                            encoding="latin-1",
                            usecols=trait_cols,
                        )
            else:
                traits_df = pd.read_csv(zf, encoding="latin-1", usecols=trait_cols)

    log.info(
        f"Loaded raw trait data: {traits_df.shape[0]} rows, {traits_df.shape[1]} columns"
    )

    # Clean and standardize the data
    traits_df = traits_df.astype({"Species": "string[pyarrow]"}).pipe(
        standardize_trait_ids
    )
    traits_df = clean_species_name(traits_df, "Species", "speciesname").drop(
        columns=["Species"]
    )

    log.info(f"After cleaning: {traits_df.shape[0]} rows, {traits_df.shape[1]} columns")

    # Get standardized trait column names
    traits_to_process = [f"X{t}" for t in cfg.datasets.Y.traits]

    # Load and prepare PFT data for filtering
    log.info(f"Loading PFT data for filtering by: {cfg.PFT}")
    pft_path = Path(cfg.raw_dir, cfg.trydb.raw.pfts)
    pft_columns = ["AccSpeciesName", "pft"]

    if pft_path.suffix == ".csv":
        pfts = pd.read_csv(pft_path, encoding="latin-1", usecols=pft_columns)
    elif pft_path.suffix == ".parquet":
        pfts = pd.read_parquet(pft_path, columns=pft_columns)
    else:
        raise ValueError(f"Unsupported PFT file format: {pft_path.suffix}")

    # Filter and clean PFT data
    from src.utils.trait_utils import filter_pft

    pfts_filtered = (
        pfts.pipe(filter_pft, cfg.PFT)
        .astype({"AccSpeciesName": "string[pyarrow]", "pft": "category"})
        .dropna(subset=["AccSpeciesName"])
        .pipe(clean_species_name, "AccSpeciesName", "speciesname")
        .drop(columns=["AccSpeciesName"])
        .drop_duplicates(subset=["speciesname"])
    )

    log.info(
        f"Loaded {len(pfts)} PFT records, filtered to {len(pfts_filtered)} species in PFT: {cfg.PFT}"
    )

    # Get species that match PFTs
    pft_species = set(pfts_filtered["speciesname"].unique())

    # Filter traits data to only include species with PFTs
    traits_df_filtered = traits_df[traits_df["speciesname"].isin(pft_species)]
    log.info(
        f"After PFT filtering: {len(traits_df)} → {len(traits_df_filtered)} observations"
    )

    # Initialize results lists
    results_final = []
    results_comparison = []

    # Calculate statistics for each trait column
    for trait_col in traits_to_process:
        # Get trait name and unit from mapping
        try:
            trait_name, trait_unit = get_trait_name_from_id(
                trait_col, length="short", cfg=cfg
            )
        except (ValueError, KeyError):
            # If trait not in mapping, use the column name as fallback
            trait_name = trait_col
            trait_unit = "unknown"

        # BEFORE PFT filtering
        trait_mask_before = traits_df[trait_col].notna()
        trait_data_before = traits_df[trait_mask_before]
        unique_species_before = trait_data_before["speciesname"].nunique()
        total_observations_before = len(trait_data_before)

        # AFTER PFT filtering
        trait_mask_after = traits_df_filtered[trait_col].notna()
        trait_data_after = traits_df_filtered[trait_mask_after]
        unique_species_after = trait_data_after["speciesname"].nunique()
        total_observations_after = len(trait_data_after)

        # Add to final results (current table format) with comma-formatted numbers
        results_final.append(
            {
                "trait_name": trait_name,
                "trait_unit": trait_unit,
                "number_of_species": f"{unique_species_after:,}",
                "number_of_observations": f"{total_observations_after:,}",
            }
        )

        # Calculate percentages
        species_percentage = (
            (unique_species_after / unique_species_before * 100)
            if unique_species_before > 0
            else 0
        )
        observations_percentage = (
            (total_observations_after / total_observations_before * 100)
            if total_observations_before > 0
            else 0
        )

        # Add to comparison results (new table format) with comma-formatted numbers
        results_comparison.append(
            {
                "trait_name": trait_name,
                "species (before)": f"{unique_species_before:,}",
                "species (after)": f"{unique_species_after:,} ({species_percentage:.1f}%)",
                "observations (before)": f"{total_observations_before:,}",
                "observations (after)": f"{total_observations_after:,} ({observations_percentage:.1f}%)",
            }
        )

        log.info(
            f"Trait {trait_name}: {unique_species_before}→{unique_species_after} species, "
            f"{total_observations_before}→{total_observations_after} observations"
        )

    # Create DataFrames
    trait_stats_df = pd.DataFrame(results_final)
    trait_comparison_df = pd.DataFrame(results_comparison)

    # Sort both dataframes by trait name alphabetically
    trait_stats_df = trait_stats_df.sort_values("trait_name").reset_index(drop=True)
    trait_comparison_df = trait_comparison_df.sort_values("trait_name").reset_index(
        drop=True
    )

    # Reorder columns for comparison table
    column_order = [
        "trait_name",
        "species (before)",
        "species (after)",
        "observations (before)",
        "observations (after)",
    ]
    trait_comparison_df = trait_comparison_df.reindex(columns=column_order)

    # Display the results
    log.info("Final Trait Statistics Summary (after PFT filtering):")
    log.info(f"\n{trait_stats_df.to_string(index=False)}")

    log.info("\nComparison: Before vs After PFT Filtering:")
    log.info(f"\n{trait_comparison_df.to_string(index=False)}")

    # Write both tables to CSV
    output_file_final = "results/raw_try_trait_stats.csv"
    output_file_comparison = "results/raw_try_trait_stats_pft_comparison.csv"

    trait_stats_df.to_csv(output_file_final, index=False)
    trait_comparison_df.to_csv(output_file_comparison, index=False)

    log.info(f"Final trait statistics saved to {output_file_final}")
    log.info(f"PFT comparison statistics saved to {output_file_comparison}")

    # Create results/tables directory if it doesn't exist
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Generate LaTeX tables
    _generate_latex_table(
        trait_stats_df,
        tables_dir / "raw_try_trait_stats.tex",
        "Summary of species richness and observation counts for plant functional traits from the TRY database after PFT filtering. Each trait shows the number of unique species and total observations available for analysis.",
    )

    _generate_latex_table(
        trait_comparison_df,
        tables_dir / "raw_try_trait_stats_pft_comparison.tex",
        "Impact of plant functional type (PFT) filtering on trait data availability from the TRY database. Numbers show species richness and observation counts before and after filtering, with retention percentages in parentheses.",
    )

    log.info(f"LaTeX tables saved to {tables_dir}")

    return trait_stats_df, trait_comparison_df


if __name__ == "__main__":
    main()
