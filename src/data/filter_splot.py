"""Filter sPlot observations to species with trait data and calculate resurvey
weights."""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.trait_utils import clean_species_name


def cli() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter sPlot data by traits and calculate resurvey weights"
    )
    parser.add_argument("--params", type=str, required=False, default=None)
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Filter sPlot observations by trait availability and calculate resurvey weights.

    This function:
    1. Extracts sPlot data from RData file
    2. Loads the traits DataFrame with nameOutWCVP and PFTs
    3. Matches sPlot observations to species with trait data
    4. Calculates weights for resurvey groups
    5. Outputs filtered DataFrame with observations, PFTs, and weights

    Resurvey groups are defined by GIVD_NU, Longitude, Latitude, and the RESURVEY
    boolean flag. Each observation in a resurvey group gets a weight of 1/n where n
    is the number of resurveys in that group.
    """
    if args is None:
        args = cli()

    cfg = get_config(params_path=args.params)

    # Initialize statistics dictionary for report
    stats: dict[str, Any] = {"start_time": datetime.now()}

    # 01. Extract sPlot data
    log.info("Extracting sPlot data...")
    splot_raw_dir = Path(cfg.raw_dir, cfg.datasets.Y.splot)
    splot_prep_dir = Path(cfg.interim_dir, cfg.splot.interim.dir)
    splot_prep_dir.mkdir(parents=True, exist_ok=True)

    if cfg.splot_open:
        header_df, vegetation_df = _extract_splot_open(splot_raw_dir)
    else:
        header_df, vegetation_df = _extract_splot_full(splot_raw_dir, splot_prep_dir)

    stats["input_plots"] = len(header_df)
    stats["input_vegetation_records"] = len(vegetation_df)

    log.info(
        f"Loaded sPlot data: {len(header_df):,} plots, "
        f"{len(vegetation_df):,} vegetation observations"
    )

    # 02. Load traits data
    log.info("Loading traits data...")
    traits_fp = Path(cfg.traits.interim_out)
    traits_cols = {
        "nameOutWCVP": "string[pyarrow]",
        "pft": "category",
    }
    traits_df = (
        pd.read_parquet(traits_fp, columns=list(traits_cols.keys()))
        .astype(traits_cols)
        .dropna(subset=["nameOutWCVP"])
        .assign(nameOutWCVP=lambda d: d["nameOutWCVP"].str.lower())
        .drop_duplicates(subset=["nameOutWCVP"])
    )

    stats["input_trait_species"] = len(traits_df)
    stats["input_pfts"] = traits_df["pft"].nunique()

    log.info(
        f"Loaded {len(traits_df):,} species with trait data "
        f"({traits_df['pft'].nunique()} unique PFTs)"
    )

    # 03. Clean species names in vegetation data
    log.info("Cleaning species names...")
    abundance_col = "Relative_cover" if cfg.splot_open else "Rel_Abund_Plot"

    vegetation_df = (
        vegetation_df[["PlotObservationID", "Species", abundance_col]]
        .dropna(subset=["Species", abundance_col])
        .pipe(clean_species_name, "Species", "speciesname")
        .drop(columns=["Species"])
        .dropna(subset=["speciesname"])
    )

    # 04. Match with traits data
    log.info("Matching vegetation observations with trait species...")
    before = len(vegetation_df)
    before_species = vegetation_df["speciesname"].nunique()

    vegetation_matched = vegetation_df.merge(
        traits_df,
        left_on="speciesname",
        right_on="nameOutWCVP",
        how="inner",
    ).drop(columns=["nameOutWCVP"])
    after = len(vegetation_matched)

    stats["matched_vegetation_records"] = after
    stats["matched_vegetation_pct"] = (after / before) * 100 if before > 0 else 0

    log.info(
        "Retained %d / %d vegetation records (%.2f%%).",
        after,
        before,
        (after / before) * 100,
    )

    n_species = vegetation_matched["speciesname"].nunique()
    n_trait_species = traits_df["nameOutWCVP"].nunique()

    stats["matched_species"] = n_species
    stats["matched_species_pct"] = (
        (100 * n_species / n_trait_species) if n_trait_species > 0 else 0
    )
    stats["input_species"] = before_species

    log.info(
        "Retained %d / %d species (%.2f%%).",
        n_species,
        n_trait_species,
        100 * n_species / n_trait_species,
    )

    total_cov = vegetation_matched[abundance_col].sum()
    stats["retained_abundance"] = total_cov
    stats["retained_abundance_pct"] = (
        (total_cov / len(header_df)) * 100 if len(header_df) > 0 else 0
    )

    log.info(
        "Retained %.2f / %d abundance (%.2f%%).",
        total_cov,
        len(header_df),  # Number of original plots
        (total_cov / len(header_df)) * 100,
    )

    # 05. Calculate resurvey weights at header level
    log.info("Calculating resurvey weights...")
    resurvey_group_cols = ["GIVD_NU", "Longitude", "Latitude"]

    # Count resurveys per location group
    resurvey_counts = (
        header_df.groupby(resurvey_group_cols)
        .agg(
            n_plots=("PlotObservationID", "count"),
            n_resurveys=("RESURVEY", "sum"),  # Count True values
        )
        .reset_index()
    )

    # Calculate weight based on number of resurveys in group
    # If there are resurveys in the group, weight = 1 / (n_resurveys + 1)
    # Otherwise weight = 1.0
    resurvey_counts["weight"] = resurvey_counts.apply(
        lambda row: 1.0 / (row["n_resurveys"] + 1) if row["n_resurveys"] > 0 else 1.0,
        axis=1,
    )

    # Merge weights back to header to get PlotObservationID -> weight mapping
    header_with_weights = header_df.merge(
        resurvey_counts[resurvey_group_cols + ["weight"]],
        on=resurvey_group_cols,
        how="left",
    )

    stats["resurvey_groups"] = (resurvey_counts["n_resurveys"] > 0).sum()
    stats["plots_with_resurveys"] = (header_with_weights["weight"] < 1.0).sum()

    log.info(
        "Calculated weights for %d resurvey groups",
        (resurvey_counts["n_resurveys"] > 0).sum(),
    )

    # 06. Merge weights and coordinates into vegetation data
    log.info("Merging weights and coordinates into vegetation data...")
    vegetation_final = vegetation_matched.merge(
        header_with_weights[["PlotObservationID", "Latitude", "Longitude", "weight"]],
        on="PlotObservationID",
        how="left",
    )

    # 07. Select final columns and save
    output_columns = [
        "PlotObservationID",
        "speciesname",
        "Latitude",
        "Longitude",
        "pft",
        "weight",
        abundance_col,
    ]

    # Rename abundance column to standard name
    vegetation_final = vegetation_final.rename(
        columns={abundance_col: "Rel_Abund_Plot"}
    )
    output_columns[-1] = "Rel_Abund_Plot"

    vegetation_filtered = vegetation_final[output_columns]

    # Save to parquet
    output_fp = splot_prep_dir / cfg.splot.interim.filtered
    log.info(f"Saving filtered sPlot data to {output_fp}...")

    vegetation_filtered.to_parquet(
        output_fp,
        compression="zstd",
        index=False,
    )

    log.info("Done! Computing final statistics...")

    # Compute and log statistics
    n_records = len(vegetation_filtered)
    n_species = vegetation_filtered["speciesname"].nunique()
    n_plots = vegetation_filtered["PlotObservationID"].nunique()

    stats["output_records"] = n_records
    stats["output_species"] = n_species
    stats["output_plots"] = n_plots
    stats["output_file"] = str(output_fp)
    stats["end_time"] = datetime.now()

    log.info(
        f"Filtered sPlot data: {n_records:,} records, "
        f"{n_species:,} unique species, {n_plots:,} unique plots"
    )

    # Save report
    _save_report(stats, output_fp.parent, cfg.splot_open)


def _save_report(stats: dict[str, Any], output_dir: Path, splot_open: bool) -> None:
    """Save a formatted pipeline stage report with filtering statistics."""
    report_path = output_dir / "filter_splot_report.md"

    duration = stats["end_time"] - stats["start_time"]

    report_content = f"""# sPlot Data Filtering Report

Generated: {stats["end_time"].strftime("%Y-%m-%d %H:%M:%S")}  
Duration: {duration.total_seconds():.1f} seconds  
Data Version: {"sPlot Open" if splot_open else "sPlot Full"}

---

## Input Data

| Metric | Count |
|--------|------:|
| sPlot plots | {stats["input_plots"]:,} |
| Vegetation observations | {stats["input_vegetation_records"]:,} |
| Input species (in vegetation) | {stats["input_species"]:,} |
| Trait species (available) | {stats["input_trait_species"]:,} |
| Plant functional types | {stats["input_pfts"]:,} |

---

## Species Matching Results

### Vegetation Records
- **Input records:** {stats["input_vegetation_records"]:,}
- **Matched records:** {stats["matched_vegetation_records"]:,} \
({stats["matched_vegetation_pct"]:.2f}%)
- **Dropped records:** \
{stats["input_vegetation_records"] - stats["matched_vegetation_records"]:,}

### Species Coverage
- **Input species:** {stats["input_species"]:,}
- **Matched species:** {stats["matched_species"]:,} \
({stats["matched_species_pct"]:.2f}% of trait database)
- **Dropped species:** \
{stats["input_species"] - stats["matched_species"]:,}

### Abundance Retention
- **Total abundance retained:** {stats["retained_abundance"]:.2f}
- **Retention rate:** {stats["retained_abundance_pct"]:.2f}%

---

## Resurvey Weighting

| Metric | Count |
|--------|------:|
| Location groups with resurveys | {stats["resurvey_groups"]:,} |
| Plots with weight < 1.0 | {stats["plots_with_resurveys"]:,} |
| Plots with weight = 1.0 | {stats["input_plots"] - stats["plots_with_resurveys"]:,} |

**Weighting scheme:** Plots in resurvey groups receive weight = 1/(n+1) \
where n is the number of resurveys at that location. Non-resurveyed plots \
receive weight = 1.0.

---

## Output Data

| Metric | Count |
|--------|------:|
| Total records | {stats["output_records"]:,} |
| Unique species | {stats["output_species"]:,} |
| Unique plots | {stats["output_plots"]:,} |

**Output file:** `{stats["output_file"]}`

**Columns:** PlotObservationID, speciesname, Latitude, Longitude, pft, \
weight, Rel_Abund_Plot

---

## Summary Statistics

- **Data reduction:** \
{(1 - stats["output_records"] / stats["input_vegetation_records"]) * 100:.1f}% \
of vegetation records removed
- **Species coverage:** \
{(stats["output_species"] / stats["input_trait_species"]) * 100:.1f}% \
of available trait species represented
- **Plot coverage:** \
{(stats["output_plots"] / stats["input_plots"]) * 100:.1f}% \
of original plots retained
- **Average records per plot:** \
{stats["output_records"] / stats["output_plots"]:.1f}
- **Average records per species:** \
{stats["output_records"] / stats["output_species"]:.1f}

---

*Report generated by filter_splot.py*
"""

    with open(report_path, "w") as f:
        f.write(report_content)

    log.info(f"Saved filtering report to {report_path}")


def _extract_splot_open(splot_raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract and convert the sPlot Open data to pandas DataFrames."""
    header_df = pd.read_csv(splot_raw_dir / "sPlotOpen_header(3).txt", sep="\t")
    vegetation_df = pd.read_csv(
        splot_raw_dir / "sPlotOpen_DT(2).txt", sep="\t", encoding="utf-8"
    )

    # Optimize DataFrames
    log.info("Optimizing DataFrames...")
    for df in [header_df, vegetation_df]:
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("string[pyarrow]")

    # Convert necessary columns to appropriate types
    if "Date" in header_df.columns:
        header_df["Date"] = pd.to_datetime(header_df["Date"], format="%Y-%m-%d")

    return header_df, vegetation_df


def _extract_splot_full(
    splot_raw_dir: Path, splot_prep_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract and convert the sPlot Full data to pandas DataFrames."""
    zip_path = splot_raw_dir / "extracted_data.zip"
    extracted_rdata_fp = splot_prep_dir / "extracted_data.RData"

    # 01. Extract sPlot data
    log.info("Unzipping %s to %s...", zip_path, splot_prep_dir)
    # Use local `unzip` as `zipfile` cannot unzip LZMA-compressed data
    subprocess.run(
        ["unzip", "-o", str(zip_path), "-d", str(splot_prep_dir)], check=False
    )

    # 02. Load sPlot data with R, coerce datetimes to strings
    log.info("Loading unzipped sPlot data...")
    r_unzip_data = f"""
    splot_prep_dir <- "{str(splot_prep_dir)}"
    splot_rdata_fp <- file.path("{str(extracted_rdata_fp)}")
    load(splot_rdata_fp)
    header$Date <- as.character(header$Date)
    """
    ro.r(r_unzip_data)

    header = ro.r["header"]
    vegetation = ro.r["vegetation"]

    # 03. Convert R dataframes to pandas DataFrames
    log.info("Converting sPlot dataframes to pandas DataFrames...")
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        header_df = ro.conversion.get_conversion().rpy2py(header)
        vegetation_df = ro.conversion.get_conversion().rpy2py(vegetation)

    # 04. Optimize DataFrames
    log.info("Optimizing DataFrames...")
    for df in [header_df, vegetation_df]:
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("string[pyarrow]")

                # Replace "NA_character_" instances with np.nan
                df[col] = df[col].replace({"NA_character_": np.nan, "NA": np.nan})

            # Data type optimizations
            if col == "PlotObservationID":
                df[col] = df[col].astype("int64")

            if col in ["Latitude", "Longitude"]:
                df[col] = df[col].astype("float64")

            if col == "Date":
                df[col] = pd.to_datetime(df[col], format="%Y-%m-%d")

            if col == "RESURVEY":
                df[col] = df[col].astype("bool")

    # 05. Clean up
    if extracted_rdata_fp.exists():
        log.info("Removing %s...", extracted_rdata_fp)
        extracted_rdata_fp.unlink()

    return header_df, vegetation_df


if __name__ == "__main__":
    main()
    log.info("Filter sPlot processing completed successfully!")
