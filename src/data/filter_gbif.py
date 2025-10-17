"""Filter GBIF observations by trait data and calculate resurvey weights."""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from distributed import Client

from src.conf.conf import get_config
from src.conf.environment import detect_system, log


def cli() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter GBIF data by traits and calculate resurvey weights"
    )
    parser.add_argument("--params", type=str, required=False, default=None)
    parser.add_argument("--country", type=str, required=False, default=None)
    return parser.parse_args()


def _create_density_plots(
    before_df: pd.DataFrame, after_df: pd.DataFrame, output_fp: Path
) -> Path:
    """Create side-by-side hexbin density plots of observations.

    Parameters
    ----------
    before_df : pd.DataFrame
        DataFrame with coordinates before filtering
        (must have decimallatitude, decimallongitude)
    after_df : pd.DataFrame
        DataFrame with coordinates after filtering
        (must have decimallatitude, decimallongitude)
    output_fp : Path
        Path to save the plot

    Returns
    -------
    Path
        Path to the saved figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Determine common extent for both plots
    lat_min = min(before_df["decimallatitude"].min(), after_df["decimallatitude"].min())
    lat_max = max(before_df["decimallatitude"].max(), after_df["decimallatitude"].max())
    lon_min = min(
        before_df["decimallongitude"].min(), after_df["decimallongitude"].min()
    )
    lon_max = max(
        before_df["decimallongitude"].max(), after_df["decimallongitude"].max()
    )

    # Before filtering
    hb1 = ax1.hexbin(
        before_df["decimallongitude"],
        before_df["decimallatitude"],
        gridsize=50,
        cmap="YlOrRd",
        mincnt=1,
        bins="log",
    )
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title(f"Before Filtering\n(n = {len(before_df):,})")
    ax1.set_xlim(lon_min, lon_max)
    ax1.set_ylim(lat_min, lat_max)
    plt.colorbar(hb1, ax=ax1, label="Count (log scale)")

    # After filtering
    hb2 = ax2.hexbin(
        after_df["decimallongitude"],
        after_df["decimallatitude"],
        gridsize=50,
        cmap="YlOrRd",
        mincnt=1,
        bins="log",
    )
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_title(f"After Filtering\n(n = {len(after_df):,})")
    ax2.set_xlim(lon_min, lon_max)
    ax2.set_ylim(lat_min, lat_max)
    plt.colorbar(hb2, ax=ax2, label="Count (log scale)")

    plt.tight_layout()

    # Save figure
    plt.savefig(output_fp, dpi=150, bbox_inches="tight")
    plt.close()

    log.info(f"Saved density plots to {output_fp}")
    return output_fp


def _save_report(stats: dict[str, Any], output_fp: Path) -> None:
    """Save a formatted pipeline stage report with filtering statistics."""
    duration = stats["end_time"] - stats["start_time"]

    report_content = f"""# GBIF Data Filtering Report

Generated: {stats["end_time"].strftime("%Y-%m-%d %H:%M:%S")}  
Duration: {duration.total_seconds():.1f} seconds  
Data Source: GBIF Occurrence Data

---

## Input Data

| Metric | Count |
|--------|------:|
| GBIF observations | {stats["input_gbif_records"]:,} |
| GBIF species | {stats["input_gbif_species"]:,} |
| Trait species (available) | {stats["input_trait_species"]:,} |
| Plant functional types | {stats["input_pfts"]:,} |

---

## Trait Matching Results

### Observations
- **Input records:** {stats["input_gbif_records"]:,}
- **Matched records:** {stats["matched_records"]:,} \
({stats["matched_records_pct"]:.2f}%)
- **Dropped records:** \
{stats["input_gbif_records"] - stats["matched_records"]:,}

### Species Coverage
- **GBIF species:** {stats["input_gbif_species"]:,}
- **Matched species:** {stats["matched_species"]:,} \
({stats["matched_species_pct"]:.2f}% of trait database)
- **Species without traits:** {stats["input_gbif_species"] - stats["matched_species"]:,}

---

## Resurvey Weighting

| Metric | Count |
|--------|------:|
| Total records | {stats["output_records"]:,} |
| Location groups | {stats["resurvey_groups"]:,} |
| Locations with multiple years | {stats["locations_with_resurveys"]:,} |
| Max years per location | {stats["max_years_per_location"]:,} |
| Average weight | {stats["avg_weight"]:.3f} |

**Weighting scheme:** Observations are grouped by publishing organization and \
location (lat/lon). Within each group, the number of unique years is counted, \
and each observation receives weight = 1/n_years. This ensures resurveyed \
locations contribute equally regardless of sampling frequency.

---

## Spatial Distribution

### Observation Density Before and After Filtering

![Density Comparison](gbif_filtering_density.png)

---

## Output Data

| Metric | Count |
|--------|------:|
| Total records | {stats["output_records"]:,} |
| Unique species | {stats["output_species"]:,} |

**Output file:** `{stats["output_file"]}`

**Columns:** specieskey, decimallatitude, decimallongitude, pft, weight

---

## Summary Statistics

- **Data retention:** \
{100 * stats["output_records"] / stats["input_gbif_records"]:.1f}% \
of original GBIF observations retained
- **Species coverage:** {stats["matched_species_pct"]:.1f}% \
of available trait species represented
- **Locations with resurveys:** \
{100 * stats["locations_with_resurveys"] / stats["resurvey_groups"]:.1f}% \
of location groups have multiple years
- **Average observations per species:** \
{stats["output_records"] / stats["output_species"]:.1f}
- **Average weight per observation:** {stats["avg_weight"]:.3f}

---

*Report generated by filter_gbif.py*
"""

    with open(output_fp, "w") as f:
        f.write(report_content)

    log.info(f"Saved filtering report to {output_fp}")


def main(args: argparse.Namespace | None = None) -> None:
    """Filter GBIF observations by trait availability and calculate resurvey weights.

    This function:
    1. Loads the traits DataFrame with GBIF keys and PFTs
    2. Loads GBIF occurrence data with efficient filtering (Portugal only)
    3. Matches GBIF observations to species with trait data
    4. Calculates resurvey weights based on unique years per location
    5. Outputs filtered DataFrame with observations, PFTs, and weights

    Resurvey weights are calculated by grouping observations by publishing organization
    and location, counting unique years, and assigning weight = 1/n_years to each
    observation in the group. This ensures resurveyed locations contribute equally
    regardless of sampling frequency.
    """
    if args is None:
        args = cli()

    cfg = get_config(params_path=args.params)
    syscfg = cfg[detect_system()][cfg.model_res].get(
        "filter_gbif", cfg[detect_system()][cfg.model_res].get("match_gbif_pfts", {})
    )

    # Initialize statistics dictionary
    stats: dict[str, Any] = {"start_time": datetime.now()}

    # 01. Load traits data
    log.info("Loading traits data...")
    traits_fp = Path(cfg.traits.interim_out)
    traits_cols = {
        "GBIFKeyGBIF": pd.Int32Dtype(),
        "pft": "category",
    }
    traits_df = (
        pd.read_parquet(traits_fp, columns=list(traits_cols.keys()))
        .astype(traits_cols)
        .dropna(subset=["GBIFKeyGBIF"])
    )

    stats["input_trait_species"] = len(traits_df)
    stats["input_pfts"] = traits_df["pft"].nunique()

    log.info(
        f"Loaded {len(traits_df)} species with trait data and GBIF keys "
        f"({traits_df['pft'].nunique()} unique PFTs)"
    )

    # 02. Set up GBIF paths
    gbif_raw_fp = f"{cfg.gbif.raw_fp}/*"
    gbif_filtered_dir = Path(cfg.gbif.filtered.out_dir, cfg.trait_type)
    gbif_filtered_dir.mkdir(parents=True, exist_ok=True)

    # Define columns to load
    gbif_columns = {
        "specieskey": pd.Int32Dtype(),
        "taxonrank": "category",
        "decimallatitude": "float64",
        "decimallongitude": "float64",
        "datasetkey": "string[pyarrow]",
        "eventdate": "string[pyarrow]",
        "publishingorgkey": "string[pyarrow]",
    }

    if args.country is not None:
        gbif_columns["countrycode"] = "category"

    def _filter_country(df: dd.DataFrame, country: str | None = None) -> dd.DataFrame:
        if country is not None:
            return df.query(f"countrycode == '{country}'")
        return df

    # 03. Load and filter GBIF data with Dask
    with Client(
        dashboard_address=cfg.dask_dashboard, n_workers=syscfg.get("n_workers", 40)
    ):
        log.info(f"Loading GBIF data from {gbif_raw_fp}...")

        gbif = (
            dd.read_parquet(
                gbif_raw_fp,
                columns=list(gbif_columns.keys()),
                engine="pyarrow",
            )
            .query("taxonrank == 'SPECIES'")
            .pipe(_filter_country, args.country)
            .astype(gbif_columns)
            .drop(columns=["taxonrank"])
            .dropna(subset=["decimallatitude", "decimallongitude", "specieskey"])
        )

        if args.country is not None:
            gbif = gbif.drop(columns=["countrycode"])

        # Collect input GBIF statistics
        stats["input_gbif_records"] = len(gbif)
        stats["input_gbif_species"] = gbif.specieskey.nunique().compute()
        log.info(
            f"Loaded {stats['input_gbif_records']:,} GBIF observations "
            f"with {stats['input_gbif_species']:,} unique species"
        )

        # 04. Match with traits data
        log.info("Matching GBIF observations with trait species...")

        # Convert traits to Dask DataFrame for efficient join
        traits_dask = dd.from_pandas(traits_df, npartitions=1)

        # Inner join to keep only species with trait data
        before = len(gbif)
        gbif_matched = gbif.merge(
            traits_dask,
            left_on="specieskey",
            right_on="GBIFKeyGBIF",
            how="inner",
        ).drop(columns=["GBIFKeyGBIF"])
        after = len(gbif_matched)

        stats["matched_records"] = after
        stats["matched_records_pct"] = (after / before) * 100 if before > 0 else 0

        log.info(
            "Dropped %d records without trait data (%.2f%%)",
            before - after,
            100 - ((after / before) * 100),
        )
        n_species = gbif_matched.specieskey.nunique().compute()
        n_trait_species = traits_df.GBIFKeyGBIF.nunique()
        stats["matched_species"] = n_species
        stats["matched_species_pct"] = (
            100 * n_species / n_trait_species if n_trait_species > 0 else 0
        )
        log.info(
            "Number of species in trait data retained: %d (%.2f%%)",
            n_species,
            100 * n_species / n_trait_species,
        )

        # 05. Calculate resurvey weights
        log.info("Calculating resurvey weights...")
        resurvey_group_cols = [
            "publishingorgkey",
            "decimallatitude",
            "decimallongitude",
        ]

        # Extract year from eventdate
        gbif_matched = gbif_matched.assign(
            year=gbif_matched["eventdate"].str[:4].astype("Int16")
        )

        # Count unique years per location group
        resurvey_counts = (
            gbif_matched.groupby(resurvey_group_cols)["year"]
            .nunique()
            .reset_index()
            .rename(columns={"year": "n_years"})
        )

        log.info("Computed unique years per location group")

        # Merge counts back and calculate weights
        gbif_matched = gbif_matched.merge(
            resurvey_counts, on=resurvey_group_cols, how="left"
        )
        # Weight = 1.0 for n_years <= 1, otherwise 1.0 / n_years
        gbif_matched["weight"] = gbif_matched["n_years"].map_partitions(
            lambda s: pd.Series(np.where(s <= 1, 1.0, 1.0 / s), index=s.index),
            meta=("weight", "f8"),
        )
        gbif_matched = gbif_matched.drop(columns=["year", "n_years", "eventdate"])

        # Collect statistics (compute Dask scalars)
        stats["before_resurvey_records"] = len(gbif_matched)
        stats["resurvey_groups"] = len(resurvey_counts)
        stats["locations_with_resurveys"] = (
            (resurvey_counts["n_years"] > 1).sum().compute()
        )
        stats["max_years_per_location"] = resurvey_counts["n_years"].max().compute()
        stats["avg_weight"] = gbif_matched["weight"].mean().compute()

        log.info(
            "Calculated weights for %d location groups (%d with resurveys)",
            stats["resurvey_groups"],
            stats["locations_with_resurveys"],
        )

        # 06. Select final columns and save
        output_columns = [
            "specieskey",
            "decimallatitude",
            "decimallongitude",
            "pft",
            "weight",
        ]

        gbif_filtered = gbif_matched[output_columns]

        # 07. Sample data for density plots before saving
        log.info("Sampling data for density plots...")
        sample_size = min(100000, len(gbif))

        # Sample data before filtering (after initial load)
        gbif_sample_before = (
            gbif[["decimallatitude", "decimallongitude"]]
            .sample(frac=sample_size / len(gbif), random_state=42)
            .compute()
        )

        # Sample data after filtering
        gbif_sample_after = (
            gbif_filtered[["decimallatitude", "decimallongitude"]]
            .sample(frac=min(1.0, sample_size / len(gbif_filtered)), random_state=42)
            .compute()
        )

        log.info("Creating density plots...")
        _create_density_plots(
            gbif_sample_before,
            gbif_sample_after,
            gbif_filtered_dir / cfg.gbif.filtered.density_plot_fp,
        )

        # Save to parquet
        output_fp = gbif_filtered_dir / cfg.gbif.filtered.fp
        log.info(f"Saving filtered GBIF data to {output_fp}...")

        gbif_filtered.to_parquet(
            output_fp,
            compression="zstd",
            write_index=False,
        )

        log.info("Done! Computing final statistics...")

        # Compute and log statistics
        n_records = len(gbif_filtered)
        n_species = gbif_filtered["specieskey"].nunique().compute()

        stats["output_records"] = n_records
        stats["output_species"] = n_species
        stats["output_file"] = str(output_fp)
        stats["end_time"] = datetime.now()

        log.info(
            f"Filtered GBIF data: {n_records:,} records, "
            f"{n_species:,} unique species, "
            f"average weight: {stats['avg_weight']:.3f}"
        )

        # 08. Generate report
        log.info("Generating filtering report...")
        _save_report(stats, gbif_filtered_dir / cfg.gbif.filtered.report_fp)


if __name__ == "__main__":
    main()
    log.info("Filter GBIF processing completed successfully!")
