"""Filter GBIF observations to species with trait data and calculate weights."""

import argparse
from pathlib import Path

import dask.dataframe as dd
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


def main(args: argparse.Namespace | None = None) -> None:
    """Filter GBIF observations by trait availability and calculate resurvey weights.

    This function:
    1. Loads the traits DataFrame with GBIF keys and PFTs
    2. Loads GBIF occurrence data with efficient filtering (Portugal only)
    3. Matches GBIF observations to species with trait data
    4. Calculates weights for resurvey groups (dataset, location, date)
    5. Outputs filtered DataFrame with observations, PFTs, and weights

    Resurvey groups are defined as observations with the same dataset key, location
    (rounded to 5 decimal places), and event date. Each observation in a group gets
    a weight of 1/n where n is the group size, ensuring the group sums to 1.0.
    """
    if args is None:
        args = cli()

    cfg = get_config(params_path=args.params)
    syscfg = cfg[detect_system()][cfg.model_res].get(
        "filter_gbif", cfg[detect_system()][cfg.model_res].get("match_gbif_pfts", {})
    )

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

    log.info(
        f"Loaded {len(traits_df)} species with trait data and GBIF keys "
        f"({traits_df['pft'].nunique()} unique PFTs)"
    )

    # 02. Set up GBIF paths
    gbif_raw_fp = f"{cfg.gbif.raw_fp}/*"
    gbif_prep_dir = Path(cfg.interim_dir, cfg.gbif.interim.dir)
    gbif_prep_dir.mkdir(parents=True, exist_ok=True)

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

        log.info(
            "Dropped %d records without trait data (%.2f%%)",
            before - after,
            (after / before) * 100,
        )
        n_species = gbif_matched.specieskey.nunique().compute()
        n_trait_species = traits_df.GBIFKeyGBIF.nunique()
        log.info(
            "Number of species in trait data retained: %d (%.2f%%)",
            n_species,
            100 * n_species / n_trait_species,
        )

        # 05. Calculate weights for resurvey groups
        log.info("Calculating resurvey weights...")
        resurvey_group_cols = [
            "publishingorgkey",
            "decimallatitude",
            "decimallongitude",
            "specieskey",
        ]

        resurveys = (
            gbif_matched.groupby(resurvey_group_cols)["eventdate"]
            .nunique()
            .reset_index()
            .rename(columns={"eventdate": "resurvey_count"})
        )
        log.info("Number of resurvey groups: %d", len(resurveys))

        # Merge group sizes back to get counts per observation
        gbif_matched = (
            gbif_matched.merge(resurveys, on=resurvey_group_cols, how="left")
            .assign(
                weight=lambda x: x.resurvey_count.apply(
                    lambda y: 1.0 / y if y > 0 else 1.0
                )
            )
            .drop(columns=["resurvey_count"])
        )

        log.info("Weights calculated - resurvey groups will sum to 1.0")

        # 06. Select final columns and save
        output_columns = [
            "specieskey",
            "decimallatitude",
            "decimallongitude",
            "pft",
            "weight",
        ]

        gbif_filtered = gbif_matched[output_columns]

        # Save to parquet
        output_fp = gbif_prep_dir / cfg.gbif.interim.filtered
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

        log.info(
            f"Filtered GBIF data: {n_records:,} records, {n_species:,} unique species"
        )


if __name__ == "__main__":
    main()
    log.info("Filter GBIF processing completed successfully!")
