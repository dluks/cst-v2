from pathlib import Path

import pandas as pd

from src.conf.environment import log

__all__ = ["match_pfts"]


def match_pfts(
    df: pd.DataFrame, pfts_fp: Path, harmonization_fp: Path, threshold: float = 0.70
) -> pd.DataFrame:
    """Match with growth forms (coarse PFTs) using species, then genus fallback.

    Strategy:
    1) Left-merge on species to retain all hydraulic rows and capture species-level
       PFTs where available.
    2) For rows still missing PFT, derive the genus (first token of the species)
       and fill using the dominant PFT for that genus ONLY if it comprises at
       least 70% of records for that genus in the PFTs table.

    Returns the input DataFrame with a categorical ``pft`` column added. Species
    without a species- or genus-level match remain with missing PFT.
    """
    # Ensure dtype on hydraulic species column
    df = df.astype({"speciesname": "string[pyarrow]"}).copy()

    # Load initial species harmonization dataframe
    species_harmonization_cols = {
        "hydNameIn": "string[pyarrow]",  # Hydraulic trait name
        "groNameIn": "string[pyarrow]",  # Growth form name
        "nameOutWFO": "string[pyarrow]",  # WFO name
        "GBIFKeyGBIF": pd.Int32Dtype(),  # GBIF key
        "nameOutWCVP": "string[pyarrow]",  # WCVP name
    }
    species_harmonization_df = (
        pd.read_csv(
            harmonization_fp,
            compression="gzip",
            usecols=species_harmonization_cols.keys(),  # pyright: ignore[reportArgumentType]
            dtype=species_harmonization_cols,  # pyright: ignore[reportArgumentType]
        )
        .assign(
            hydNameIn=lambda d: d["hydNameIn"].str.lower(),
            groNameIn=lambda d: d["groNameIn"].str.lower(),
            nameOutWFO=lambda d: d["nameOutWFO"].str.lower(),
            nameOutWCVP=lambda d: d["nameOutWCVP"].str.lower(),
        )
        .drop_duplicates()
    )

    # Add GBIF keys and WCVP + WFO names for downstream matching with GBIF and sPlot
    df = df.merge(
        species_harmonization_df[
            ["hydNameIn", "GBIFKeyGBIF", "nameOutWCVP", "nameOutWFO"]
        ],
        left_on="speciesname",
        right_on="hydNameIn",
        how="left",
    ).drop(columns=["hydNameIn"])

    # Load and normalize PFTs table
    # After normalizing, extract TRY growth form species that have matches with WFO (and
    # therefore also hydraulic traits as they were already harmonized with WFO)
    pfts = (
        pd.read_parquet(pfts_fp, columns=["AccSpeciesName", "pft"])
        .rename(columns={"AccSpeciesName": "speciesname"})
        .astype({"speciesname": "string[pyarrow]", "pft": "category"})
        .assign(speciesname=lambda d: d["speciesname"].str.lower())
        .drop_duplicates(subset=["speciesname"])
        .merge(
            (
                species_harmonization_df[["groNameIn", "nameOutWFO"]]
                # Filter out TRY growth form species that have no WFO match
                # These had no valid
                .query("groNameIn.notna()")
                .drop_duplicates()
                .dropna()
            ),
            left_on="speciesname",
            right_on="groNameIn",
            how="left",
        )
        .drop(columns=["groNameIn"])
    )

    # 1) Species-level left merge
    matched = df.merge(
        pfts[["nameOutWFO", "pft"]]
        # There are quite a few TRY species that match to the same WFO species, so we
        # need to deduplicate. This may drop a very small amount of duplicate WFO
        # records that contain different PFT assignments.
        .dropna()
        .drop_duplicates(subset=["nameOutWFO"]),
        left_on="nameOutWFO",
        right_on="nameOutWFO",
        how="left",
    )
    matched = matched.rename(columns={"pft": "pft_species"})

    n_total = matched.shape[0]
    n_species = int(matched["pft_species"].notna().sum())
    log.info(
        "Matched PFTs at species level: %d/%d (%.1f%%)",
        n_species,
        n_total,
        100.0 * n_species / max(1, n_total),
    )

    # 2) Genus-level fallback for remaining rows (with confidence threshold)
    pfts = pfts.assign(genus=lambda d: d["speciesname"].str.split().str[0])

    # Compute dominant PFT per genus and its proportion; require >= 0.70
    genus_pft_counts = (
        pfts.dropna(subset=["genus", "pft"])  # keep valid genus and pft rows
        .groupby(["genus", "pft"])  # count per (genus, pft)
        .size()
        .reset_index(name="n")
    )
    genus_totals = (
        genus_pft_counts.groupby("genus")["n"].sum().to_frame("n_total").reset_index()
    )
    genus_stats = genus_pft_counts.merge(genus_totals, on="genus", how="left")
    genus_stats = genus_stats.assign(prop=lambda d: d["n"] / d["n_total"])  # float

    # Pick dominant pft per genus, then filter by threshold
    dominant = genus_stats.sort_values(
        ["genus", "n"], ascending=[True, False]
    ).drop_duplicates(subset=["genus"], keep="first")
    genus_mode = (
        dominant.loc[dominant["prop"] >= threshold, ["genus", "pft"]]
        .rename(columns={"pft": "pft_genus"})
        .reset_index(drop=True)
    )

    # Derive genus in hydraulic table and merge genus-level PFT
    matched = matched.assign(genus=lambda d: d["speciesname"].str.split().str[0])
    matched = matched.merge(genus_mode[["genus", "pft_genus"]], on="genus", how="left")

    # Consolidate: prefer species-level, else genus-level
    # Use strings for assignment then recast to category
    matched["pft"] = matched["pft_species"].astype("string[pyarrow]")
    missing_mask = matched["pft"].isna()
    matched.loc[missing_mask, "pft"] = matched.loc[missing_mask, "pft_genus"].astype(
        "string[pyarrow]"
    )
    n_final = int(matched["pft"].notna().sum())
    n_genus_added = n_final - n_species
    log.info(
        "Added PFTs via genus fallback (>=%d%% confidence): %d "
        "(now %d/%d, %.1f%% total)",
        threshold * 100,
        n_genus_added,
        n_final,
        n_total,
        100.0 * n_final / max(1, n_total),
    )

    # Cleanup helper columns and finalize dtype
    matched = matched.drop(
        columns=["pft_species", "pft_genus", "genus"], errors="ignore"
    )
    matched["pft"] = matched["pft"].astype("category")

    # Drop rows with missing PFT
    matched = matched.dropna(subset=["pft"])

    return matched
