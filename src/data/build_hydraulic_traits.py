"""Build hydraulic trait dataset from Knighton et al. species-level medians.

This script loads the Knighton et al. median hydraulic traits from an Excel file,
cleans and formats species names, applies a Yeo-Johnson power transform to the
hydraulic traits, and saves both the transformed dataset and the fitted
transformer for downstream use.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import PowerTransformer

from src.conf.conf import get_config
from src.conf.environment import log


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build hydraulic traits.")
    parser.add_argument(
        "-p", "--params", type=str, default=None, help="Path to params.yaml"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Load, clean, transform, and save Knighton hydraulic traits."""
    cfg = (
        get_config(params_path=args.params) if args.params is not None else get_config()
    )

    # Input path (Excel)
    df = _load_knighton_excel(Path(cfg.traits.raw), cfg.traits.sheet, cfg.traits.names)

    # Clean species names per notebook: drop non-binomial, lowercase
    df = _clean_species_names(df)

    # Match with growth forms (coarse PFTs)
    df = _match_with_pfts(df, cfg.traits.pfts)

    if cfg.traits.transform == "power":
        # Transform traits
        df, transformer = _power_transform(df, cfg.traits.names)
        _save_outputs(
            df=df, out_fp=Path(cfg.traits.interim_out), transformer=transformer
        )
    else:
        _save_outputs(df=df, out_fp=Path(cfg.traits.interim_out))


def _load_knighton_excel(fp: Path, sheet: str, traits: list[str]) -> pd.DataFrame:
    """Load the Knighton et al. Excel and return a minimally cleaned DataFrame.

    Returns a DataFrame with columns:
    - speciesname (original values from spec.name)
    - gsmax, P12, P50, P88, rdmax, WUE (float32)
    """

    if not fp.exists():
        raise FileNotFoundError(
            f"Hydraulic traits file not found at {fp}. Place file at this path."
        )

    log.info("Loading Knighton hydraulic traits from %s (sheet: %s)", fp, sheet)
    df = pd.read_excel(fp, sheet_name=sheet)

    required_cols = {"spec.name", *traits}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns in Knighton Excel: " + ", ".join(sorted(missing))
        )

    # Keep only the needed columns and cast dtypes
    df = (
        df[list(required_cols)]
        .rename(columns={"spec.name": "speciesname"})
        .astype(
            {
                "speciesname": "string[pyarrow]",
                **{t: "float32" for t in traits},
            }
        )
    )

    log.info("Loaded %d rows with %d hydraulic traits", df.shape[0], len(traits))
    return df


def _clean_species_names(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-binomial taxa (var./subsp./hybrids) and standardize species names.

    This follows the notebook decision to drop varieties/subspecies/cultivars rather
    than aggregating, by keeping only rows where the species string contains exactly
    two words. Then convert to lowercase.
    """
    if "speciesname" not in df.columns:
        raise ValueError("'speciesname' column not found during cleaning stage")

    # Keep only binomials (two words) and normalize case. This is done because the
    # species have already been median-aggregated, and if we dropped varieties, we risk
    # over-weighting their influence.
    before = df.shape[0]
    df = (
        df.copy()
        .query("speciesname.str.split().str.len() == 2")
        .assign(speciesname=lambda df: df["speciesname"].str.lower())
    )
    after = df.shape[0]
    log.info(
        "Filtered non-binomial taxa: kept %d of %d rows (%.2f%%)",
        after,
        before,
        100 * after / max(1, before),
    )
    return df


def _match_with_pfts(traits: pd.DataFrame, pfts_fp: Path) -> pd.DataFrame:
    """Match with growth forms (coarse PFTs) using species, then genus fallback.

    Strategy:
    1) Left-merge on species to retain all hydraulic rows and capture species-level
       PFTs where available.
    2) For rows still missing PFT, derive the genus (first token of the species)
       and fill using the most common PFT observed for that genus in the PFTs
       table.

    Returns the input DataFrame with a categorical ``pft`` column added. Species
    without a species- or genus-level match remain with missing PFT.
    """
    # Load and normalize PFTs table
    pfts = (
        pd.read_parquet(pfts_fp, columns=["AccSpeciesName", "pft"])
        .astype({"AccSpeciesName": "string[pyarrow]", "pft": "category"})
        .assign(
            speciesname=lambda d: d["AccSpeciesName"]
            .str.lower()
            .astype("string[pyarrow]")
        )
        .drop(columns=["AccSpeciesName"])
        .drop_duplicates(subset=["speciesname"])
    )

    # Ensure dtype on hydraulic species column
    traits = traits.astype({"speciesname": "string[pyarrow]"}).copy()

    # 1) Species-level left merge
    matched = traits.merge(pfts[["speciesname", "pft"]], on="speciesname", how="left")
    matched = matched.rename(columns={"pft": "pft_species"})

    n_total = matched.shape[0]
    n_species = int(matched["pft_species"].notna().sum())
    log.info(
        "Matched PFTs at species level: %d/%d (%.1f%%)",
        n_species,
        n_total,
        100.0 * n_species / max(1, n_total),
    )

    # 2) Genus-level fallback for remaining rows
    # Build genus mode PFT mapping from PFTs table
    pfts = pfts.assign(genus=lambda d: d["speciesname"].str.split().str[0])
    genus_mode = (
        pfts.dropna(subset=["genus", "pft"])
        .groupby("genus")["pft"]
        .agg(lambda s: s.value_counts().index[0])
        .to_frame("pft_genus")
        .reset_index()
    )

    # Derive genus in hydraulic table and merge genus-level PFT
    matched = matched.assign(genus=lambda d: d["speciesname"].str.split().str[0])
    matched = matched.merge(genus_mode, on="genus", how="left")

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
        "Added PFTs via genus fallback: %d (now %d/%d, %.1f%% total)",
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

    return matched


def _power_transform(
    df: pd.DataFrame, traits: list[str]
) -> tuple[pd.DataFrame, PowerTransformer]:
    """Apply Yeo-Johnson power transform to hydraulic traits and return the result.

    The transformer is fit jointly across the hydraulic trait columns and applied to
    those columns only. Returns the transformed DataFrame (same columns) and the
    fitted transformer.
    """
    for col in traits:
        if col not in df.columns:
            raise ValueError(f"Expected trait column '{col}' not found")

    trait_values = df[traits]
    log.info("Applying Yeo-Johnson to %d traits: %s", len(traits), traits)
    pt = PowerTransformer(method="yeo-johnson")
    transformed = pt.fit_transform(trait_values)

    df_t = df.copy()
    df_t[traits] = transformed

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

    # Save data (speciesname + transformed hydraulic traits)
    df.to_parquet(out_fp, index=False, compression="zstd")
    log.info(
        "Saved %s hydraulic traits to %s",
        "transformed" if transformer is not None else "untransformed",
        out_fp,
    )


if __name__ == "__main__":
    log.info("Starting Knighton hydraulic trait processing")
    main(cli())
    log.info("Hydraulic trait processing completed successfully!")
