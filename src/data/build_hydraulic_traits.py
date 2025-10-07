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
    parser.add_argument(
        "-t", "--traits", type=list, default=None, help="List of traits to process"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Load, clean, transform, and save Knighton hydraulic traits."""
    cfg = get_config(args.params) if args.params is not None else get_config()

    # Input path (Excel)
    df = _load_knighton_excel(Path(cfg.traits.raw), args.traits)

    # Clean species names per notebook: drop non-binomial, lowercase
    df = _clean_species_names(df)

    # Transform traits
    df_t, transformer = _power_transform(df)

    # Output directory under data/interim/try/hydraulic/
    out_dir = Path(cfg.interim_dir) / cfg.trydb.interim.dir / HYDRAULIC_DIRNAME
    _save_outputs(df_t, transformer, out_dir)


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
        df.rename(columns={"spec.name": "speciesname"})
        .loc[:, ["speciesname", *traits]]
        .astype(
            {
                "speciesname": "string[pyarrow]",
                **{c: "float32" for c in traits},
            }
        )
    )

    log.info(
        "Loaded %d rows with %d hydraulic traits", df.shape[0], len(HYDRAULIC_TRAITS)
    )
    return df


def _clean_species_names(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-binomial taxa (var./subsp./hybrids) and standardize species names.

    This follows the notebook decision to drop varieties/subspecies/cultivars rather
    than aggregating, by keeping only rows where the species string contains exactly
    two words. Then convert to lowercase.
    """
    if "speciesname" not in df.columns:
        raise ValueError("'speciesname' column not found during cleaning stage")

    # Keep only binomials (two words) and normalize case
    before = df.shape[0]
    df = df[df["speciesname"].str.split().str.len() == 2].copy()
    df["speciesname"] = df["speciesname"].str.lower()
    after = df.shape[0]
    log.info(
        "Filtered non-binomial taxa: kept %d of %d rows (%.2f%%)",
        after,
        before,
        100 * after / max(1, before),
    )
    return df


def _power_transform(df: pd.DataFrame) -> tuple[pd.DataFrame, PowerTransformer]:
    """Apply Yeo-Johnson power transform to hydraulic traits and return the result.

    The transformer is fit jointly across the hydraulic trait columns and applied to
    those columns only. Returns the transformed DataFrame (same columns) and the
    fitted transformer.
    """
    for col in HYDRAULIC_TRAITS:
        if col not in df.columns:
            raise ValueError(f"Expected trait column '{col}' not found")

    trait_values = df[HYDRAULIC_TRAITS].astype("float32")
    log.info(
        "Applying Yeo-Johnson to %d traits: %s", len(HYDRAULIC_TRAITS), HYDRAULIC_TRAITS
    )
    pt = PowerTransformer(method="yeo-johnson")
    transformed = pt.fit_transform(trait_values)

    df_t = df.copy()
    # Convert transformed array to DataFrame for easier assignment
    transformed_df = pd.DataFrame(transformed, columns=HYDRAULIC_TRAITS, index=df.index)
    for col in HYDRAULIC_TRAITS:
        df_t[col] = transformed_df[col].astype("float32")

    return df_t, pt


def _save_outputs(
    df: pd.DataFrame, transformer: PowerTransformer, out_dir: Path
) -> None:
    """Save transformed traits as parquet and transformer as pickle to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_fp = out_dir / "traits.parquet"
    transformer_fp = out_dir / "power_transformer.pkl"

    # Save data (speciesname + transformed hydraulic traits)
    df.to_parquet(parquet_fp, index=False, compression="zstd")
    log.info("Saved transformed hydraulic traits to %s", parquet_fp)

    # Save transformer
    with open(transformer_fp, "wb") as f:
        pickle.dump(transformer, f)
    log.info("Saved PowerTransformer to %s", transformer_fp)


if __name__ == "__main__":
    log.info("Starting Knighton hydraulic trait processing")
    main()
    log.info("Hydraulic trait processing completed successfully!")
