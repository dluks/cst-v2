"""Utility functions for model training."""

import numpy as np
import pandas as pd

from src.utils.dataset_utils import get_cv_splits_dir


def set_yx_index(df: pd.DataFrame) -> pd.DataFrame:
    """Set the DataFrame index to "y" and "x"."""
    if not df.index.names == ["y", "x"]:
        return df.set_index(["y", "x"])
    return df


def assign_splits(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Assign the cross-validation splits to the DataFrame based on the label column."""
    splits = pd.read_parquet(get_cv_splits_dir() / f"{label_col}.parquet")

    return df.pipe(set_yx_index).merge(
        splits.pipe(set_yx_index), validate="m:1", right_index=True, left_index=True
    )


def filter_trait_set(df: pd.DataFrame, trait_set: str) -> pd.DataFrame:
    """Filter the DataFrame based on the trait set."""
    # Check if "splot" and "gbif" are in split trait set
    if sorted(trait_set.split("_")) == ["gbif", "splot"]:
        # Remove duplicated rows in favor of "source" == "s"
        return df.sort_values(by="source", ascending=False).drop_duplicates(
            subset=["x", "y"], keep="first"
        )

    # Otherwise return the rows where "source" == "s" or "g" (depending
    # on trait_set)
    return df[df.source == trait_set[0]]


def calculate_gbif_weight(df: pd.DataFrame) -> float:
    """
    Calculate GBIF weight as the inverse proportion of SPLOT samples.

    This weights GBIF samples such that their total contribution equals
    the SPLOT contribution, effectively telling the model that SPLOT
    samples are more valuable.

    Formula: w_gbif = n_splot / n_total

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with source column

    Returns
    -------
    float
        Weight to apply to GBIF samples
    """
    n_splot = (df.source == "s").sum()
    n_total = len(df)
    return n_splot / n_total if n_total > 0 else 1.0


def assign_weights(
    df: pd.DataFrame,
    w_splot: int | float = 1.0,
    w_gbif: int | float | None = None,
    trait_name: str | None = None,
    splot_always_reliable: bool = True,
) -> pd.DataFrame:
    """
    Assign weights to the DataFrame based on the source column.

    If only one source is present, assign a uniform weight of 1.0.

    If a trait_name is provided and the corresponding reliability column exists
    (e.g., '{trait_name}_reliability'), the final weights will be the product of
    source weights and reliability weights:
        final_weight = source_weight * reliability_weight

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with source column and optional reliability columns
    w_splot : int | float, default=1.0
        Weight for sPlot records (source == "s")
    w_gbif : int | float | None, default=None
        Weight for GBIF records (source == "g"). If None, automatically
        calculated as n_splot / n_total (inverse proportion weighting).
    trait_name : str | None, default=None
        Trait name to look for reliability weights. If provided, will use
        the column '{trait_name}_reliability' if it exists.
    splot_always_reliable : bool, default=True
        If True, SPLOT samples always get reliability=1.0 regardless of the
        reliability column value. GBIF samples still use the reliability column.
        This accounts for the fact that SPLOT plot-level surveys represent
        comprehensive sampling even when n is low.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'weights' column added
    """
    # Calculate source-based weights
    if df.source.unique().size == 1:
        source_weights = 1.0
    else:
        # Auto-calculate GBIF weight if not provided
        if w_gbif is None:
            w_gbif = calculate_gbif_weight(df)
        source_weights = np.where(df.source == "s", w_splot, w_gbif)

    # If trait_name is provided, check for reliability column
    if trait_name is not None:
        reliability_col = f"{trait_name}_reliability"
        if reliability_col in df.columns:
            if splot_always_reliable:
                # SPLOT always gets reliability=1.0, GBIF uses column value
                reliability_weights = np.where(
                    df.source == "s",
                    1.0,
                    df[reliability_col].values,
                )
            else:
                reliability_weights = df[reliability_col].values

            final_weights = source_weights * reliability_weights
            return df.assign(weights=final_weights)

    # No reliability weighting - use source weights only
    return df.assign(weights=source_weights)
