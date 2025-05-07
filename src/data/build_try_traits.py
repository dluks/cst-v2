"""Extracts, cleans, and gets species mean trait values from TRY data."""

import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from box import ConfigBox
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.df_utils import filter_outliers
from src.utils.trait_utils import (
    clean_species_name,
    filter_pft,
    get_trait_number_from_id,
)


def main(cfg: ConfigBox = get_config()) -> None:
    """Extract, clean, and get species mean trait values from TRY data."""
    try_params = f"try{cfg.try_version}"
    try_raw_dir = Path(cfg.raw_dir, cfg.trydb.raw[try_params].dir)
    try_prep_dir = Path(cfg.interim_dir, cfg.trydb.interim.dir)

    log.info(f"Processing TRY version {cfg.try_version} data")
    if cfg.try_version not in [5, 6]:
        raise ValueError(f"Unsupported TRY version: {cfg.try_version}")
    if cfg.try_version == 6:
        trait_cols = [f"X.X{t}." for t in cfg.datasets.Y.traits]
    else:
        trait_cols = [f"X{t}" for t in cfg.datasets.Y.traits]

    log.info(f"Selected {len(trait_cols)} traits from TRY: {cfg.datasets.Y.traits}")
    trait_cols = ["Species"] + trait_cols

    log.info(f"Extracting raw TRY traits from: {cfg.trydb.raw[try_params].zipfile_csv}")
    with zipfile.ZipFile(try_raw_dir / cfg.trydb.raw[try_params].zip, "r") as zip_ref:  # noqa: SIM117
        with zip_ref.open(cfg.trydb.raw[try_params].zipfile_csv) as zf:
            if cfg.try_version == 6:
                with zipfile.ZipFile(zf, "r") as nested_zip_ref:  # noqa: SIM117
                    with nested_zip_ref.open(nested_zip_ref.namelist()[0]) as csvfile:
                        log.info("Reading from nested zip file")
                        traits = pd.read_csv(
                            csvfile,
                            encoding="latin-1",
                            usecols=trait_cols,
                        )
            else:
                traits = pd.read_csv(zf, encoding="latin-1", usecols=trait_cols)

    log.info(
        f"Loaded raw trait data: {traits.shape[0]} rows, {traits.shape[1]} columns"
    )

    log.info("Standardizing trait IDs and computing species mean trait values...")
    traits = traits.astype({"Species": "string[pyarrow]"}).pipe(standardize_trait_ids)
    log.info(f"After standardizing trait IDs: {list(traits.columns)}")

    # Check if filtering by quantile range
    if cfg.trydb.interim.quantile_range:
        log.info(
            f"Filtering outliers with quantile range: {cfg.trydb.interim.quantile_range}"
        )
    else:
        log.info("No outlier filtering applied")

    mean_filt_traits = (
        traits.pipe(_filter_if_specified, trait_cols, cfg.trydb.interim.quantile_range)
        .pipe(clean_species_name, "Species", "speciesname")
        .drop(columns=["Species"])
        .groupby("speciesname")
        .mean()
    )
    log.info(f"Computed mean trait values for {mean_filt_traits.shape[0]} species")

    # Match with PFTs
    log.info(f"Filtering by plant functional types: {cfg.PFT}")
    log.info(f"Loading PFTs from {cfg.trydb.raw.pfts}")
    pft_path = Path(cfg.raw_dir, cfg.trydb.raw.pfts)
    pft_columns = ["AccSpeciesName", "pft"]
    if pft_path.suffix == ".csv":
        pfts = pd.read_csv(
            pft_path,
            encoding="latin-1",
            usecols=pft_columns,
        )
    elif pft_path.suffix == ".parquet":
        pfts = pd.read_parquet(pft_path, columns=pft_columns)
    else:
        raise ValueError(f"Unsupported PFT file format: {pft_path.suffix}")

    log.info(f"Loaded PFT data: {pfts.shape[0]} rows")
    pfts = (
        pfts.pipe(filter_pft, cfg.PFT)
        .astype(
            {
                "AccSpeciesName": "string[pyarrow]",
                "pft": "category",
            }
        )
        .dropna(subset=["AccSpeciesName"])
        .pipe(clean_species_name, "AccSpeciesName", "speciesname")
        .drop(columns=["AccSpeciesName"])
        .drop_duplicates(subset=["speciesname"])
        .set_index("speciesname")
    )
    log.info(f"Filtered to {pfts.shape[0]} species in requested PFT: {cfg.PFT}")

    log.info("Matching trait data with filtered PFTs...")
    initial_count = mean_filt_traits.shape[0]
    mean_filt_traits = mean_filt_traits.join(pfts, how="inner").drop(columns=["pft"])
    log.info(
        f"After PFT matching: {mean_filt_traits.shape[0]} species retained ({initial_count - mean_filt_traits.shape[0]} removed, {mean_filt_traits.shape[0] / initial_count:.1%} kept)"
    )

    # Apply transformation
    log.info(f"Applying transformation: {cfg.trydb.interim.transform}")
    transformer_path = try_prep_dir / cfg.trydb.interim.transformer_fn
    mean_filt_traits = mean_filt_traits.pipe(
        _transform,
        cfg.trydb.interim.transform,
        transformer_fn=transformer_path,
    )
    log.info(f"Saved transformer to {transformer_path}")

    # Apply PCA if specified
    if hasattr(cfg.trydb.interim, "perform_pca") and cfg.trydb.interim.perform_pca:
        log.info("Performing PCA on trait data...")
        pca_path = try_prep_dir / cfg.trydb.interim.pca_fn
        mean_filt_traits = _apply_pca(
            mean_filt_traits,
            pca_fn=pca_path,
            n_components=cfg.trydb.interim.pca_n_components,
        )
    else:
        log.info("Skipping PCA, using full trait set")

    # Reset index after all transformations
    mean_filt_traits = mean_filt_traits.reset_index()
    log.info(
        f"Final dataset: {mean_filt_traits.shape[0]} species, {mean_filt_traits.shape[1]} columns"
    )

    log.info("Saving filtered and mean trait values...")
    out_fn = try_prep_dir / cfg.trydb.interim.filtered
    if out_fn.exists():
        out_fn.unlink()
        log.info(f"Removed existing file: {out_fn}")
    mean_filt_traits.to_parquet(out_fn, index=False, compression="zstd")
    log.info(f"Saved processed trait data to {out_fn}")


def _filter_if_specified(
    df: pd.DataFrame, trait_cols: list[str], quantile_range: tuple[float, float] | None
) -> pd.DataFrame:
    """Filter out outliers if specified."""
    if quantile_range is not None:
        original_size = df.shape[0]
        # Filter only numeric columns and exclude string columns like "Species"
        numeric_cols = [
            col
            for col in trait_cols
            if col in df and pd.api.types.is_numeric_dtype(df[col])
        ]
        if numeric_cols:
            log.info(
                f"Filtering outliers in {len(numeric_cols)} numeric columns: {numeric_cols}"
            )
            filtered_df = filter_outliers(
                df, cols=numeric_cols, quantiles=quantile_range
            )
            log.info(
                f"Filtered outliers: removed {original_size - filtered_df.shape[0]} rows ({(original_size - filtered_df.shape[0]) / original_size:.2%})"
            )
            return filtered_df
        else:
            log.warning("No numeric columns found for outlier filtering")
            return df
    return df


def _log_transform_long_tails(df: pd.DataFrame, keep: list[str] | None) -> pd.DataFrame:
    """Log-transform columns with long tails."""
    if keep is None:
        keep = []

    trait_cols = [col for col in df.columns if col.startswith("X")]
    log.info(f"Found {len(trait_cols)} trait columns")

    long_tailed_cols = [
        col for col in trait_cols if get_trait_number_from_id(col) not in keep
    ]

    if not long_tailed_cols:
        log.info("No long-tailed columns to log-transform.")
        return df

    log.info(f"Log-transforming {len(long_tailed_cols)} long-tailed columns")
    log.debug(f"Log-transformed columns: {long_tailed_cols}")
    return df.assign(
        **{f"{col}_ln": (df[col].apply(np.log1p)) for col in long_tailed_cols}
    ).drop(columns=long_tailed_cols)


def _power_transform(df: pd.DataFrame, transformer_fn: Path) -> pd.DataFrame:
    """Power transform (Yeo-Johnson) the data and save the lambdas for later."""
    log.info(f"Applying Yeo-Johnson power transformation to {df.shape[1]} columns")
    pt = PowerTransformer(method="yeo-johnson")
    df_t = pt.fit_transform(df)

    log.info(f"Transformation lambdas: {pt.lambdas_}")

    if transformer_fn.exists():
        transformer_fn.unlink()
    with open(transformer_fn, "wb") as f:
        pickle.dump(pt, f)
    log.info(f"Saved PowerTransformer to {transformer_fn}")

    return pd.DataFrame(df_t, columns=df.columns, index=df.index)


def _standard_normalize(df: pd.DataFrame, transformer_fn: Path) -> pd.DataFrame:
    """Standardize the data using StandardScaler and save the scaler."""
    log.info(f"Applying standard normalization to {df.shape[1]} columns")
    scaler = StandardScaler()
    df_t = scaler.fit_transform(df)

    # Log original data statistics (before standardization)
    if (
        hasattr(scaler, "mean_")
        and hasattr(scaler, "scale_")
        and scaler.mean_ is not None
        and scaler.scale_ is not None
    ):
        if len(scaler.mean_) > 0 and len(scaler.scale_) > 0:
            log.info(
                f"Original data means (before standardization): [{np.min(scaler.mean_):.4f}, {np.max(scaler.mean_):.4f}]"
            )
            log.info(
                f"Original data std devs (before standardization): [{np.min(scaler.scale_):.4f}, {np.max(scaler.scale_):.4f}]"
            )

    # Optionally validate the standardization worked correctly
    df_transformed = pd.DataFrame(df_t, columns=df.columns)
    means = df_transformed.mean()
    stds = df_transformed.std()
    log.info(
        f"After standardization - mean range: [{means.min():.4f}, {means.max():.4f}] (should be ~0)"
    )
    log.info(
        f"After standardization - std range: [{stds.min():.4f}, {stds.max():.4f}] (should be ~1)"
    )

    if transformer_fn.exists():
        transformer_fn.unlink()
    with open(transformer_fn, "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"Saved StandardScaler to {transformer_fn}")

    return pd.DataFrame(df_t, columns=df.columns, index=df.index)


def _apply_pca(df: pd.DataFrame, pca_fn: Path, n_components=0.95) -> pd.DataFrame:
    """Apply PCA to reduce trait dimensionality.

    Args:
        df: DataFrame with trait columns
        pca_fn: Path to save PCA object
        n_components: Number of components to keep (if float, it represents
                      variance percentage to retain)

    Returns:
        DataFrame with reduced trait dimensions
    """
    log.info(
        f"Performing PCA on {df.shape[1]} traits with target explained variance: {n_components}"
    )

    # Check for missing values
    original_rows = df.shape[0]
    missing_count = df.isna().any(axis=1).sum()

    if missing_count > 0:
        missing_percent = missing_count / original_rows
        log.info(
            f"Found {missing_count} rows ({missing_percent:.2%}) with missing values"
        )

        if missing_percent <= 0.1:  # We can drop up to 10% of the data
            log.info(
                f"Dropping {missing_count} rows with missing values (within acceptable 10% threshold)"
            )
            df_clean = df.dropna()
            log.info(
                f"Retained {df_clean.shape[0]} rows ({df_clean.shape[0] / original_rows:.2%} of original data)"
            )
        else:
            error_msg = (
                f"Too many missing values for PCA: {missing_percent:.2%} of rows contain NaNs. "
                f"Data should be imputed before applying PCA."
            )
            log.error(error_msg)
            raise ValueError(error_msg)
    else:
        log.info("No missing values detected in the data")
        df_clean = df

    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(df_clean)

    # Create naming for principal components
    pc_cols = [f"PC{i + 1}" for i in range(pca.n_components_)]

    # Create new dataframe with PCA results
    pca_df = pd.DataFrame(X_pca, columns=pc_cols, index=df_clean.index)

    # Log the variance explained
    variance = pca.explained_variance_ratio_
    cumulative = np.cumsum(variance)
    log.info(
        f"PCA reduced {df_clean.shape[1]} traits to {len(pc_cols)} principal components"
    )
    log.info(f"Top 3 component variances: {variance[:3]}")
    log.info(f"Cumulative variance explained: {cumulative[-1]:.4f}")

    # Additional details on the PCA components
    top_component_loadings = []
    for i in range(min(3, pca.n_components_)):
        # Get indices of largest absolute values in each component
        top_indices = np.argsort(np.abs(pca.components_[i]))[-3:]
        top_features = [df_clean.columns[j] for j in top_indices]
        top_values = [pca.components_[i][j] for j in top_indices]
        top_component_loadings.append(
            f"PC{i + 1}: "
            + ", ".join(
                [f"{feat}={val:.3f}" for feat, val in zip(top_features, top_values)]
            )
        )

    log.info("Top feature loadings by component:")
    for loading in top_component_loadings:
        log.info(f"  {loading}")

    # Save PCA object
    if pca_fn.exists():
        pca_fn.unlink()
    with open(pca_fn, "wb") as f:
        pickle.dump(pca, f)
    log.info(f"Saved PCA model to {pca_fn}")

    return pca_df


def _transform(df: pd.DataFrame, transform: str | None, **kwargs) -> pd.DataFrame:
    """Apply a transformation to the data."""
    if transform is None:
        log.info("No transformation requested, skipping")
        # Write a placeholder transformer file to satisfy DVC
        with open(kwargs["transformer_fn"], "wb") as f:
            pickle.dump(None, f)
        return df

    if transform == "log":
        log.info("Log-transforming data with long-tailed distributions...")
        return _log_transform_long_tails(df, **kwargs)
    if transform == "power":
        log.info("Power-transforming data...")
        return _power_transform(df, **kwargs)
    if transform == "norm":
        log.info("Standard normalizing data...")
        return _standard_normalize(df, **kwargs)

    log.error(f"Unknown transformation requested: {transform}")
    raise ValueError(f"Unknown transformation: {transform}")


def standardize_trait_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize trait IDs between TRY5 and TRY6."""
    old_trait_cols = df.columns[df.columns.str.startswith("X")]
    log.info(f"Standardizing {len(old_trait_cols)} trait column IDs")
    new_trait_cols = [col.split(".")[1] for col in old_trait_cols if "." in col]

    # If no dots in column names, we might already have standardized names
    if not new_trait_cols:
        log.info("No trait ID standardization needed, names already in standard format")
        return df

    return df.rename(columns=dict(zip(old_trait_cols, new_trait_cols)))


if __name__ == "__main__":
    log.info("Starting TRY trait data processing")
    main()
    log.info("TRY trait processing completed successfully!")
