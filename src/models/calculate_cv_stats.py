"""
Calculate cross-validation statistics for a trained model.

This module provides functions to calculate CV statistics after all fold models
have been trained. It uses pre-prepared XY data from the prepare_xy_data stage
to avoid redundant data loading and merging.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from box import ConfigBox
from sklearn import metrics
from sklearn.preprocessing import PowerTransformer

from src.conf.conf import get_config
from src.conf.environment import log
from src.models.run_utils import get_latest_run_id
from src.utils.spatial_utils import lat_weights, weighted_pearson_r
from src.visualization.figures.scatterplots import plot_observed_vs_predicted


proj_root = os.environ.get("PROJECT_ROOT", None)
if proj_root is None:
    raise ValueError("PROJECT_ROOT environment variable not set.")

def load_trait_metadata(trait_name: str, cfg: ConfigBox) -> dict[str, Any] | None:
    """
    Load transformation metadata JSON for a trait.

    Args:
        trait_name: Trait identifier (e.g., "X55")
        cfg: Configuration object

    Returns:
        Dictionary containing transformation metadata, or None if not found
    """
    if not cfg.traits.get("power_transform", False):
        return None

    transformer_dir = Path(proj_root) / cfg.traits.transformer_dir
    metadata_path = transformer_dir / f"{trait_name}_metadata.json"

    if not metadata_path.exists():
        log.warning(
            "Power transform enabled but metadata not found for %s: %s",
            trait_name,
            metadata_path,
        )
        return None

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata


def load_trait_transformer(trait_name: str, cfg: ConfigBox) -> PowerTransformer | None:
    """
    Load pickled power transformer for a trait.

    Args:
        trait_name: Trait identifier (e.g., "X55")
        cfg: Configuration object

    Returns:
        PowerTransformer object, or None if not found or trait not transformed
    """
    metadata = load_trait_metadata(trait_name, cfg)
    if metadata is None or not metadata.get("transformed", False):
        return None

    transformer_dir = Path(proj_root) / cfg.traits.transformer_dir
    transformer_path = transformer_dir / f"{trait_name}_transformer.pkl"

    if not transformer_path.exists():
        log.warning(
            "Transformer metadata indicates transformation but transformer file not found: %s",
            transformer_path,
        )
        return None

    with open(transformer_path, "rb") as f:
        transformer = pickle.load(f)

    return transformer


def get_stats(
    cv_obs_vs_pred: pd.DataFrame,
    resolution: int | float | None = None,
    wt_pearson: bool = False,
) -> dict[str, Any]:
    """
    Calculate statistics for a given DataFrame of observed and predicted values.

    Args:
        cv_obs_vs_pred: DataFrame with 'obs' and 'pred' columns
        resolution: Resolution for weighted Pearson correlation calculation
        wt_pearson: Whether to calculate weighted Pearson correlation

    Returns:
        Dictionary containing various performance metrics
    """
    obs = cv_obs_vs_pred.obs.to_numpy()
    pred = cv_obs_vs_pred.pred.to_numpy()

    r2 = metrics.r2_score(obs, pred)
    pearsonr = cv_obs_vs_pred[["obs", "pred"]].corr().iloc[0, 1]
    pearsonr_wt = None

    if wt_pearson and resolution is not None:
        pearsonr_wt = weighted_pearson_r(
            cv_obs_vs_pred.set_index(["y", "x"]),
            lat_weights(cv_obs_vs_pred.y.unique(), resolution),
        )

    root_mean_squared_error = metrics.root_mean_squared_error(obs, pred)
    norm_root_mean_squared_error = root_mean_squared_error / (
        cv_obs_vs_pred.obs.quantile(0.99) - cv_obs_vs_pred.obs.quantile(0.01)
    )
    mean_squared_error = np.mean((obs - pred) ** 2)
    mean_absolute_error = metrics.mean_absolute_error(obs, pred)
    median_absolute_error = metrics.median_absolute_error(obs, pred)

    return {
        "r2": r2,
        "pearsonr": pearsonr,
        "pearsonr_wt": pearsonr_wt,
        "root_mean_squared_error": root_mean_squared_error,
        "norm_root_mean_squared_error": norm_root_mean_squared_error,
        "mean_squared_error": mean_squared_error,
        "mean_absolute_error": mean_absolute_error,
        "median_absolute_error": median_absolute_error,
    }


def get_fold_wise_stats(
    fold_results: list[pd.DataFrame],
    resolution: int | float | None = None,
    wt_pearson: bool = False,
) -> dict[str, Any]:
    """
    Calculate statistics for each fold and return means and standard deviations.

    Args:
        fold_results: List of DataFrames, one per fold, each containing obs/pred columns
        resolution: Resolution for weighted Pearson correlation calculation
        wt_pearson: Whether to calculate weighted Pearson correlation

    Returns:
        Dictionary containing mean and std for each metric across folds
    """
    fold_stats = []

    for fold_df in fold_results:
        stats = get_stats(fold_df, resolution=resolution, wt_pearson=wt_pearson)
        fold_stats.append(stats)

    # Convert to DataFrame for easier calculation
    fold_stats_df = pd.DataFrame(fold_stats)

    # Calculate means and standard deviations across folds
    result = {}
    for metric in fold_stats_df.columns:
        # Skip None values (e.g., pearsonr_wt when not calculated)
        values = fold_stats_df[metric].dropna()
        if len(values) > 0:
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std())
        else:
            result[f"{metric}_mean"] = None
            result[f"{metric}_std"] = None

    return result


def save_scatterplot(
    cv_obs_vs_pred: pd.DataFrame,
    trait_id: str,
    output_path: Path,
    r_value: float | None = None,
    rmse: float | None = None,
    nrmse: float | None = None,
) -> None:
    """
    Create and save a scatterplot of observed vs predicted values.

    Args:
        cv_obs_vs_pred: DataFrame with 'obs' and 'pred' columns
        trait_id: Trait identifier for title
        output_path: Path to save the plot
        r_value: Pearson correlation value to display
        rmse: Root mean squared error to display
        nrmse: Normalized root mean squared error to display
    """
    log.info(f"Creating scatterplot: {output_path}")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    # Build stats dict for display
    stats = {}
    if r_value is not None:
        stats["Pearson's r"] = r_value
    if rmse is not None:
        stats["RMSE"] = rmse
    if nrmse is not None:
        stats["nRMSE"] = nrmse

    # Use the imported plotting function from scatterplots.py
    plot_observed_vs_predicted(
        ax=ax,
        observed=cv_obs_vs_pred.obs,
        predicted=cv_obs_vs_pred.pred,
        name=trait_id,
        density=True,
        stats=stats,
    )

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Scatterplot saved to {output_path}")


def load_xy_data(trait_id: str, trait_set: str, cfg: ConfigBox) -> pd.DataFrame:
    """
    Load pre-prepared XY data for a specific trait.

    This function uses the pre-prepared XY data from the prepare_xy_data stage,
    eliminating redundant data loading and merging.

    Note: For CV validation, we always use splot-only data regardless of the
    trait_set, as splot data is more reliable and represents the patterns we
    want the model to learn. This matches the validation data used during
    fold training in autogluon.py.

    Args:
        trait_id: Trait identifier (column name in Y data)
        trait_set: Trait set name (splot, gbif, or splot_gbif)
        cfg: Configuration object

    Returns:
        DataFrame with features, labels, and fold assignments for the trait
    """
    log.info(f"Loading pre-prepared XY data for {trait_id}...")
    xy_all = pd.read_parquet(Path(cfg.train.xy_data.fp))

    # Validate trait_set
    valid_trait_sets = {"splot", "gbif", "splot_gbif"}
    if trait_set not in valid_trait_sets:
        raise ValueError(f"Unknown trait_set: {trait_set}")

    # Always use splot-only data for CV validation to match training validation
    sources = ["s"]

    # Get fold column name for this trait
    fold_col = f"{trait_id}_fold"

    # Get all trait columns (anything that's not a feature or metadata column)
    # Features are numeric columns that don't end with "_fold" and aren't trait values
    # Metadata: x, y, source, is_test
    metadata_cols = {"x", "y", "source", "is_test"}

    # Get list of all trait columns (those that have corresponding _fold columns)
    # Filter out reliability columns that might have gotten fold columns created
    trait_value_cols = {
        col.replace("_fold", "")
        for col in xy_all.columns
        if col.endswith("_fold")
        and not col.replace("_fold", "").endswith("_reliability")
    }

    # Feature columns are numeric columns that aren't metadata, traits, fold columns,
    # or reliability columns
    feature_cols = [
        col
        for col in xy_all.columns
        if col not in metadata_cols
        and col not in trait_value_cols
        and not col.endswith("_fold")
        and not col.endswith("_reliability")
    ]

    # Select only the columns we need: metadata + this trait + this fold + features
    required_cols = ["x", "y", "source", trait_id, fold_col] + feature_cols
    if "is_test" in xy_all.columns:
        required_cols.append("is_test")

    # Filter to only columns that exist
    # (handle case where trait or fold might be missing)
    available_cols = [col for col in required_cols if col in xy_all.columns]

    xy_trait = (
        xy_all[available_cols]
        .query(f"source in {sources}")
        .dropna(subset=[trait_id, fold_col])
        .rename(columns={trait_id: "obs", fold_col: "fold"})
        .drop(
            columns=["source"]
            + (["is_test"] if "is_test" in available_cols else [])
        )
    )

    log.info(f"Loaded {len(xy_trait)} splot samples for {trait_id} ({trait_set})")
    return xy_trait


def generate_fold_predictions(
    fold_dir: Path, xy_fold: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate predictions for a single fold.

    Args:
        fold_dir: Directory containing the trained fold model
        xy_fold: DataFrame with features and labels for the fold

    Returns:
        DataFrame with x, y, obs, and pred columns
    """
    log.info(f"Loading model from {fold_dir}...")
    predictor = TabularPredictor.load(str(fold_dir))

    log.info(f"Generating predictions for {len(xy_fold)} samples...")
    # Drop obs and fold columns for prediction
    X_fold = xy_fold.drop(columns=["obs", "fold", "x", "y"])
    pred = predictor.predict(X_fold)

    # Combine with spatial coordinates and observations
    result = pd.DataFrame({
        "x": xy_fold["x"].values,
        "y": xy_fold["y"].values,
        "obs": xy_fold["obs"].values,
        "pred": pred.values if hasattr(pred, "values") else pred,
    })

    return result


def back_transform_predictions(
    cv_obs_vs_pred: pd.DataFrame,
    fold_results: list[pd.DataFrame] | None,
    trait_id: str,
    cfg: ConfigBox,
) -> tuple[pd.DataFrame | None, list[pd.DataFrame] | None]:
    """
    Back-transform predictions if data was transformed during training.

    Uses the new per-trait transformer system that loads metadata and transformers
    from cfg.traits.transformer_dir.

    Args:
        cv_obs_vs_pred: Combined CV predictions (in model/transformed space)
        fold_results: Individual fold results (optional, in model/transformed space)
        trait_id: Trait identifier (e.g., "X55")
        cfg: Configuration object

    Returns:
        Tuple of (back_transformed_cv_obs_vs_pred, back_transformed_fold_results)
        in original space. Returns (None, None) if no transformation was applied.
    """
    # Load trait metadata to check if transformation was applied
    metadata = load_trait_metadata(trait_id, cfg)
    if metadata is None or not metadata.get("transformed", False):
        log.info(f"No transformation applied to {trait_id}, skipping back-transform")
        return None, None

    # Load the transformer
    transformer = load_trait_transformer(trait_id, cfg)
    if transformer is None:
        log.warning(f"Could not load transformer for {trait_id}, skipping back-transform")
        return None, None

    log.info(
        f"Back-transforming {trait_id} predictions using {metadata.get('method', 'unknown')} method"
    )

    # Back-transform overall CV predictions
    inv_obs = transformer.inverse_transform(cv_obs_vs_pred[["obs"]].values).ravel()
    inv_pred = transformer.inverse_transform(cv_obs_vs_pred[["pred"]].values).ravel()

    cv_obs_vs_pred_original = cv_obs_vs_pred.assign(
        obs=inv_obs, pred=inv_pred
    ).dropna()

    # Back-transform fold results if provided
    fold_results_original = None
    if fold_results:
        transformed_folds = []
        for fold in fold_results:
            fold_inv_obs = transformer.inverse_transform(fold[["obs"]].values).ravel()
            fold_inv_pred = transformer.inverse_transform(fold[["pred"]].values).ravel()

            transformed_folds.append(
                fold.assign(obs=fold_inv_obs, pred=fold_inv_pred).dropna()
            )

        fold_results_original = transformed_folds

    return cv_obs_vs_pred_original, fold_results_original


def aggregate_feature_importance(
    training_dir: Path,
    n_folds: int,
    cfg: ConfigBox,
) -> pd.DataFrame | None:
    """
    Aggregate feature importance across all CV folds.

    Args:
        training_dir: Directory containing trained models
        n_folds: Number of CV folds
        cfg: Configuration object

    Returns:
        DataFrame with aggregated feature importance, or None if not available
    """
    cv_dir = training_dir / "cv"

    # Check if feature importance files exist
    fold_fi_files = []
    for fold_id in range(n_folds):
        fi_path = cv_dir / f"fold_{fold_id}" / cfg.train.feature_importance
        if fi_path.exists():
            fold_fi_files.append(fi_path)

    if len(fold_fi_files) == 0:
        log.info("No feature importance files found, skipping aggregation")
        return None

    if len(fold_fi_files) < n_folds:
        log.warning(
            f"Only {len(fold_fi_files)}/{n_folds} feature importance files found"
        )

    # Load all feature importance files
    log.info(f"Loading {len(fold_fi_files)} feature importance files...")
    fold_fi_dfs = []
    for fi_path in fold_fi_files:
        df = pd.read_csv(fi_path)
        fold_fi_dfs.append(df)

    # Combine all fold feature importances
    combined_fi = pd.concat(fold_fi_dfs, ignore_index=True)

    # Get the feature column name (could be 'feature' or index column)
    # AutoGluon's feature_importance typically has features as index or 'feature' column
    if 'feature' in combined_fi.columns:
        feature_col = 'feature'
    else:
        # Assume first column is the feature identifier
        feature_col = combined_fi.columns[0]

    # Aggregate across folds for each feature
    log.info("Aggregating feature importance across folds...")

    # Columns to aggregate (numeric columns excluding fold and feature identifier)
    numeric_cols = combined_fi.select_dtypes(include=[np.number]).columns.tolist()
    if 'fold' in numeric_cols:
        numeric_cols.remove('fold')

    # Group by feature and calculate mean and std
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ['mean', 'std', 'count']

    aggregated = combined_fi.groupby(feature_col).agg(agg_dict)

    # Flatten multi-level columns
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
    aggregated = aggregated.reset_index()

    log.info(f"Aggregated feature importance for {len(aggregated)} features")

    return aggregated


def calculate_cv_stats_for_trait(
    trait: str,
    trait_set: str,
    cfg: ConfigBox,
    include_fold_results: bool = True,
    overwrite: bool = False,
    run_id: str | None = None,
) -> None:
    """
    Calculate CV statistics for a specific trait and trait set.

    Args:
        trait: Trait identifier
        trait_set: Trait set name (splot, gbif, or splot_gbif)
        cfg: Configuration object
        include_fold_results: Whether to calculate fold-wise statistics
        overwrite: Whether to overwrite existing CV stats
        run_id: Run ID to use. If None, uses the most recent run.
    """
    log.info(f"Calculating CV statistics for {trait} ({trait_set})...")

    # Build path to model directory
    trait_models_dir = Path(cfg.models.dir_fp) / trait / cfg.train.arch

    # Find run ID if not provided
    if run_id is None:
        run_id = get_latest_run_id(trait_models_dir)
        if run_id is None:
            raise FileNotFoundError(
                f"No training runs found in: {trait_models_dir}"
            )
        log.info(f"Using run: {run_id}")

    training_dir = trait_models_dir / run_id / trait_set

    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")

    cv_dir = training_dir / "cv"
    if not cv_dir.exists():
        raise FileNotFoundError(f"CV directory not found: {cv_dir}")

    # Check if CV stats already exist
    results_path = training_dir / cfg.train.eval_results
    cv_obs_vs_pred_path = training_dir / "cv_obs_vs_pred.parquet"
    scatterplot_original_path = (
        training_dir / "cv_obs_vs_pred_scatter_original.pdf"
    )
    scatterplot_transformed_path = (
        training_dir / "cv_obs_vs_pred_scatter_transformed.pdf"
    )
    scatterplot_single_path = training_dir / "cv_obs_vs_pred_scatter.pdf"

    # If stats and predictions exist, check if we only need to generate the plot(s)
    if not overwrite and results_path.exists() and cv_obs_vs_pred_path.exists():
        # Check trait metadata to determine if transformation was applied
        metadata = load_trait_metadata(trait, cfg)
        is_transformed = metadata is not None and metadata.get("transformed", False)

        missing_plots = []
        if is_transformed:
            if not scatterplot_original_path.exists():
                missing_plots.append("original")
            if not scatterplot_transformed_path.exists():
                missing_plots.append("transformed")
        else:
            if not scatterplot_single_path.exists():
                missing_plots.append("single")

        if missing_plots:
            log.info(
                f"CV stats exist but scatterplot(s) missing for {trait} ({trait_set}). "
                f"Generating: {', '.join(missing_plots)}..."
            )
            # Load existing predictions and stats
            cv_obs_vs_pred_model = pd.read_parquet(cv_obs_vs_pred_path)
            stats_df = pd.read_csv(results_path)

            if is_transformed:
                # Generate both scatterplots
                # Original space stats
                stats_original = stats_df.query("scale == 'original'").iloc[0]
                r_value_original = stats_original["pearsonr"]

                # Model space stats
                stats_model = stats_df.query("scale == 'model'").iloc[0]
                r_value_model = stats_model["pearsonr"]

                # Back-transform predictions for original space plot
                transformer = load_trait_transformer(trait, cfg)
                if transformer is not None:
                    inv_obs = transformer.inverse_transform(
                        cv_obs_vs_pred_model[["obs"]].values
                    ).ravel()
                    inv_pred = transformer.inverse_transform(
                        cv_obs_vs_pred_model[["pred"]].values
                    ).ravel()
                    cv_obs_vs_pred_original = cv_obs_vs_pred_model.assign(
                        obs=inv_obs, pred=inv_pred
                    )

                    if "original" in missing_plots:
                        save_scatterplot(
                            cv_obs_vs_pred_original,
                            trait,
                            scatterplot_original_path,
                            r_value=r_value_original,
                        )

                if "transformed" in missing_plots:
                    save_scatterplot(
                        cv_obs_vs_pred_model,
                        trait,
                        scatterplot_transformed_path,
                        r_value=r_value_model,
                    )
            else:
                # Only one scatterplot needed
                stats_row = stats_df.query("scale == 'original'").iloc[0]
                r_value = stats_row["pearsonr"]

                save_scatterplot(
                    cv_obs_vs_pred_model,
                    trait,
                    scatterplot_single_path,
                    r_value=r_value,
                )

            log.info(f"Scatterplot(s) generated for {trait} ({trait_set})!")
            return
        else:
            log.info(
                f"CV stats already exist for {trait} ({trait_set}). "
                "Use --overwrite to recalculate."
            )
            return

    # Check if all folds are complete
    n_folds = cfg.train.cv_splits.n_splits
    fold_dirs = sorted([d for d in cv_dir.glob("fold_*") if d.is_dir()])

    if len(fold_dirs) < n_folds:
        raise ValueError(
            f"Not all CV folds complete ({len(fold_dirs)}/{n_folds}) "
            f"for {trait} ({trait_set})"
        )

    # Check if CV predictions already exist and can be reused
    if not overwrite and cv_obs_vs_pred_path.exists():
        log.info(f"Loading existing CV predictions from {cv_obs_vs_pred_path}...")
        cv_obs_vs_pred = pd.read_parquet(cv_obs_vs_pred_path)
        # Can't reconstruct individual fold results from combined file
        fold_results = None
    else:
        # Load pre-prepared XY data
        xy_trait = load_xy_data(trait, trait_set, cfg)

        # Generate predictions for each fold
        fold_results = []
        for fold_dir in fold_dirs:
            fold_id = int(fold_dir.stem.split("_")[-1])
            log.info(f"Processing fold {fold_id}...")

            # Filter data for this fold
            xy_fold = xy_trait.query(f"fold == {fold_id}")

            # Generate predictions
            fold_pred = generate_fold_predictions(fold_dir, xy_fold)
            fold_results.append(fold_pred)

        # Concatenate all fold predictions
        log.info("Concatenating fold predictions...")
        cv_obs_vs_pred = pd.concat(fold_results, ignore_index=True)

        # Save combined CV predictions
        log.info(f"Saving CV predictions to {cv_obs_vs_pred_path}...")
        cv_obs_vs_pred.to_parquet(cv_obs_vs_pred_path, index=False)

    # Calculate statistics
    log.info("Calculating statistics...")
    pearsonr_wt = cfg.crs == "EPSG:4326"

    # Back-transform if necessary
    cv_obs_vs_pred_orig = cv_obs_vs_pred.copy()
    fold_results_orig = (
        fold_results.copy() if fold_results and include_fold_results else None
    )

    cv_obs_vs_pred_trans, fold_results_trans = back_transform_predictions(
        cv_obs_vs_pred, fold_results_orig, trait, cfg
    )

    # If back-transformation was applied, use transformed data for final stats
    if cv_obs_vs_pred_trans is not None:
        cv_obs_vs_pred = cv_obs_vs_pred_trans
        fold_results_for_stats = fold_results_trans
    else:
        fold_results_for_stats = fold_results_orig

    # Calculate overall statistics
    log.info("Calculating overall statistics...")
    stats_overall = get_stats(
        cv_obs_vs_pred,
        cfg.target_resolution,
        wt_pearson=pearsonr_wt,
    )

    # Calculate fold-wise statistics if requested
    if include_fold_results and fold_results_for_stats:
        log.info("Calculating fold-wise statistics...")
        stats_foldwise = get_fold_wise_stats(
            fold_results_for_stats,
            cfg.target_resolution,
            wt_pearson=pearsonr_wt,
        )
        stats = {**stats_overall, **stats_foldwise}
    else:
        stats = stats_overall

    # Create results DataFrame
    all_stats = pd.DataFrame()

    if cv_obs_vs_pred_trans is not None:
        # We have both model space (transformed) and original space stats
        # Calculate stats on model space (original predictions before back-transform)
        stats_model_overall = get_stats(
            cv_obs_vs_pred_orig,
            cfg.target_resolution,
            wt_pearson=pearsonr_wt,
        )

        if include_fold_results and fold_results_orig:
            stats_model_foldwise = get_fold_wise_stats(
                fold_results_orig,
                cfg.target_resolution,
                wt_pearson=pearsonr_wt,
            )
            stats_model = {**stats_model_overall, **stats_model_foldwise}
        else:
            stats_model = stats_model_overall

        # Combine both - stats on original space and model space
        all_stats = pd.concat(
            [
                pd.DataFrame(stats, index=[0]).assign(scale="original"),
                pd.DataFrame(stats_model, index=[0]).assign(scale="model"),
            ],
            ignore_index=True,
        )
    else:
        # No transformation applied - only original scale stats
        all_stats = pd.DataFrame(stats, index=[0]).assign(scale="original")

    # Save statistics
    results_path = training_dir / cfg.train.eval_results
    log.info(f"Saving statistics to {results_path}...")

    # Backup old results if they exist
    if results_path.exists():
        old_path = results_path.with_name(
            f"{results_path.stem}_old{results_path.suffix}"
        )
        results_path.rename(old_path)

    all_stats.to_csv(results_path, index=False)

    # Create CV scatterplot(s)
    if cv_obs_vs_pred_trans is not None:
        # Transformation was applied - generate both scatterplots
        log.info(f"Generating CV scatterplots for {trait} (both scales)...")

        # Original space scatterplot
        save_scatterplot(
            cv_obs_vs_pred,
            trait,
            scatterplot_original_path,
            r_value=stats_overall["pearsonr"],
        )

        # Model space scatterplot (using original predictions before back-transform)
        save_scatterplot(
            cv_obs_vs_pred_orig,
            trait,
            scatterplot_transformed_path,
            r_value=stats_model_overall["pearsonr"],
        )
    else:
        # No transformation - generate single scatterplot
        log.info(f"Generating CV scatterplot for {trait}...")
        save_scatterplot(
            cv_obs_vs_pred,
            trait,
            scatterplot_single_path,
            r_value=stats_overall["pearsonr"],
        )

    # Generate test set scatterplot(s) if test predictions exist
    test_obs_vs_pred_path = training_dir / "test_obs_vs_pred.parquet"
    test_results_path = training_dir / "test_eval_results.csv"
    test_scatterplot_original_path = (
        training_dir / "test_obs_vs_pred_scatter_original.pdf"
    )
    test_scatterplot_transformed_path = (
        training_dir / "test_obs_vs_pred_scatter_transformed.pdf"
    )
    test_scatterplot_single_path = training_dir / "test_obs_vs_pred_scatter.pdf"

    if test_obs_vs_pred_path.exists() and test_results_path.exists():
        # Check trait metadata to determine if transformation was applied
        metadata = load_trait_metadata(trait, cfg)
        is_transformed = metadata is not None and metadata.get("transformed", False)

        # Determine which plots need to be generated
        plots_needed = False
        if is_transformed:
            if (
                not test_scatterplot_original_path.exists()
                or not test_scatterplot_transformed_path.exists()
                or overwrite
            ):
                plots_needed = True
        else:
            if not test_scatterplot_single_path.exists() or overwrite:
                plots_needed = True

        if plots_needed:
            log.info(f"Generating test set scatterplot(s) for {trait} ({trait_set})...")
            test_obs_vs_pred_model = pd.read_parquet(test_obs_vs_pred_path)
            test_results_df = pd.read_csv(test_results_path)

            if is_transformed:
                # Generate both scatterplots for test set
                # Get r-values from test results for both scales
                test_r_original = test_results_df.query("scale == 'original'").iloc[0][
                    "pearsonr"
                ]
                test_r_model = test_results_df.query("scale == 'model'").iloc[0][
                    "pearsonr"
                ]

                # Back-transform test predictions for original space plot
                transformer = load_trait_transformer(trait, cfg)
                if transformer is not None:
                    inv_obs = transformer.inverse_transform(
                        test_obs_vs_pred_model[["obs"]].values
                    ).ravel()
                    inv_pred = transformer.inverse_transform(
                        test_obs_vs_pred_model[["pred"]].values
                    ).ravel()
                    test_obs_vs_pred_original = test_obs_vs_pred_model.assign(
                        obs=inv_obs, pred=inv_pred
                    )

                    # Original space scatterplot
                    save_scatterplot(
                        test_obs_vs_pred_original,
                        trait,
                        test_scatterplot_original_path,
                        r_value=test_r_original,
                    )

                # Model space scatterplot
                save_scatterplot(
                    test_obs_vs_pred_model,
                    trait,
                    test_scatterplot_transformed_path,
                    r_value=test_r_model,
                )
                msg = f"Test set scatterplots saved (both scales) for {trait}"
                log.info(f"{msg} ({trait_set})")
            else:
                # Only one scatterplot needed
                test_r_value = test_results_df["pearsonr"].iloc[0]

                save_scatterplot(
                    test_obs_vs_pred_model,
                    trait,
                    test_scatterplot_single_path,
                    r_value=test_r_value,
                )
                log.info(f"Test set scatterplot saved for {trait} ({trait_set})")
        else:
            log.info(f"Test set scatterplot(s) already exist for {trait} ({trait_set})")
    else:
        log.info(f"No test set predictions found for {trait} ({trait_set})")

    # Aggregate feature importance across folds
    log.info(f"Aggregating feature importance for {trait} ({trait_set})...")
    feature_importance_path = training_dir / "feature_importance_aggregated.csv"

    if not overwrite and feature_importance_path.exists():
        log.info(
            f"Aggregated feature importance already exists for {trait} ({trait_set}). "
            "Use --overwrite to recalculate."
        )
    else:
        aggregated_fi = aggregate_feature_importance(
            training_dir,
            cfg.train.cv_splits.n_splits,
            cfg,
        )

        if aggregated_fi is not None:
            log.info(f"Saving aggregated feature importance to {feature_importance_path}...")
            aggregated_fi.to_csv(feature_importance_path, index=False)
            log.info(f"Aggregated feature importance saved!")
        else:
            log.info("No feature importance to aggregate")

    log.info(f"CV statistics calculation complete for {trait} ({trait_set})!")


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate CV statistics for a trained model."
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to parameters file.",
    )
    parser.add_argument(
        "--trait",
        type=str,
        required=True,
        help="Trait identifier.",
    )
    parser.add_argument(
        "--trait-set",
        type=str,
        required=True,
        choices=["splot", "gbif", "splot_gbif"],
        help="Trait set name.",
    )
    parser.add_argument(
        "--no-fold-results",
        action="store_true",
        help="Don't calculate fold-wise statistics.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing CV stats.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID to use (format: run_YYYYMMDD_HHMMSS). "
        "If not specified, uses the most recent run.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None, cfg: ConfigBox | None = None) -> None:
    """Main function for calculating CV statistics."""
    if args is None:
        args = cli()
    if cfg is None:
        cfg = get_config(params_path=args.params)

    calculate_cv_stats_for_trait(
        trait=args.trait,
        trait_set=args.trait_set,
        cfg=cfg,
        include_fold_results=not args.no_fold_results,
        overwrite=args.overwrite,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
