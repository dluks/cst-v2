"""Train a single AutoGluon model for a specific trait, trait set, and fold/full model."""

import argparse
import datetime
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import activate_env, log
from src.utils.df_utils import pipe_log
from src.utils.log_utils import set_dry_run_text, suppress_dask_logging
from src.utils.training_utils import assign_weights, filter_trait_set


def now() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def prep_full_xy(
    feats: dd.DataFrame,
    labels: dd.DataFrame,
    label_col: str,
    cfg: ConfigBox,
) -> pd.DataFrame:
    """
    Prepare the input data for modeling by merging features and labels.

    Args:
        feats (dd.DataFrame): The input features.
        labels (dd.DataFrame): The input labels.
        label_col (str): The column name of the labels.
        cfg: Configuration object

    Returns:
        pd.DataFrame: The prepared input data for modeling.
    """
    log.info("Loading splits...")
    cv_splits_dir = Path(cfg.train.cv_splits.dir_fp)
    splits = (
        dd.read_parquet(cv_splits_dir / f"{label_col}.parquet")
        .compute()
        .set_index(["y", "x"])
    )

    log.info("Merging splits and label data...")
    label = (
        labels[["x", "y", label_col, "source"]]
        .compute()
        .set_index(["y", "x"])
        .merge(splits, validate="m:1", right_index=True, left_index=True)
    )

    return (
        feats.compute()
        .set_index(["y", "x"])
        .pipe(pipe_log, "Merging features and label data...")
        .merge(label, validate="1:m", right_index=True, left_index=True)
        .reset_index()
    )


def load_data(cfg: ConfigBox) -> tuple[dd.DataFrame, dd.DataFrame]:
    """Load the input data for modeling."""
    feats = dd.read_parquet(
        Path(cfg.train.predict.fp)
    ).reset_index()  # X data is already indexed by y and x
    labels = dd.read_parquet(Path(cfg.train.Y.fp))
    return feats, labels


def get_or_create_run_dir(
    trait_name: str,
    trait_set: str,
    debug: bool = False,
    cfg: ConfigBox | None = None,
) -> Path:
    """
    Get or create the run directory for a trait and trait set.

    Args:
        trait_name: Name of the trait
        trait_set: Name of the trait set (splot, gbif, or splot_gbif)
        debug: Whether to use debug mode
        cfg: Configuration object

    Returns:
        Path to the run directory for the trait set
    """
    if cfg is None:
        cfg = get_config()

    # Build path: models/{product_code}/{trait}/{arch}/{trait_set}
    trait_models_dir = Path(cfg.models.dir_fp) / trait_name / cfg.train.arch
    runs_dir = trait_models_dir / "debug" if debug else trait_models_dir
    training_dir = runs_dir / trait_set
    training_dir.mkdir(parents=True, exist_ok=True)

    return training_dir


def determine_presets(cfg_presets: list[str] | None, n_train_samples: int) -> list[str]:
    if cfg_presets is None:
        return (
            ["extreme"]
            if n_train_samples < 30000
            else ["best"] + ["optimize_for_deployment"]
        )
    else:
        return cfg_presets + ["optimize_for_deployment"]


def train_cv_fold(
    trait_name: str,
    trait_set: str,
    fold_id: int,
    xy: pd.DataFrame,
    training_dir: Path,
    cfg: ConfigBox,
    sample: float = 1.0,
    dry_run: bool = False,
) -> None:
    """
    Train a single cross-validation fold model.

    Args:
        trait_name: Name of the trait to train
        trait_set: Trait set to use (splot, gbif, or splot_gbif)
        fold_id: Fold number to train
        xy: Full training data with features and labels
        training_dir: Directory to save the trained model
        cfg: Configuration object
        sample: Fraction of data to sample for training
        dry_run: Whether to perform a dry run without training
    """
    dry_run_text = set_dry_run_text(dry_run)

    log.info(
        "Training %s model for fold %d with %s trait set...%s",
        trait_name,
        fold_id,
        trait_set.upper(),
        dry_run_text,
    )

    if dry_run:
        log.info("Dry run complete for fold %d", fold_id)
        return

    # Sample data if requested
    if sample < 1.0:
        log.info("Subsampling %.1f%% of data...", sample * 100)
        xy = xy.sample(frac=sample, random_state=cfg.random_seed)

    cv_dir = training_dir / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    fold_model_path = cv_dir / f"fold_{fold_id}"

    # Columns to drop (not features)
    cols_to_drop = ["x", "y", "source", "fold"]
    if "is_test" in xy.columns:
        cols_to_drop.append("is_test")
    if f"{trait_name}_reliability" in xy.columns:
        cols_to_drop.append(f"{trait_name}_reliability")

    # Prepare training and validation data
    train = TabularDataset(
        xy[xy["fold"] != fold_id]
        .pipe(filter_trait_set, trait_set)
        .dropna(subset=[trait_name])
        .pipe(assign_weights, w_gbif=cfg.train.weights.gbif)
        .drop(columns=cols_to_drop)
        .reset_index(drop=True)
    )
    val = TabularDataset(
        xy[xy["fold"] == fold_id]
        .query("source == 's'")
        .dropna(subset=[trait_name])
        .assign(weights=1.0)
        .drop(columns=cols_to_drop)
        .reset_index(drop=True)
    )

    presets = determine_presets(cfg.autogluon.presets, len(train))
    log.info(
        "Using presets '%s' for %d training samples (fold %d)",
        presets,
        len(train),
        fold_id,
    )

    log.info("Training fold %d model...", fold_id)
    predictor = TabularPredictor(
        label=trait_name,
        sample_weight="weights",  # pyright: ignore[reportArgumentType]
        path=str(fold_model_path),
    ).fit(
        train,
        presets=presets,
        included_model_types=cfg.autogluon.included_model_types,
        excluded_model_types=cfg.autogluon.excluded_model_types,
        time_limit=cfg.autogluon.cv_fit_time_limit * 3600,  # Convert hours to seconds
        num_bag_folds=cfg.autogluon.num_bag_folds,
        num_stack_levels=cfg.autogluon.num_stack_levels,
        dynamic_stacking=cfg.autogluon.dynamic_stacking,
    )

    # Calculate feature importance
    if cfg.autogluon.feature_importance:
        log.info("Calculating feature importance...")
        features = predictor.feature_metadata_in.get_features()
        feat_ds_map = {
            "canopy_height": {"startswith": True, "match": "ETH"},
            "soilgrids": {"startswith": False, "match": "cm_mean"},
            "modis": {"startswith": True, "match": "sur_refl"},
            "vodca": {"startswith": True, "match": "vodca"},
            "worldclim": {"startswith": True, "match": "wc2.1"},
        }

        # Generate a list of tuples of (dataset, [features]) for each dataset
        datasets = []
        for ds, ds_info in feat_ds_map.items():
            if ds_info["startswith"]:
                ds_feats = [
                    feat for feat in features if feat.startswith(ds_info["match"])
                ]
            else:
                ds_feats = [
                    feat for feat in features if feat.endswith(ds_info["match"])
                ]
            datasets.append((ds, ds_feats))

        # Add all features as well
        datasets += features

        feature_importance = predictor.feature_importance(
            val,
            features=datasets,
            time_limit=cfg.autogluon.FI_time_limit * 3600,  # Convert hours to seconds
            num_shuffle_sets=cfg.autogluon.FI_num_shuffle_sets,
        ).assign(fold=fold_id)

        feature_importance.to_csv(fold_model_path / cfg.train.feature_importance)

    # Evaluate the model
    log.info("Evaluating fold %d model...", fold_id)
    eval_results = predictor.evaluate(val, auxiliary_metrics=True, detailed_report=True)

    # Normalize RMSE by the 99th percentile - 1st percentile range of the target
    norm_factor = val[trait_name].quantile(0.99) - val[trait_name].quantile(0.01)
    eval_results["norm_root_mean_squared_error"] = (
        eval_results["root_mean_squared_error"] / norm_factor
    )

    pd.DataFrame({col: [val] for col, val in eval_results.items()}).assign(
        fold=fold_id
    ).to_csv(fold_model_path / cfg.train.eval_results)

    predictor.save_space()

    # Mark fold as complete
    fold_complete_flag = cv_dir / f"cv_fold_{fold_id}_complete.flag"
    fold_complete_flag.touch()

    log.info("Fold %d training complete", fold_id)


def train_full_model(
    trait_name: str,
    trait_set: str,
    xy: pd.DataFrame,
    training_dir: Path,
    cfg: ConfigBox,
    sample: float = 1.0,
    dry_run: bool = False,
) -> None:
    """
    Train a full model on all data for a trait and trait set.

    Args:
        trait_name: Name of the trait to train
        trait_set: Trait set to use (splot, gbif, or splot_gbif)
        xy: Full training data with features and labels
        training_dir: Directory to save the trained model
        cfg: Configuration object
        sample: Fraction of data to sample for training
        dry_run: Whether to perform a dry run without training
    """
    dry_run_text = set_dry_run_text(dry_run)

    log.info(
        "Training %s full model with %s trait set...%s",
        trait_name,
        trait_set.upper(),
        dry_run_text,
    )

    if dry_run:
        log.info("Dry run complete for full model")
        return

    # Sample data if requested
    if sample < 1.0:
        log.info("Subsampling %.1f%% of data...", sample * 100)
        xy = xy.sample(frac=sample, random_state=cfg.random_seed)

    full_model_path = training_dir / "full_model"

    # Columns to drop (not features)
    cols_to_drop = ["x", "y", "source", "fold"]
    if "is_test" in xy.columns:
        cols_to_drop.append("is_test")
    if f"{trait_name}_reliability" in xy.columns:
        cols_to_drop.append(f"{trait_name}_reliability")

    # Prepare training data
    train_full = TabularDataset(
        xy.pipe(filter_trait_set, trait_set)
        .dropna(subset=[trait_name])
        .pipe(assign_weights, w_gbif=cfg.train.weights.gbif)
        .drop(columns=cols_to_drop)
    )

    presets = determine_presets(cfg.autogluon.presets, len(train_full))
    log.info(
        "Using presets '%s' for %d training samples (full model)",
        presets,
        len(train_full),
    )

    log.info("Training full model...")
    predictor = TabularPredictor(
        label=trait_name,
        sample_weight="weights",  # pyright: ignore[reportArgumentType]
        path=str(full_model_path),
    ).fit(
        train_full,
        presets=presets,
        included_model_types=cfg.autogluon.included_model_types,
        excluded_model_types=cfg.autogluon.excluded_model_types,
        time_limit=cfg.autogluon.full_fit_time_limit * 3600,  # Convert hours to seconds
        num_bag_folds=cfg.autogluon.num_bag_folds,
        num_stack_levels=cfg.autogluon.num_stack_levels,
        dynamic_stacking=cfg.autogluon.dynamic_stacking,
    )

    predictor.save_space()

    # Mark full model as complete
    full_model_complete_flag = training_dir / "full_model_complete.flag"
    full_model_complete_flag.touch()

    log.info("Full model training complete")


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a single AutoGluon model for a specific trait, trait set, and fold/full model"
    )
    parser.add_argument(
        "-t", "--trait", type=str, required=True, help="Trait name to train"
    )
    parser.add_argument(
        "-ts",
        "--trait-set",
        type=str,
        required=True,
        choices=["splot", "gbif", "splot_gbif"],
        help="Trait set to use (splot, gbif, or splot_gbif)",
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        default=None,
        help="Fold number to train (if not specified, trains full model)",
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        default=None,
        help="Path to parameters file",
    )
    parser.add_argument(
        "-s", "--sample", type=float, default=1.0, help="Fraction of data to sample"
    )
    parser.add_argument("-r", "--resume", action="store_true", help="Resume training")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Dry run")
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite existing models"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for training a single model."""
    args = cli()
    activate_env()
    suppress_dask_logging()

    # Load configuration
    cfg = get_config(Path(args.params).absolute() if args.params else None)

    # Get or create run directory
    training_dir = get_or_create_run_dir(args.trait, args.trait_set, args.debug, cfg)

    # Check if model already exists
    if args.fold is not None:
        fold_model_path = training_dir / "cv" / f"fold_{args.fold}"
        fold_complete_flag = training_dir / "cv" / f"cv_fold_{args.fold}_complete.flag"
        if fold_complete_flag.exists() and not args.overwrite:
            log.info(
                "Fold %d model for %s/%s already trained. Use --overwrite to retrain.",
                args.fold,
                args.trait,
                args.trait_set,
            )
            return
    else:
        full_model_complete_flag = training_dir / "full_model_complete.flag"
        if full_model_complete_flag.exists() and not args.overwrite:
            log.info(
                "Full model for %s/%s already trained. Use --overwrite to retrain.",
                args.trait,
                args.trait_set,
            )
            return

    # Load pre-generated XY data
    xy_path = Path(cfg.train.xy_data.fp).resolve()

    if not xy_path.exists():
        log.error(
            "XY data not found at %s. Please run the prepare_xy_data stage first.",
            xy_path,
        )
        raise FileNotFoundError(
            "XY data not found. Run 'dvc repro prepare_xy_data' first."
        )

    log.info("Loading pre-generated XY data for %s...", args.trait)
    xy_cols = dd.read_parquet(xy_path).columns.to_list()

    # Get trait column names to filter out other traits
    all_y_cols = (
        pd.read_parquet(Path(cfg.train.Y.fp), engine="pyarrow")
        .columns.difference(["x", "y", "source"])
        .to_list()
    )

    # Select relevant columns for this trait
    # Keep: x, y, source, trait value, trait-specific fold column, and all features
    # Exclude: other trait values and other fold columns
    fold_col = f"{args.trait}_fold"
    reliability_col = f"{args.trait}_reliability"
    cols_to_exclude = set()

    # Exclude other trait values and their reliability columns
    cols_to_exclude.update(col for col in all_y_cols if col != args.trait)
    cols_to_exclude.update(
        col
        for col in all_y_cols
        if col.endswith("_reliability") and col != reliability_col
    )

    # Exclude other fold columns which only exist in the xy_data dataframe
    cols_to_exclude.update(
        col for col in xy_cols if col.endswith("_fold") and col != fold_col
    )

    # Select columns to keep
    cols_to_keep = [col for col in xy_cols if col not in cols_to_exclude]

    # Filter and rename fold column
    xy = (
        pd.read_parquet(xy_path, engine="pyarrow", columns=cols_to_keep)
        .rename(columns={fold_col: "fold"})
        .reset_index(drop=True)
    )

    log.info("All loaded columns: %s", xy.columns.to_list())

    log.info("Loaded data shape: %s", xy.shape)

    # Train the model
    if args.fold is not None:
        train_cv_fold(
            args.trait,
            args.trait_set,
            args.fold,
            xy,
            training_dir,
            cfg,
            args.sample,
            args.dry_run,
        )
    else:
        train_full_model(
            args.trait,
            args.trait_set,
            xy,
            training_dir,
            cfg,
            args.sample,
            args.dry_run,
        )

    log.info("Done! \U00002705")


if __name__ == "__main__":
    main()
