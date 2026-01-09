"""Predict a single trait using trained models.

This module handles prediction for a single trait-trait_set combination,
supporting both standard prediction and Coefficient of Variation (CoV) calculation.
It uses Dask for parallel processing when batch size > 1.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Literal

import dask.dataframe as dd
import pandas as pd
from autogluon.tabular import TabularPredictor
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, df_to_dd, init_dask
from src.utils.dataset_utils import (
    get_cov_dir,
    get_predict_dir,
    get_predict_imputed_fn,
    get_predict_mask_fn,
)
from src.utils.df_utils import pipe_log, rasterize_points
from src.utils.raster_utils import pack_xr, xr_to_raster


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict a single trait using trained models."
    )
    parser.add_argument(
        "--trait",
        type=str,
        required=True,
        help="Trait to predict (e.g., 'leaf_N_per_dry_mass')",
    )
    parser.add_argument(
        "--trait-set",
        type=str,
        required=True,
        help="Trait set to use (e.g., 'Shrub_Tree_Grass')",
    )
    parser.add_argument(
        "--cov",
        action="store_true",
        help="Calculate Coefficient of Variation (instead of normal prediction)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "-b",
        "--batches",
        type=int,
        default=None,
        help="Number of batches for prediction (overrides config)",
    )
    parser.add_argument(
        "-n",
        "--n-workers",
        type=int,
        default=None,
        help="Number of workers (overrides config)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to params.yaml file (default: uses active config)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode"
    )
    return parser.parse_args()


def predict_partition(partition: pd.DataFrame, predictor_path: Path) -> pd.DataFrame:
    """Predict on a single partition.

    Args:
        partition: DataFrame partition with features and x, y coordinates
        predictor_path: Path to the AutoGluon predictor

    Returns:
        DataFrame with x, y coordinates and predictions
    """
    fold_predictor = TabularPredictor.load(str(predictor_path))

    predictions = fold_predictor.predict(
        partition.drop(columns=["x", "y"]), as_pandas=True
    )

    return pd.concat(
        [
            partition[["x", "y"]].reset_index(drop=True),
            predictions.reset_index(  # pyright: ignore[reportAttributeAccessIssue]
                drop=True
            ),
        ],
        axis=1,
    )


def predict_trait_ag_dask(
    data: dd.DataFrame,
    model_path: Path,
) -> pd.DataFrame:
    """Predict the trait using the given model in batches, optimized for Dask DataFrames.

    Args:
        data: Dask DataFrame with features and x, y coordinates
        model_path: Path to the full_model directory

    Returns:
        DataFrame with predictions indexed by (y, x)
    """
    log.info("Prediction with Dask...")
    return (
        data.map_partitions(predict_partition, predictor_path=model_path)
        .compute()
        .set_index(["y", "x"])
    )


def predict_cov_dask(
    predict_data: dd.DataFrame, cv_dir: Path, tmp_dir: Path
) -> pd.DataFrame:
    """Calculate the Coefficient of Variation for the given model using parallel predictions.

    Args:
        predict_data: Dask DataFrame with features and x, y coordinates
        cv_dir: Path to directory containing CV fold models
        tmp_dir: Directory to store intermediate fold predictions

    Returns:
        DataFrame with CoV values indexed by (y, x)
    """
    cv_predictions = []
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for fold_model_path in cv_dir.iterdir():
        cv_prediction_fn = Path(tmp_dir) / f"{fold_model_path.stem}.parquet"

        if not fold_model_path.is_dir():
            continue

        if cv_prediction_fn.exists():
            log.info("Skipping %s, already exists", cv_prediction_fn)
            cv_predictions.append(cv_prediction_fn)
            continue

        log.info("Predicting with %s...", fold_model_path.stem)
        predict_data.map_partitions(
            predict_partition, predictor_path=fold_model_path
        ).compute().set_index(["y", "x"]).to_parquet(cv_prediction_fn)

        cv_predictions.append(cv_prediction_fn)

    log.info("CV predictions complete. Re-loading...")
    dfs = [pd.read_parquet(f) for f in cv_predictions]

    log.info("Calculating CoV...")
    cov = (
        pd.concat(dfs, axis=1)
        # Add minimum value to all values. This is necessary because CoV is only
        # meaningful when zero is meaningful. If the data was power or log-transformed,
        # zero becomes meaningless, and may even result in the mean being zero or close
        # to zero, which would result in a CoV of infinity.
        .pipe(lambda _df: _df + abs(_df.min().min()))
        .pipe(lambda _df: _df.std(axis=1) / _df.mean(axis=1))  # CoV calculation
        .rename("cov")
        .to_frame()
    )

    return cov


def predict_dask(
    predict_data: dd.DataFrame, model_path: Path, cov: bool, tmp_dir: Path | None
) -> tuple[pd.DataFrame, Path | None]:
    """Predict the trait using the given model, with optional CoV calculation.

    Args:
        predict_data: Dask DataFrame with features and x, y coordinates
        model_path: Path to the trait_set directory (containing full_model and cv subdirs)
        cov: Whether to calculate CoV instead of standard prediction
        tmp_dir: Directory for temporary files (used for CoV calculation)

    Returns:
        Tuple of (predictions DataFrame, temp directory path or None)
    """
    if cov:
        if tmp_dir is None:
            raise ValueError("tmp_dir must be provided for CoV calculation")
        return predict_cov_dask(predict_data, model_path / "cv", tmp_dir), tmp_dir
    return predict_trait_ag_dask(predict_data, model_path / "full_model"), None


def load_predict_data(
    tmp_predict_fn: Path, batches: int = 1
) -> pd.DataFrame | dd.DataFrame:
    """Load masked predict data from disk or mask imputed features.

    Args:
        tmp_predict_fn: Path to store/load masked predict data
        batches: Number of batches (1 = pandas, >1 = Dask)

    Returns:
        DataFrame with masked features and x, y coordinates
    """
    log.info("Checking for existing masked predict data...")
    if not tmp_predict_fn.exists():
        log.info("No existing masked predict data found. Masking imputed features...")
        predict = (
            dd.read_parquet(get_predict_imputed_fn())
            .pipe(pipe_log, "Reading imputed predict features...")
            .compute()
            .reset_index(drop=True)
            .pipe(pipe_log, "Setting index to ['y', 'x']")
            .set_index(["y", "x"])
            .pipe(pipe_log, "Reading mask and masking imputed features...")
            .mask(
                dd.read_parquet(get_predict_mask_fn()).compute().set_index(["y", "x"])
            )
            .reset_index()
        )

        log.info("Writing masked predict data to disk for later usage...")
        predict.to_parquet(tmp_predict_fn, compression="zstd")

    else:
        log.info("Found existing masked predict data. Reading...")

    return (
        pd.read_parquet(tmp_predict_fn)
        if batches == 1
        else dd.read_parquet(tmp_predict_fn).repartition(npartitions=batches)
    )


def predict_single_trait(
    trait: str,
    trait_set: str,
    predict_data: pd.DataFrame | dd.DataFrame,
    models_dir: Path,
    out_dir: Path,
    res: int | float,
    crs: str,
    predict_cfg: ConfigBox,
    overwrite: bool = False,
    mode: Literal["predict", "cov"] = "predict",
) -> Path:
    """Predict a single trait and save to raster.

    Args:
        trait: Trait name
        trait_set: Trait set name
        predict_data: DataFrame with features and x, y coordinates
        models_dir: Base directory containing trained models
        out_dir: Output directory for predictions
        res: Spatial resolution
        crs: Coordinate reference system
        predict_cfg: Prediction configuration
        overwrite: Whether to overwrite existing output
        mode: Either "predict" for standard prediction or "cov" for CoV calculation

    Returns:
        Path to output file

    Raises:
        FileNotFoundError: If model directory not found
        ValueError: If model structure is invalid
    """
    dask: bool = predict_cfg.batches > 1
    cov: bool = mode == "cov"

    # Find model directory
    trait_dir = models_dir / trait
    if not trait_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {trait_dir}")

    # Find latest run
    autogluon_dir = trait_dir / "autogluon"
    if not autogluon_dir.exists():
        raise FileNotFoundError(f"AutoGluon directory not found: {autogluon_dir}")

    # Get latest run directory (highest number)
    run_dirs = [d for d in autogluon_dir.iterdir() if d.is_dir() and d.stem.isdigit()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in: {autogluon_dir}")
    latest_run = max(run_dirs, key=lambda d: int(d.stem))

    # Find trait_set directory
    trait_set_dir = latest_run / trait_set
    if not trait_set_dir.exists():
        raise FileNotFoundError(f"Trait set directory not found: {trait_set_dir}")

    # Check for required model directories
    full_model_dir = trait_set_dir / "full_model"
    if not full_model_dir.exists():
        raise ValueError(f"full_model directory not found: {full_model_dir}")

    if cov:
        cv_dir = trait_set_dir / "cv"
        if not cv_dir.exists():
            raise ValueError(f"cv directory not found: {cv_dir}")

    # Set up output path
    out_fn = (
        out_dir / trait / trait_set / f"{trait}_{trait_set}_{'cov' if cov else 'predict'}.tif"
    )
    out_fn.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and out_fn.exists():
        log.info("Output file already exists: %s", out_fn)
        return out_fn

    log.info("Predicting %s for %s/%s...", "CoV" if cov else "trait", trait, trait_set)

    # Set up Dask if needed
    tmp_dir = None
    if dask:
        if isinstance(predict_data, pd.DataFrame):
            predict_data = df_to_dd(predict_data, npartitions=predict_cfg.batches)

        # Set env vars according to n_workers and batches
        defined_cpus = os.environ.get("OMP_NUM_THREADS", None)
        defined_cpus = os.cpu_count() if defined_cpus is None else int(defined_cpus)
        num_cpus = str(defined_cpus // predict_cfg.n_workers)
        os.environ["OMP_NUM_THREADS"] = num_cpus
        os.environ["MKL_NUM_THREADS"] = num_cpus
        os.environ["OPENBLAS_NUM_THREADS"] = num_cpus
        os.environ["LOKY_MAX_CPU_COUNT"] = num_cpus

        client, _ = init_dask(
            dashboard_address=get_config().dask_dashboard,
            n_workers=predict_cfg.n_workers,
            threads_per_worker=1,
        )

        if cov:
            tmp_dir = out_dir / "tmp" / trait / trait_set
        pred, tmp_dir = predict_dask(predict_data, trait_set_dir, cov, tmp_dir)
        close_dask(client)
    else:
        # Ensure we have a pandas DataFrame for non-Dask prediction
        if isinstance(predict_data, dd.DataFrame):
            predict_data = predict_data.compute()
        assert isinstance(predict_data, pd.DataFrame)

        if cov:
            tmp_dir = out_dir / "tmp" / trait / trait_set
            tmp_dir.mkdir(parents=True, exist_ok=True)
            pred = predict_cov_dask(
                df_to_dd(predict_data, npartitions=1), trait_set_dir / "cv", tmp_dir
            )
        else:
            pred = predict_trait_ag_dask(
                df_to_dd(predict_data, npartitions=1), trait_set_dir / "full_model"
            )

    log.info("Writing predictions to raster...")
    pred_r = rasterize_points(pred, data_col="cov" if cov else trait, res=res, crs=crs)
    pred_r = pack_xr(pred_r)
    xr_to_raster(pred_r, out_fn)

    if tmp_dir is not None and tmp_dir.exists():
        log.info("Cleaning up temporary files...")
        shutil.rmtree(tmp_dir)

    log.info("Prediction complete: %s", out_fn)
    return out_fn


def main(args: argparse.Namespace, cfg: ConfigBox | None = None) -> Path:
    """Main function to predict a single trait.

    Args:
        args: Command-line arguments
        cfg: Configuration (if None, loads from get_config())

    Returns:
        Path to output file
    """
    if cfg is None:
        cfg = get_config()

    predict_cfg = cfg.predict[detect_system()]

    # Override config with CLI args if provided
    if args.batches is not None:
        predict_cfg.batches = args.batches
    if args.n_workers is not None:
        predict_cfg.n_workers = args.n_workers

    if not args.verbose:
        log.setLevel("WARNING")

    models_dir = cfg.models_dir
    mode: Literal["predict", "cov"] = "cov" if args.cov else "predict"
    out_dir = get_cov_dir(cfg) if args.cov else get_predict_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predict data
    tmp_predict_fn = Path(out_dir / "predict.parquet")
    predict_data = load_predict_data(tmp_predict_fn, predict_cfg.batches)

    # Run prediction
    try:
        out_fn = predict_single_trait(
            trait=args.trait,
            trait_set=args.trait_set,
            predict_data=predict_data,
            models_dir=models_dir,
            out_dir=out_dir,
            res=cfg.target_resolution,
            crs=cfg.crs,
            predict_cfg=predict_cfg,
            overwrite=args.overwrite,
            mode=mode,
        )
    finally:
        # Clean up temporary predict data if we created it
        if tmp_predict_fn.exists():
            log.info("Cleaning up temporary predict data...")
            tmp_predict_fn.unlink()

    log.info("Done!")
    return out_fn


if __name__ == "__main__":
    main(cli())
