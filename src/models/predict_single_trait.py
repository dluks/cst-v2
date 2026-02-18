"""Predict a single trait using trained models.

This module handles prediction for a single trait-trait_set combination,
supporting both standard prediction and Coefficient of Variation (CoV) calculation.
"""

import argparse
import os
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import dask.dataframe as dd
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dataset_utils import (
    get_cov_dir,
    get_predict_dir,
)
from src.utils.df_utils import rasterize_points
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


# Module-level state for multiprocessing workers
_worker_state: dict = {}


def _init_predict_worker(model_path_str: str, n_threads: int) -> None:
    """Initialize a prediction worker process.

    Called once per worker when the Pool is created. Sets thread limits
    to avoid oversubscription, then loads the AutoGluon model.
    """
    global _worker_state
    thread_str = str(n_threads)
    os.environ["OMP_NUM_THREADS"] = thread_str
    os.environ["OPENBLAS_NUM_THREADS"] = thread_str
    os.environ["MKL_NUM_THREADS"] = thread_str
    _worker_state["predictor"] = TabularPredictor.load(model_path_str)


def _predict_row_groups(args: tuple[str, list[int]]) -> pd.DataFrame:
    """Predict on specific parquet row groups.

    Each worker loads its assigned row groups from the parquet file,
    runs prediction, and returns results indexed by (y, x).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    predict_fp_str, row_group_indices = args
    predictor = _worker_state["predictor"]

    pf = pq.ParquetFile(predict_fp_str)
    tables = [pf.read_row_group(i) for i in row_group_indices]
    data = pa.concat_tables(tables).to_pandas()

    # Ensure x/y are columns (they may be stored as the parquet index)
    if "x" not in data.columns or "y" not in data.columns:
        data = data.reset_index()

    coords = data[["x", "y"]]
    features = data.drop(columns=["x", "y"])

    predictions = predictor.predict(features, as_pandas=True)

    result = pd.concat(
        [coords.reset_index(drop=True), predictions.reset_index(drop=True)],
        axis=1,
    )
    return result.set_index(["y", "x"])


def predict_trait_ag(
    data: pd.DataFrame | dd.DataFrame | None,
    model_path: Path,
    n_workers: int = 1,
    predict_fp: Path | None = None,
) -> pd.DataFrame:
    """Predict using model loaded once.

    Loads the AutoGluon model once and runs prediction on the entire dataset.
    When n_workers > 1, uses parallel chunked prediction where each worker
    loads a subset of the data from the parquet file independently.

    Args:
        data: DataFrame with features and x, y coordinates (None when parallel)
        model_path: Path to the full_model directory
        n_workers: Number of parallel workers (1 = single-process)
        predict_fp: Path to predict parquet file (required when n_workers > 1)

    Returns:
        DataFrame with predictions indexed by (y, x)
    """
    if n_workers > 1:
        if predict_fp is None:
            raise ValueError("predict_fp is required for parallel prediction")
        return _predict_parallel(predict_fp, model_path, n_workers)

    if data is None:
        raise ValueError("data is required for single-process prediction")

    log.info("Loading AutoGluon predictor from %s...", model_path)
    predictor = TabularPredictor.load(str(model_path))

    # Convert to pandas if Dask
    if isinstance(data, dd.DataFrame):
        log.info("Computing Dask DataFrame to pandas...")
        data = data.compute()

    log.info("Running predictions on %d rows...", len(data))
    coords = data[["x", "y"]]
    features = data.drop(columns=["x", "y"])

    predictions = predictor.predict(features, as_pandas=True)

    result = pd.concat(
        [
            coords.reset_index(drop=True),
            predictions.reset_index(drop=True),
        ],
        axis=1,
    )

    return result.set_index(["y", "x"])


def _predict_parallel(
    predict_fp: Path, model_path: Path, n_workers: int
) -> pd.DataFrame:
    """Run prediction in parallel using multiprocessing.

    Splits the parquet file's row groups across workers. Each worker
    loads its subset of the data and the model independently.

    Args:
        predict_fp: Path to predict parquet file
        model_path: Path to the model directory
        n_workers: Number of parallel workers
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(predict_fp))
    n_row_groups = pf.metadata.num_row_groups

    n_cpus = os.cpu_count() or n_workers
    threads_per_worker = max(1, n_cpus // n_workers)

    log.info(
        "Parallel prediction: %d workers, %d row groups, %d threads/worker",
        n_workers,
        n_row_groups,
        threads_per_worker,
    )

    # Distribute row groups across workers
    chunks = np.array_split(range(n_row_groups), n_workers)
    work_items = [
        (str(predict_fp), list(indices))
        for indices in chunks
        if len(indices) > 0
    ]

    with Pool(
        len(work_items),
        initializer=_init_predict_worker,
        initargs=(str(model_path), threads_per_worker),
    ) as pool:
        results = pool.map(_predict_row_groups, work_items)

    log.info("Concatenating %d chunk results...", len(results))
    return pd.concat(results)


def predict_cov(
    predict_data: pd.DataFrame | dd.DataFrame | None,
    cv_dir: Path,
    tmp_dir: Path,
    n_workers: int = 1,
    predict_fp: Path | None = None,
) -> pd.DataFrame:
    """Calculate the Coefficient of Variation using CV fold models.

    Loads each CV fold model once and predicts on the entire dataset.

    Args:
        predict_data: DataFrame with features and x, y coordinates (None when parallel)
        cv_dir: Path to directory containing CV fold models
        tmp_dir: Directory to store intermediate fold predictions
        n_workers: Number of parallel workers for each fold's prediction
        predict_fp: Path to predict parquet file (required when n_workers > 1)

    Returns:
        DataFrame with CoV values indexed by (y, x)
    """
    cv_predictions = []
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Convert to pandas once if Dask
    if isinstance(predict_data, dd.DataFrame):
        log.info("Computing Dask DataFrame to pandas...")
        predict_data = predict_data.compute()

    for fold_model_path in cv_dir.iterdir():
        cv_prediction_fn = Path(tmp_dir) / f"{fold_model_path.stem}.parquet"

        if not fold_model_path.is_dir():
            continue

        if cv_prediction_fn.exists():
            log.info("Skipping %s, already exists", cv_prediction_fn)
            cv_predictions.append(cv_prediction_fn)
            continue

        log.info("Predicting with %s...", fold_model_path.stem)
        # Use predict_trait_ag which loads model once
        pred = predict_trait_ag(predict_data, fold_model_path, n_workers, predict_fp)
        pred.to_parquet(cv_prediction_fn)

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


def predict(
    predict_data: pd.DataFrame | dd.DataFrame | None,
    model_path: Path,
    cov: bool,
    tmp_dir: Path | None,
    n_workers: int = 1,
    predict_fp: Path | None = None,
) -> tuple[pd.DataFrame, Path | None]:
    """Predict the trait using the given model, with optional CoV calculation.

    Args:
        predict_data: DataFrame with features and x, y coordinates (None when parallel)
        model_path: Path to the trait_set directory (containing full_model and cv subdirs)
        cov: Whether to calculate CoV instead of standard prediction
        tmp_dir: Directory for temporary files (used for CoV calculation)
        n_workers: Number of parallel workers
        predict_fp: Path to predict parquet file (required when n_workers > 1)

    Returns:
        Tuple of (predictions DataFrame, temp directory path or None)
    """
    if cov:
        if tmp_dir is None:
            raise ValueError("tmp_dir must be provided for CoV calculation")
        cv_dir = model_path / "cv"
        return (
            predict_cov(
                predict_data, cv_dir, tmp_dir, n_workers, predict_fp
            ),
            tmp_dir,
        )
    full_model = model_path / "full_model"
    return (
        predict_trait_ag(predict_data, full_model, n_workers, predict_fp),
        None,
    )


def load_predict_data(
    predict_fp: Path, batches: int = 1
) -> pd.DataFrame | dd.DataFrame:
    """Load predict data from disk.

    Args:
        predict_fp: Path to predict features parquet file
        batches: Number of batches (1 = pandas, >1 = Dask)

    Returns:
        DataFrame with features and x, y coordinates
    """
    log.info("Loading predict data from %s...", predict_fp)
    if not predict_fp.exists():
        raise FileNotFoundError(f"Predict data not found: {predict_fp}")

    # Reset index to convert x/y from index to columns
    if batches == 1:
        return pd.read_parquet(predict_fp).reset_index()
    else:
        return dd.read_parquet(predict_fp).reset_index().repartition(npartitions=batches)


def predict_single_trait(
    trait: str,
    trait_set: str,
    predict_data: pd.DataFrame | dd.DataFrame | None,
    models_dir: Path,
    out_dir: Path,
    res: int | float,
    crs: str,
    predict_cfg: ConfigBox,
    dask_dashboard: str,
    overwrite: bool = False,
    mode: Literal["predict", "cov"] = "predict",
    predict_fp: Path | None = None,
) -> Path:
    """Predict a single trait and save to raster.

    Args:
        trait: Trait name
        trait_set: Trait set name
        predict_data: DataFrame with features and x, y coordinates
            (None when using parallel prediction)
        models_dir: Base directory containing trained models
        out_dir: Output directory for predictions
        res: Spatial resolution
        crs: Coordinate reference system
        predict_cfg: Prediction configuration
        dask_dashboard: Dask dashboard address
        overwrite: Whether to overwrite existing output
        mode: Either "predict" or "cov" for CoV calculation
        predict_fp: Path to predict parquet file (for parallel mode)

    Returns:
        Path to output file

    Raises:
        FileNotFoundError: If model directory not found
        ValueError: If model structure is invalid
    """
    cov: bool = mode == "cov"

    # Find model directory
    trait_dir = models_dir / trait
    if not trait_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {trait_dir}")

    # Find latest run
    autogluon_dir = trait_dir / "autogluon"
    if not autogluon_dir.exists():
        raise FileNotFoundError(f"AutoGluon directory not found: {autogluon_dir}")

    # Get latest run directory (pattern: run_YYYYMMDD_HHMMSS)
    run_dirs = [
        d for d in autogluon_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in: {autogluon_dir}")
    latest_run = max(run_dirs, key=lambda d: d.name)

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

    # Set up temp dir for CoV calculation
    tmp_dir = None
    if cov:
        tmp_dir = out_dir / "tmp" / trait / trait_set
        tmp_dir.mkdir(parents=True, exist_ok=True)

    # Run prediction (handles both pandas and Dask DataFrames)
    n_workers = getattr(predict_cfg, "n_workers", 1)
    pred, tmp_dir = predict(
        predict_data, trait_set_dir, cov, tmp_dir, n_workers, predict_fp
    )

    log.info("Writing predictions to raster...")
    pred_r = rasterize_points(
        pred, data_cols="cov" if cov else trait, res=res, crs=crs
    )
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
        cfg = get_config(params_path=getattr(args, "params", None))

    predict_cfg = cfg.predict[detect_system()]

    # Override config with CLI args if provided
    if args.batches is not None:
        predict_cfg.batches = args.batches
    if args.n_workers is not None:
        predict_cfg.n_workers = args.n_workers

    if not args.verbose:
        log.setLevel("WARNING")

    models_dir = Path(cfg.models.dir_fp)
    mode: Literal["predict", "cov"] = "cov" if args.cov else "predict"
    out_dir = get_cov_dir(cfg) if args.cov else get_predict_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predict data (skip when using parallel workers)
    predict_fp = Path(cfg.train.predict.fp)
    n_workers = getattr(predict_cfg, "n_workers", 1)
    if n_workers > 1:
        log.info(
            "Parallel mode: %d workers (data loaded per-worker)",
            n_workers,
        )
        predict_data = None
    else:
        predict_data = load_predict_data(predict_fp, predict_cfg.batches)

    # Run prediction
    out_fn = predict_single_trait(
        trait=args.trait,
        trait_set=args.trait_set,
        predict_data=predict_data,
        models_dir=models_dir,
        out_dir=out_dir,
        res=cfg.target_resolution,
        crs=cfg.crs,
        predict_cfg=predict_cfg,
        dask_dashboard=cfg.dask_dashboard,
        overwrite=args.overwrite,
        mode=mode,
        predict_fp=predict_fp,
    )

    log.info("Done!")
    return out_fn


if __name__ == "__main__":
    main(cli())
