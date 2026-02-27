"""Predict a single trait using trained models.

This module handles prediction for a single trait-trait_set combination,
supporting both standard prediction and Coefficient of Variation (CoV) calculation.

Supports three execution modes:
- Standard: Load all data, predict, rasterize (single process)
- Chunk: Load a subset of parquet row groups, predict, save as parquet
- Merge: Load chunk parquets, concatenate, rasterize to tif
"""

import argparse
import shutil
import time
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
        "--params",
        type=str,
        default=None,
        help="Path to params.yaml file (default: uses active config)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode"
    )
    # Chunk/merge mode arguments (for Slurm-level parallelism)
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="Chunk index (0-based) for chunked prediction",
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=None,
        help="Total number of chunks for chunked prediction",
    )
    parser.add_argument(
        "--merge-chunks",
        action="store_true",
        help="Merge chunk parquet results and rasterize to tif",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="CV fold index (0-based) for chunked CoV prediction",
    )
    return parser.parse_args()


# ============================================================
# Data loading
# ============================================================


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
        raise FileNotFoundError(
            f"Predict data not found: {predict_fp}"
        )

    # Reset index to convert x/y from index to columns
    if batches == 1:
        return pd.read_parquet(predict_fp).reset_index()
    else:
        return dd.read_parquet(predict_fp).reset_index().repartition(npartitions=batches)


def load_predict_data_chunk(
    predict_fp: Path, chunk_index: int, n_chunks: int
) -> pd.DataFrame:
    """Load a chunk of predict data from specific parquet row groups.

    Distributes row groups evenly across chunks and loads only this
    chunk's row groups, keeping memory usage proportional to 1/n_chunks.

    Args:
        predict_fp: Path to predict features parquet file
        chunk_index: 0-based index of this chunk
        n_chunks: Total number of chunks

    Returns:
        DataFrame with features and x, y columns
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(predict_fp))
    n_row_groups = pf.metadata.num_row_groups

    # Distribute row groups across chunks
    all_indices = np.array_split(range(n_row_groups), n_chunks)
    my_indices = list(all_indices[chunk_index])

    log.info(
        "Chunk %d/%d: loading %d row groups (of %d total)...",
        chunk_index,
        n_chunks,
        len(my_indices),
        n_row_groups,
    )

    t0 = time.time()
    tables = [pf.read_row_group(i) for i in my_indices]
    data = pa.concat_tables(tables).to_pandas()

    # Ensure x/y are columns (they may be stored as the parquet index)
    if "x" not in data.columns or "y" not in data.columns:
        data = data.reset_index()

    log.info(
        "Loaded %s rows for chunk %d in %.1fs",
        f"{len(data):,}",
        chunk_index,
        time.time() - t0,
    )
    return data


# ============================================================
# Model path resolution
# ============================================================


def find_model_path(models_dir: Path, trait: str, trait_set: str) -> Path:
    """Find the model directory for a trait/trait_set combination.

    Returns the trait_set directory containing full_model and cv subdirs.

    Args:
        models_dir: Base directory containing trained models
        trait: Trait name
        trait_set: Trait set name

    Returns:
        Path to the trait_set directory

    Raises:
        FileNotFoundError: If model directory not found
    """
    trait_dir = models_dir / trait
    if not trait_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {trait_dir}")

    autogluon_dir = trait_dir / "autogluon"
    if not autogluon_dir.exists():
        raise FileNotFoundError(f"AutoGluon directory not found: {autogluon_dir}")

    run_dirs = [
        d
        for d in autogluon_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in: {autogluon_dir}")
    latest_run = max(run_dirs, key=lambda d: d.name)

    trait_set_dir = latest_run / trait_set
    if not trait_set_dir.exists():
        raise FileNotFoundError(f"Trait set directory not found: {trait_set_dir}")

    return trait_set_dir


# ============================================================
# Prediction functions
# ============================================================


def predict_trait_ag(
    data: pd.DataFrame | dd.DataFrame,
    model_path: Path,
) -> pd.DataFrame:
    """Predict using AutoGluon model.

    Loads the AutoGluon model once and runs prediction on the entire dataset.

    Args:
        data: DataFrame with features and x, y coordinates
        model_path: Path to the full_model directory

    Returns:
        DataFrame with predictions indexed by (y, x)
    """
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


def predict_cov(
    predict_data: pd.DataFrame | dd.DataFrame,
    cv_dir: Path,
    tmp_dir: Path,
) -> pd.DataFrame:
    """Calculate the Coefficient of Variation using CV fold models.

    Loads each CV fold model once and predicts on the entire dataset.

    Args:
        predict_data: DataFrame with features and x, y coordinates
        cv_dir: Path to directory containing CV fold models
        tmp_dir: Directory to store intermediate fold predictions

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
        pred = predict_trait_ag(predict_data, fold_model_path)
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


def predict_fn(
    predict_data: pd.DataFrame | dd.DataFrame,
    model_path: Path,
    cov: bool,
    tmp_dir: Path | None,
) -> tuple[pd.DataFrame, Path | None]:
    """Predict the trait using the given model, with optional CoV calculation.

    Args:
        predict_data: DataFrame with features and x, y coordinates
        model_path: Path to the trait_set directory
            (containing full_model and cv subdirs)
        cov: Whether to calculate CoV instead of standard prediction
        tmp_dir: Directory for temporary files (used for CoV calculation)

    Returns:
        Tuple of (predictions DataFrame, temp directory path or None)
    """
    if cov:
        if tmp_dir is None:
            raise ValueError("tmp_dir must be provided for CoV calculation")
        cv_dir = model_path / "cv"
        return (
            predict_cov(predict_data, cv_dir, tmp_dir),
            tmp_dir,
        )
    full_model = model_path / "full_model"
    return (
        predict_trait_ag(predict_data, full_model),
        None,
    )


# ============================================================
# Chunk and merge modes
# ============================================================


def predict_and_save_chunk(args: argparse.Namespace, cfg: ConfigBox) -> Path:
    """Load a data chunk, predict, and save results as parquet.

    Each chunk job loads a subset of parquet row groups, runs prediction
    in a single process, and saves the result as a parquet file in a
    chunks subdirectory.

    Args:
        args: CLI args (must include chunk_index, n_chunks, trait, trait_set)
        cfg: Configuration

    Returns:
        Path to chunk parquet file
    """
    predict_fp = Path(cfg.train.predict.fp)
    models_dir = Path(cfg.models.dir_fp)
    out_dir = get_predict_dir(cfg)

    # Skip if chunk parquet already exists
    chunks_dir = out_dir / args.trait / args.trait_set / "chunks"
    chunk_fp = chunks_dir / f"chunk_{args.chunk_index:03d}.parquet"
    if chunk_fp.exists():
        log.info(
            "Chunk %d already exists at %s, skipping",
            args.chunk_index,
            chunk_fp,
        )
        return chunk_fp

    # Load chunk data
    data = load_predict_data_chunk(predict_fp, args.chunk_index, args.n_chunks)

    # Find model
    trait_set_dir = find_model_path(models_dir, args.trait, args.trait_set)
    model_path = trait_set_dir / "full_model"
    if not model_path.exists():
        raise ValueError(f"full_model directory not found: {model_path}")

    # Load model and predict
    log.info("Loading AutoGluon predictor from %s...", model_path)
    predictor = TabularPredictor.load(str(model_path))

    coords = data[["x", "y"]]
    features = data.drop(columns=["x", "y"])

    # Predict in sub-batches for progress reporting
    t0 = time.time()
    n_rows = len(features)
    sub_batch_size = 500_000
    if n_rows > sub_batch_size:
        predictions_list = []
        for start in range(0, n_rows, sub_batch_size):
            end = min(start + sub_batch_size, n_rows)
            batch_pred = predictor.predict(
                features.iloc[start:end], as_pandas=True
            )
            predictions_list.append(batch_pred)
            elapsed = time.time() - t0
            rows_per_s = end / elapsed if elapsed > 0 else 0
            log.info(
                "Chunk %d: %s / %s rows predicted (%.1fs elapsed, %.0f rows/s)",
                args.chunk_index,
                f"{end:,}",
                f"{n_rows:,}",
                elapsed,
                rows_per_s,
            )
        predictions = pd.concat(predictions_list, ignore_index=True)
    else:
        predictions = predictor.predict(features, as_pandas=True)

    t_total = time.time() - t0
    log.info(
        "Chunk %d prediction complete in %.1fs (%.0f rows/s)",
        args.chunk_index,
        t_total,
        n_rows / t_total if t_total > 0 else 0,
    )

    # Build result with coords
    result = pd.concat(
        [coords.reset_index(drop=True), predictions.reset_index(drop=True)],
        axis=1,
    ).set_index(["y", "x"])

    # Save to chunks directory
    chunks_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(chunk_fp)
    log.info("Chunk %d saved to %s", args.chunk_index, chunk_fp)

    return chunk_fp


def merge_and_rasterize(args: argparse.Namespace, cfg: ConfigBox) -> Path:
    """Load chunk parquets, concatenate, rasterize, and save as tif.

    This is the merge step that runs after all chunk jobs complete.
    It reads all chunk parquet files, concatenates them, rasterizes
    to a GeoTIFF, and cleans up the chunk files.

    Args:
        args: CLI args (must include trait, trait_set)
        cfg: Configuration

    Returns:
        Path to output tif file
    """
    out_dir = get_predict_dir(cfg)
    chunks_dir = out_dir / args.trait / args.trait_set / "chunks"

    # Find and load all chunk parquets
    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk parquets found in {chunks_dir}")

    log.info("Merging %d chunk files from %s...", len(chunk_files), chunks_dir)
    t0 = time.time()
    dfs = [pd.read_parquet(f) for f in chunk_files]
    pred = pd.concat(dfs)
    log.info("Merged %s rows in %.1fs", f"{len(pred):,}", time.time() - t0)

    # Set up output path
    out_fn = (
        out_dir
        / args.trait
        / args.trait_set
        / f"{args.trait}_{args.trait_set}_predict.tif"
    )
    out_fn.parent.mkdir(parents=True, exist_ok=True)

    if not args.overwrite and out_fn.exists():
        log.info("Output file already exists: %s", out_fn)
        return out_fn

    log.info("Rasterizing predictions...")
    pred_r = rasterize_points(
        pred, data_cols=args.trait, res=cfg.target_resolution, crs=cfg.crs
    )
    pred_r = pack_xr(pred_r)
    xr_to_raster(pred_r, out_fn)
    log.info("Raster saved to %s", out_fn)

    # Clean up chunks
    log.info("Cleaning up chunk files...")
    shutil.rmtree(chunks_dir)

    log.info("Merge complete: %s", out_fn)
    return out_fn


def cov_chunk_predict(args: argparse.Namespace, cfg: ConfigBox) -> Path:
    """Load a data chunk, predict with a single CV fold model, and save as parquet.

    Each job predicts one chunk of data using one CV fold model.
    Output: {cov_dir}/{trait}/{trait_set}/chunks/fold_{F:02d}_chunk_{N:03d}.parquet

    Args:
        args: CLI args (must include chunk_index, n_chunks, fold_index, trait, trait_set)
        cfg: Configuration

    Returns:
        Path to chunk parquet file
    """
    predict_fp = Path(cfg.train.predict.fp)
    models_dir = Path(cfg.models.dir_fp)
    out_dir = get_cov_dir(cfg)

    # Skip if chunk parquet already exists
    chunks_dir = out_dir / args.trait / args.trait_set / "chunks"
    chunk_fp = chunks_dir / f"fold_{args.fold_index:02d}_chunk_{args.chunk_index:03d}.parquet"
    if chunk_fp.exists():
        log.info(
            "CoV chunk (fold %d, chunk %d) already exists at %s, skipping",
            args.fold_index,
            args.chunk_index,
            chunk_fp,
        )
        return chunk_fp

    # Load chunk data
    data = load_predict_data_chunk(predict_fp, args.chunk_index, args.n_chunks)

    # Find CV fold model
    trait_set_dir = find_model_path(models_dir, args.trait, args.trait_set)
    cv_dir = trait_set_dir / "cv"
    if not cv_dir.exists():
        raise ValueError(f"cv directory not found: {cv_dir}")

    # Find the fold directory (fold_0, fold_1, ...)
    fold_dir = cv_dir / f"fold_{args.fold_index}"
    if not fold_dir.exists():
        raise ValueError(f"CV fold directory not found: {fold_dir}")

    # Load model and predict
    log.info(
        "Loading AutoGluon predictor for fold %d from %s...",
        args.fold_index,
        fold_dir,
    )
    predictor = TabularPredictor.load(str(fold_dir))

    coords = data[["x", "y"]]
    features = data.drop(columns=["x", "y"])

    # Predict in sub-batches for progress reporting
    t0 = time.time()
    n_rows = len(features)
    sub_batch_size = 500_000
    if n_rows > sub_batch_size:
        predictions_list = []
        for start in range(0, n_rows, sub_batch_size):
            end = min(start + sub_batch_size, n_rows)
            batch_pred = predictor.predict(
                features.iloc[start:end], as_pandas=True
            )
            predictions_list.append(batch_pred)
            elapsed = time.time() - t0
            rows_per_s = end / elapsed if elapsed > 0 else 0
            log.info(
                "CoV fold %d chunk %d: %s / %s rows "
                "(%.1fs elapsed, %.0f rows/s)",
                args.fold_index,
                args.chunk_index,
                f"{end:,}",
                f"{n_rows:,}",
                elapsed,
                rows_per_s,
            )
        predictions = pd.concat(predictions_list, ignore_index=True)
    else:
        predictions = predictor.predict(features, as_pandas=True)

    t_total = time.time() - t0
    log.info(
        "CoV fold %d chunk %d prediction complete in %.1fs (%.0f rows/s)",
        args.fold_index,
        args.chunk_index,
        t_total,
        n_rows / t_total if t_total > 0 else 0,
    )

    # Build result with coords
    result = pd.concat(
        [coords.reset_index(drop=True), predictions.reset_index(drop=True)],
        axis=1,
    ).set_index(["y", "x"])

    # Save to chunks directory
    chunks_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(chunk_fp)
    log.info(
        "CoV fold %d chunk %d saved to %s",
        args.fold_index,
        args.chunk_index,
        chunk_fp,
    )

    return chunk_fp


def cov_merge_and_rasterize(args: argparse.Namespace, cfg: ConfigBox) -> Path:
    """Load CoV chunk parquets, compute CoV across folds, rasterize, and save as tif.

    Groups chunk files by fold, concatenates each fold's chunks into a full
    dataset prediction, then computes CoV (std/mean with value shifting)
    across all folds.

    Args:
        args: CLI args (must include trait, trait_set)
        cfg: Configuration

    Returns:
        Path to output tif file
    """
    out_dir = get_cov_dir(cfg)
    chunks_dir = out_dir / args.trait / args.trait_set / "chunks"

    # Find all fold×chunk parquets
    chunk_files = sorted(chunks_dir.glob("fold_*_chunk_*.parquet"))
    if not chunk_files:
        raise FileNotFoundError(f"No CoV chunk parquets found in {chunks_dir}")

    log.info("Found %d CoV chunk files in %s", len(chunk_files), chunks_dir)

    # Group by fold index
    from collections import defaultdict

    fold_chunks: dict[int, list[Path]] = defaultdict(list)
    for fp in chunk_files:
        # Parse fold index from filename: fold_XX_chunk_YYY.parquet
        parts = fp.stem.split("_")
        fold_idx = int(parts[1])
        fold_chunks[fold_idx].append(fp)

    n_folds = len(fold_chunks)
    log.info("Found %d folds with chunks", n_folds)

    # For each fold, concatenate all its chunks into one prediction series
    t0 = time.time()
    fold_predictions = []
    for fold_idx in sorted(fold_chunks.keys()):
        fold_files = sorted(fold_chunks[fold_idx])
        log.info("Loading fold %d: %d chunk files...", fold_idx, len(fold_files))
        dfs = [pd.read_parquet(f) for f in fold_files]
        fold_pred = pd.concat(dfs)
        # Rename column to fold index to avoid duplicate column names
        fold_pred.columns = [f"fold_{fold_idx}"]
        fold_predictions.append(fold_pred)

    log.info("Concatenating %d fold predictions...", n_folds)
    all_preds = pd.concat(fold_predictions, axis=1)
    log.info(
        "Merged %s rows x %d folds in %.1fs",
        f"{len(all_preds):,}",
        n_folds,
        time.time() - t0,
    )

    # Compute CoV: shift values, then std / mean
    log.info("Calculating CoV...")
    cov = (
        all_preds
        .pipe(lambda _df: _df + abs(_df.min().min()))
        .pipe(lambda _df: _df.std(axis=1) / _df.mean(axis=1))
        .rename("cov")
        .to_frame()
    )

    # Set up output path
    out_fn = (
        out_dir
        / args.trait
        / args.trait_set
        / f"{args.trait}_{args.trait_set}_cov.tif"
    )
    out_fn.parent.mkdir(parents=True, exist_ok=True)

    if not args.overwrite and out_fn.exists():
        log.info("Output file already exists: %s", out_fn)
        return out_fn

    log.info("Rasterizing CoV...")
    cov_r = rasterize_points(
        cov, data_cols="cov", res=cfg.target_resolution, crs=cfg.crs
    )
    cov_r = pack_xr(cov_r)
    xr_to_raster(cov_r, out_fn)
    log.info("CoV raster saved to %s", out_fn)

    # Clean up chunks
    log.info("Cleaning up chunk files...")
    shutil.rmtree(chunks_dir)

    log.info("CoV merge complete: %s", out_fn)
    return out_fn


# ============================================================
# Standard prediction pipeline
# ============================================================


def predict_single_trait(
    trait: str,
    trait_set: str,
    predict_data: pd.DataFrame | dd.DataFrame,
    models_dir: Path,
    out_dir: Path,
    res: int | float,
    crs: str,
    predict_cfg: ConfigBox,
    dask_dashboard: str,
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
        dask_dashboard: Dask dashboard address
        overwrite: Whether to overwrite existing output
        mode: Either "predict" or "cov" for CoV calculation

    Returns:
        Path to output file

    Raises:
        FileNotFoundError: If model directory not found
        ValueError: If model structure is invalid
    """
    cov: bool = mode == "cov"

    # Find model directory
    trait_set_dir = find_model_path(models_dir, trait, trait_set)

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
        out_dir
        / trait
        / trait_set
        / f"{trait}_{trait_set}_{'cov' if cov else 'predict'}.tif"
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
    pred, tmp_dir = predict_fn(predict_data, trait_set_dir, cov, tmp_dir)

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


# ============================================================
# Main entry point
# ============================================================


def main(args: argparse.Namespace, cfg: ConfigBox | None = None) -> Path:
    """Main function to predict a single trait.

    Dispatches to one of five modes:
    - CoV merge (--cov --merge-chunks): merge CoV chunk parquets, compute CoV, rasterize
    - CoV chunk (--cov --chunk-index --n-chunks --fold-index): predict with CV fold
    - Predict merge (--merge-chunks): merge predict chunk parquets and rasterize
    - Predict chunk (--chunk-index + --n-chunks): predict on data subset, save parquet
    - Standard mode: load all data, predict, rasterize (original behavior)

    Args:
        args: Command-line arguments
        cfg: Configuration (if None, loads from get_config())

    Returns:
        Path to output file
    """
    if cfg is None:
        cfg = get_config(params_path=getattr(args, "params", None))

    if not args.verbose:
        log.setLevel("WARNING")

    # Dispatch: CoV merge mode
    if args.cov and args.merge_chunks:
        return cov_merge_and_rasterize(args, cfg)

    # Dispatch: CoV chunk mode
    if args.cov and args.chunk_index is not None and args.fold_index is not None:
        return cov_chunk_predict(args, cfg)

    # Dispatch: predict merge mode
    if args.merge_chunks:
        return merge_and_rasterize(args, cfg)

    # Dispatch: predict chunk mode
    if args.chunk_index is not None and args.n_chunks is not None:
        return predict_and_save_chunk(args, cfg)

    # Standard mode (original behavior)
    predict_cfg = cfg.predict[detect_system()]
    if args.batches is not None:
        predict_cfg.batches = args.batches

    models_dir = Path(cfg.models.dir_fp)
    mode: Literal["predict", "cov"] = "cov" if args.cov else "predict"
    out_dir = get_cov_dir(cfg) if args.cov else get_predict_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    predict_data = load_predict_data(Path(cfg.train.predict.fp), predict_cfg.batches)

    return predict_single_trait(
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
    )


if __name__ == "__main__":
    main(cli())
