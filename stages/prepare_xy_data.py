#!/usr/bin/env python
"""
Prepare XY data for all traits before model training.

This script pre-generates the merged feature and label data for all traits,
saving them to a single parquet file so that parallel training jobs can read
and filter it without race conditions.

Execution mode is determined by the USE_SLURM environment variable (from .env file).
Set USE_SLURM=false to use local execution by default, or use --local flag to override.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import dask.dataframe as dd
from simple_slurm import Slurm

from src.conf.conf import get_config
from src.pipeline.entrypoint_utils import (
    add_common_args,
    add_resource_args,
    build_base_command,
    determine_execution_mode,
    get_existing_job_names,
    setup_environment,
    submit_job_with_retry,
    wait_for_job_completion,
)

project_root = setup_environment()


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare XY data for all traits locally or on Slurm."
    )
    add_common_args(parser)
    add_resource_args(
        parser,
        time_default="02:00:00",
        cpus_default=64,
        mem_default="128GB",
        include_gpus=False,
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for Slurm job to complete (submit and exit).",
    )
    return parser.parse_args()


def prepare_xy_data(cfg) -> None:
    """
    Prepare XY data for all traits.

    Creates a single merged dataframe with all features, labels, CV splits,
    and source information that can be filtered during training.

    Args:
        cfg: Configuration object
    """
    # Import here after CONFIG_PATH is set
    from src.conf.environment import activate_env, log
    from src.utils.df_utils import pipe_log
    from src.utils.log_utils import suppress_dask_logging

    activate_env()
    suppress_dask_logging()

    # Prepare output directory
    output_path = Path(cfg.train.xy_data.fp).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output file: {output_path.absolute()}")

    # Load features
    print("\nLoading features...")
    feats = dd.read_parquet(Path(cfg.train.predict.fp)).reset_index()

    # Load labels (all traits)
    print("Loading labels...")
    labels = dd.read_parquet(Path(cfg.train.Y.fp))

    # Load CV splits for all traits
    print("Loading CV splits...")
    cv_splits_dir = Path(cfg.train.cv_splits.dir_fp)

    # Get all trait columns (excluding x, y, source, and reliability columns)
    all_cols = labels.columns.difference(["x", "y", "source"]).to_list()
    trait_cols = [col for col in all_cols if not col.endswith("_reliability")]
    print(f"Found {len(trait_cols)} traits: {', '.join(trait_cols)}")

    # Check for reliability columns
    reliability_cols = [col for col in all_cols if col.endswith("_reliability")]
    if reliability_cols:
        print(
            f"Found {len(reliability_cols)} reliability weight columns: {', '.join(reliability_cols)}"
        )

    # Read all CV split files and concatenate
    print("Loading CV split assignments...")
    split_dfs = []
    for trait in trait_cols:
        split_file = cv_splits_dir / f"{trait}.parquet"
        if split_file.exists():
            df = dd.read_parquet(split_file)
            # Rename fold column to include trait name to avoid conflicts
            df = df.rename(
                columns={
                    col: f"{trait}_fold" for col in df.columns if col not in ["x", "y"]
                }
            )
            split_dfs.append(df)

    # Merge all split assignments
    log.info("Merging CV split assignments...")
    splits = split_dfs[0]
    for split_df in split_dfs[1:]:
        splits = splits.merge(
            split_df,
            on=["x", "y"],
            how="outer",
        )

    splits = splits.compute().set_index(["y", "x"])

    # Merge labels with splits
    log.info("Merging labels with CV splits...")
    labels_with_splits = (
        labels.compute()
        .set_index(["y", "x"])
        .merge(splits, validate="m:1", right_index=True, left_index=True)
    )

    # Merge features with labels
    log.info("Merging features with labels...")
    xy_all = (
        feats.compute()
        .set_index(["y", "x"])
        .pipe(pipe_log, "Final merge...")
        .merge(labels_with_splits, validate="1:m", right_index=True, left_index=True)
        .reset_index()
    )

    # Create a 10% holdout test set from sPlot records only
    log.info("Creating 10% holdout test set from sPlot records only...")
    import numpy as np

    np.random.seed(cfg.random_seed)

    # Initialize is_test column (all False by default)
    xy_all["is_test"] = False

    # Get indices of sPlot records only (source == "s")
    splot_mask = xy_all["source"] == "s"
    splot_indices = xy_all.index[splot_mask].to_numpy()
    n_splot_samples = len(splot_indices)

    if n_splot_samples == 0:
        log.warning("No sPlot records found, skipping test set creation")
    else:
        # Select 10% of sPlot records for test set
        test_set_size = int(n_splot_samples * 0.1)
        test_splot_indices = np.random.choice(
            splot_indices, size=test_set_size, replace=False
        )

        # Mark selected sPlot indices as test set
        xy_all.loc[test_splot_indices, "is_test"] = True

        n_total_samples = len(xy_all)
        log.info(
            f"Created test set: {test_set_size} sPlot samples "
            f"({test_set_size / n_splot_samples * 100:.1f}% of sPlot, "
            f"{test_set_size / n_total_samples * 100:.1f}% of total)"
        )
        log.info(
            f"Train/CV set: {n_total_samples - test_set_size} samples "
            f"({(n_total_samples - test_set_size) / n_total_samples * 100:.1f}% of total)"
        )

    # Save to parquet (single file, not partitioned)
    log.info(f"Saving merged XY data to {output_path}...")

    # Remove old directory if it exists (from previous Dask partitioned format)
    if output_path.exists() and output_path.is_dir():
        import shutil

        log.info(f"Removing old partitioned directory: {output_path}")
        shutil.rmtree(output_path)

    xy_all.to_parquet(output_path, compression="zstd", index=False, engine="pyarrow")

    # Calculate final statistics
    n_total = len(xy_all)
    n_test = xy_all["is_test"].sum()
    n_train_cv = n_total - n_test

    print(f"\n{'=' * 60}")
    print("✓ Successfully prepared XY data for all traits!")
    print(f"  Output: {output_path}")
    print(f"  Shape: {xy_all.shape}")
    print(f"  Traits: {len(trait_cols)}")
    print(f"  Test set (sPlot only): {n_test} samples")
    print(f"  Train/CV set: {n_train_cv} samples")


def run_local(
    params_path: str | None,
    cfg,
) -> None:
    """Run XY data preparation locally."""
    print("\nRunning XY data preparation locally...")

    try:
        prepare_xy_data(cfg)
    except Exception as e:
        print(f"\n✗ Failed to prepare XY data: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def run_slurm(
    params_path: str | None,
    partition: str,
    time_limit: str,
    cpus: int,
    mem: str,
    wait: bool,
    cfg,
) -> None:
    """Submit a single Slurm job to prepare all XY data."""
    print("\nSubmitting Slurm job to prepare XY data...")

    # Create log directory
    product_code = cfg.product_code
    log_dir = Path("logs/prepare_xy_data") / product_code
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create job name
    job_name = "prep_xy_data"

    # Check if job already exists in queue
    print("Checking for existing jobs in queue...")
    existing_jobs = get_existing_job_names()
    
    if job_name in existing_jobs:
        existing_job_id, existing_state = existing_jobs[job_name]
        print(
            f"Job '{job_name}' already in queue "
            f"(job {existing_job_id}, state {existing_state})"
        )
        
        if wait:
            # Wait for the existing job to complete
            print(f"\nWaiting for existing job {existing_job_id} to complete...")
            success = wait_for_job_completion(existing_job_id)

            if success:
                print("\n✓ XY data preparation completed successfully!")
            else:
                print(f"✗ Job {existing_job_id} failed. Check logs:")
                print(f"  {log_dir.absolute()}/{existing_job_id}_prepare_xy.err")
                sys.exit(1)
        else:
            print(f"\nJob already running. To check status: squeue -j {existing_job_id}")
        
        return

    # Construct command
    cmd_parts = build_base_command(
        "stages.prepare_xy_data",
        params_path=params_path,
        overwrite=False,
        extra_args={},
    )
    # Add --local flag to force local execution in the Slurm job
    cmd_parts.append("--local")
    command = " ".join(cmd_parts)

    # Create Slurm job configuration
    slurm = Slurm(
        job_name=job_name,
        output=str(log_dir / "%j_prepare_xy.log"),
        error=str(log_dir / "%j_prepare_xy.err"),
        time=time_limit,
        cpus_per_task=cpus,
        mem=mem,
        partition=partition,
    )

    # Submit the job with retry logic for temporary Slurm failures
    job_id = submit_job_with_retry(slurm, command, max_retries=5)
    print(f"  Submitted job {job_id}")
    print(f"Logs directory: {log_dir.absolute()}")

    if wait:
        # Wait for job to complete
        print(f"\nWaiting for job {job_id} to complete...")
        success = wait_for_job_completion(job_id)

        if success:
            print("\n✓ XY data preparation completed successfully!")
        else:
            print(f"✗ Job {job_id} failed. Check logs:")
            print(f"  {log_dir.absolute()}/{job_id}_prepare_xy.err")
            sys.exit(1)
    else:
        print("\nJob submitted. Not waiting for completion (--no-wait flag set).")
        print(f"To check status: squeue -j {job_id}")


def main() -> None:
    """Main function to prepare XY data locally or on Slurm."""
    args = cli()

    # Convert paths to absolute if provided
    params_path = str(Path(args.params).resolve()) if args.params else None

    # Set CONFIG_PATH BEFORE importing to avoid module-level load error
    if params_path is not None:
        os.environ["CONFIG_PATH"] = params_path

    # Get configuration
    cfg = get_config(params_path=params_path)

    # Determine execution mode
    use_local, mode = determine_execution_mode(args.local)

    print(f"Execution mode: {mode}")

    if use_local:
        # Run locally
        run_local(params_path, cfg)
    else:
        # Submit to Slurm
        run_slurm(
            params_path,
            args.partition,
            args.time,
            args.cpus,
            args.mem,
            wait=not args.no_wait,
            cfg=cfg,
        )


if __name__ == "__main__":
    main()
