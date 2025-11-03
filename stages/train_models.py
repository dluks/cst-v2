#!/usr/bin/env python
"""
Entry point script to train models either locally or on Slurm.

This script provides a unified interface to run model training for all traits, trait sets,
and CV folds in parallel either on the local machine or submit it as Slurm jobs.

Execution mode is determined by the USE_SLURM environment variable (from .env file).
Set USE_SLURM=false to use local execution by default, or use --local flag to override.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from simple_slurm import Slurm

from src.conf.conf import get_config

# Setup environment and path
from src.pipeline.entrypoint_utils import (
    PartitionDistributor,
    add_common_args,
    add_partition_args,
    add_resource_args,
    build_base_command,
    determine_execution_mode,
    resolve_partitions,
    setup_environment,
    wait_for_job_completion,
)

project_root = setup_environment()


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train models locally or on Slurm.")
    add_common_args(parser)
    add_partition_args(parser, enable_multi_partition=True)
    add_resource_args(
        parser,
        time_default="04:00:00",
        cpus_default=112,
        mem_default="128GB",
        include_gpus=False,
        gpus_default="0",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    parser.add_argument(
        "-s",
        "--sample",
        type=float,
        default=1.0,
        help="Fraction of data to sample for training (default: 1.0).",
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Resume training from last run.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for Slurm job to complete (submit and exit).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help=(
            "Maximum number of models to train in parallel (default: 1). "
            "Only applies to local execution."
        ),
    )
    parser.add_argument(
        "--traits",
        type=str,
        nargs="+",
        default=None,
        help="Specific trait(s) to train. If not specified, trains all traits.",
    )
    parser.add_argument(
        "--trait-sets",
        type=str,
        nargs="+",
        default=None,
        choices=["splot", "gbif", "splot_gbif"],
        help="Specific trait set(s) to train. If not specified, trains all trait sets.",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run model training locally or on Slurm."""
    args = cli()

    # Convert paths to absolute if provided
    params_path = str(Path(args.params).resolve()) if args.params else None

    # Set CONFIG_PATH BEFORE importing dataset_utils to avoid module-level load error
    if params_path is not None:
        os.environ["CONFIG_PATH"] = params_path

    # Get configuration
    cfg = get_config(params_path=params_path)

    # Determine execution mode
    use_local, mode = determine_execution_mode(args.local)

    # Get traits to train
    if args.traits:
        traits = args.traits
        print(f"Training specified traits: {', '.join(traits)}")
    else:
        y_fp = Path(project_root, cfg.train.Y.fp).resolve()
        traits = (
            dd.read_parquet(y_fp).columns.difference(["x", "y", "source"]).to_list()
        )
        print(f"Found {len(traits)} traits to train: {', '.join(traits)}")

    # Get trait sets to train
    trait_sets = args.trait_sets or cfg.train.trait_sets
    print(f"Training trait sets: {', '.join(trait_sets)}")

    # Get number of CV folds
    n_folds = cfg.train.cv_splits.n_splits
    print(f"Number of CV folds: {n_folds}")

    # Generate all training tasks
    tasks = []
    for trait in traits:
        for trait_set in trait_sets:
            # Add CV fold tasks
            for fold in range(n_folds):
                tasks.append(
                    {
                        "trait": trait,
                        "trait_set": trait_set,
                        "fold": fold,
                        "task_type": "cv_fold",
                    }
                )
            # Add full model task
            tasks.append(
                {
                    "trait": trait,
                    "trait_set": trait_set,
                    "fold": None,
                    "task_type": "full_model",
                }
            )

    print(f"\nTotal training tasks: {len(tasks)}")
    print(
        f"  - CV fold tasks: {len([t for t in tasks if t['task_type'] == 'cv_fold'])}"
    )
    print(
        f"  - Full model tasks: {len([t for t in tasks if t['task_type'] == 'full_model'])}"
    )
    print(f"\nExecution mode: {mode}")

    if use_local:
        # Run locally
        run_local(
            params_path,
            args.debug,
            args.sample,
            args.resume,
            args.overwrite,
            tasks,
            args.max_parallel,
            cfg,
        )
    else:
        # Determine partitions to use
        partitions = resolve_partitions(args.partition, args.partitions)
        if len(partitions) > 1:
            print(
                f"Distributing jobs across {len(partitions)} partitions: "
                f"{', '.join(partitions)}"
            )
        else:
            print(f"Using partition: {partitions[0]}")

        # Submit to Slurm
        run_slurm(
            params_path,
            args.debug,
            args.sample,
            args.resume,
            args.overwrite,
            partitions,
            args.time,
            args.cpus,
            args.mem,
            wait=not args.no_wait,
            tasks=tasks,
            cfg=cfg,
        )


def aggregate_cv_results(training_dir: Path, cfg) -> None:
    """
    Aggregate CV results from individual folds.

    Args:
        training_dir: Directory containing CV fold results
        cfg: Configuration object
    """
    cv_dir = training_dir / "cv"

    if not cv_dir.exists():
        print(f"Warning: CV directory not found: {cv_dir}")
        return

    # Check if all folds are complete
    n_folds = cfg.train.cv_splits.n_splits
    complete_folds = list(cv_dir.glob("cv_fold_*_complete.flag"))

    if len(complete_folds) < n_folds:
        print(f"Warning: Not all CV folds complete ({len(complete_folds)}/{n_folds})")
        return

    print(f"Aggregating CV results from {training_dir}...")

    # Aggregate evaluation results
    eval_results = []
    for fold_dir in sorted(cv_dir.glob("fold_*")):
        eval_file = fold_dir / cfg.train.eval_results
        if eval_file.exists():
            eval_results.append(pd.read_csv(eval_file, index_col=0))

    if eval_results:
        eval_df = (
            pd.concat(eval_results)
            .drop(columns=["fold"])
            .reset_index(names="index")
            .groupby("index")
            .agg(["mean", "std"])
        )
        eval_df.to_csv(training_dir / cfg.train.eval_results)
        print(f"  - Saved aggregated evaluation results")

    # Aggregate feature importance
    fi_results = []
    for fold_dir in sorted(cv_dir.glob("fold_*")):
        fi_file = fold_dir / cfg.train.feature_importance
        if fi_file.exists():
            fi_results.append(pd.read_csv(fi_file, index_col=0))

    if fi_results:
        fi_df = (
            pd.concat(fi_results)
            .drop(columns=["fold"])
            .reset_index(names="index")
            .groupby("index")
            .agg(["mean", "std"])
        )
        fi_df.to_csv(training_dir / cfg.train.feature_importance)
        print(f"  - Saved aggregated feature importance")

    # Mark CV as complete
    cv_complete_flag = training_dir / "cv_complete.flag"
    cv_complete_flag.touch()
    print(f"  - Marked CV as complete")


def run_single_task(
    task: dict,
    params_path: str | None,
    debug: bool,
    sample: float,
    resume: bool,
    overwrite: bool,
    product_code: str,
) -> tuple[dict, bool]:
    """
    Run a single training task.

    Args:
        task: Task dictionary with trait, trait_set, fold, and task_type
        params_path: Path to parameters file
        debug: Enable debug mode
        sample: Fraction of data to sample
        resume: Resume from last run
        overwrite: Overwrite existing models

    Returns:
        Tuple of (task, success)
    """
    trait = task["trait"]
    trait_set = task["trait_set"]
    fold = task["fold"]
    task_type = task["task_type"]

    extra_args: dict[str, str | None] = {
        "--trait": trait,
        "--trait-set": trait_set,
    }

    if fold is not None:
        extra_args["--fold"] = str(fold)

    if debug:
        extra_args["--debug"] = None

    if sample < 1.0:
        extra_args["--sample"] = str(sample)

    if resume:
        extra_args["--resume"] = None

    if overwrite:
        extra_args["--overwrite"] = None

    cmd = build_base_command(
        "src.models.autogluon",
        params_path=params_path,
        overwrite=False,  # Handled by --overwrite flag in extra_args
        extra_args=extra_args,
    )

    # Set CONFIG_PATH in subprocess environment to avoid import-time errors
    env = os.environ.copy()
    if params_path:
        env["CONFIG_PATH"] = params_path

    # Create log directory and file names
    log_dir = Path("logs/train_models") / product_code
    log_dir.mkdir(parents=True, exist_ok=True)

    task_log_name = f"{trait}_{trait_set}"
    if fold is not None:
        task_log_name += f"_fold_{fold}"
    else:
        task_log_name += "_full_model"

    log_file = log_dir / f"{task_log_name}.log"
    err_file = log_dir / f"{task_log_name}.err"

    task_name = f"{trait}/{trait_set}"
    if fold is not None:
        task_name += f"/fold_{fold}"
    else:
        task_name += "/full_model"

    print(f"  Starting: {task_name}")
    print(f"    Logs: {log_file}")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Write stdout and stderr to log files
    if result.stdout:
        log_file.write_text(result.stdout)
    if result.stderr:
        err_file.write_text(result.stderr)

    if result.returncode == 0:
        print(f"  ✓ Completed: {task_name}")
        return (task, True)
    else:
        print(f"  ✗ Failed: {task_name}")
        print(f"    Exit code: {result.returncode}")
        print(f"    Check error log: {err_file}")
        if result.stderr:
            # Print last 20 lines of stderr for better error visibility
            stderr_lines = result.stderr.strip().split("\n")
            print(f"    Last {min(20, len(stderr_lines))} lines of stderr:")
            for line in stderr_lines[-20:]:
                print(f"      {line}")
        return (task, False)


def run_local(
    params_path: str | None,
    debug: bool,
    sample: float,
    resume: bool,
    overwrite: bool,
    tasks: list[dict],
    max_parallel: int,
    cfg,
) -> None:
    """Run model training locally with parallel processing."""
    print(f"\nRunning model training locally (max {max_parallel} tasks in parallel)...")

    # Process tasks in parallel
    print(f"\nProcessing {len(tasks)} tasks with up to {max_parallel} workers...")
    successful = []
    failed = []

    product_code = cfg.product_code

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all training tasks
        futures = {
            executor.submit(
                run_single_task,
                task,
                params_path,
                debug,
                sample,
                resume,
                overwrite,
                product_code,
            ): task
            for task in tasks
        }

        # Wait for completion
        for future in as_completed(futures):
            task, success = future.result()
            if success:
                successful.append(task)
            else:
                failed.append(task)

    # Report results
    print(f"\n{'=' * 60}")
    print(f"Completed: {len(successful)}/{len(tasks)} tasks")
    if failed:
        print(f"Failed: {len(failed)} tasks")
        for task in failed:
            task_name = f"{task['trait']}/{task['trait_set']}"
            if task["fold"] is not None:
                task_name += f"/fold_{task['fold']}"
            else:
                task_name += "/full_model"
            print(f"  - {task_name}")
        print("✗ Some tasks failed to process")
        sys.exit(1)

    # Aggregate CV results for each trait/trait_set combination
    print("\nAggregating CV results...")

    trait_set_combos = set(
        (t["trait"], t["trait_set"]) for t in successful if t["task_type"] == "cv_fold"
    )

    for trait, trait_set in trait_set_combos:
        # Build path: models/{product_code}/{trait}/{arch}/{trait_set}
        trait_models_dir = Path(cfg.models.dir_fp) / trait / cfg.train.arch
        runs_dir = trait_models_dir / "debug" if debug else trait_models_dir
        training_dir = runs_dir / trait_set

        if training_dir.exists():
            aggregate_cv_results(training_dir, cfg)

    print("\n✓ Model training completed successfully!")


def run_slurm(
    params_path: str | None,
    debug: bool,
    sample: float,
    resume: bool,
    overwrite: bool,
    partitions: list[str],
    time_limit: str,
    cpus: int,
    mem: str,
    wait: bool,
    tasks: list[dict],
    cfg,
) -> None:
    """Submit separate Slurm jobs for each training task."""
    print(f"\nSubmitting {len(tasks)} Slurm jobs (one per task)...")

    # Create log directory
    product_code = cfg.product_code
    log_dir = Path("logs/train_models") / product_code
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create partition distributor for round-robin distribution
    distributor = PartitionDistributor(partitions)

    # Submit a job for each task
    job_ids = []
    task_to_job = {}

    for task in tasks:
        trait = task["trait"]
        trait_set = task["trait_set"]
        fold = task["fold"]
        task_type = task["task_type"]

        # Construct command for this task
        extra_args: dict[str, str | None] = {
            "--trait": trait,
            "--trait-set": trait_set,
        }

        if fold is not None:
            extra_args["--fold"] = str(fold)

        if debug:
            extra_args["--debug"] = None

        if sample < 1.0:
            extra_args["--sample"] = str(sample)

        if resume:
            extra_args["--resume"] = None

        if overwrite:
            extra_args["--overwrite"] = None

        cmd_parts = build_base_command(
            "src.models.autogluon",
            params_path=params_path,
            overwrite=False,  # Handled by --overwrite flag in extra_args
            extra_args=extra_args,
        )
        command = " ".join(cmd_parts)

        # Create job name
        job_name_parts = [f"train", trait[:8], trait_set[:4]]
        if fold is not None:
            job_name_parts.append(f"f{fold}")
        else:
            job_name_parts.append("full")
        job_name = "_".join(job_name_parts)

        # Create log file names
        task_name = f"{trait}_{trait_set}"
        if fold is not None:
            task_name += f"_fold_{fold}"
        else:
            task_name += "_full_model"

        # Get partition using round-robin distribution
        partition = distributor.get_next()

        # Create Slurm job configuration
        slurm = Slurm(
            job_name=job_name,
            output=str(log_dir / f"%j_{task_name}.log"),
            error=str(log_dir / f"%j_{task_name}.err"),
            time=time_limit,
            cpus_per_task=cpus,
            mem=mem,
            partition=partition,
        )

        # Submit the job
        job_id = slurm.sbatch(command)
        job_ids.append(job_id)
        task_to_job[job_id] = task

        task_desc = f"{trait}/{trait_set}"
        if fold is not None:
            task_desc += f"/fold_{fold}"
        else:
            task_desc += "/full_model"

        # Show partition in output if using multiple partitions
        partition_info = f" [{partition}]" if len(distributor) > 1 else ""
        print(f"  Submitted job {job_id} for: {task_desc}{partition_info}")

    print(f"\nSubmitted {len(job_ids)} jobs")

    # Show distribution summary if using multiple partitions
    if len(distributor) > 1:
        summary = distributor.get_summary()
        print("Job distribution across partitions:")
        for partition, count in summary.items():
            print(f"  {partition}: {count} jobs")

    print(f"Logs directory: {log_dir.absolute()}")

    if wait:
        # Wait for all jobs to complete
        print(f"\nWaiting for {len(job_ids)} jobs to complete...")
        successful = []
        failed = []

        for job_id in job_ids:
            task = task_to_job[job_id]
            task_name = f"{task['trait']}/{task['trait_set']}"
            if task["fold"] is not None:
                task_name += f"/fold_{task['fold']}"
            else:
                task_name += "/full_model"

            success = wait_for_job_completion(job_id, poll_interval=5)

            if success:
                successful.append(task)
            else:
                failed.append(task)
                print(f"✗ Job {job_id} ({task_name}) failed. Check logs:")
                print(
                    f"  {log_dir.absolute()}/{job_id}_{task_name.replace('/', '_')}.err"
                )

        # Report results
        print(f"\n{'=' * 60}")
        print(f"Completed: {len(successful)}/{len(tasks)} tasks")
        if failed:
            print(f"Failed: {len(failed)} tasks")
            for task in failed:
                t_name = f"{task['trait']}/{task['trait_set']}"
                if task["fold"] is not None:
                    t_name += f"/fold_{task['fold']}"
                else:
                    t_name += "/full_model"
                print(f"  - {t_name}")
            print("✗ Some jobs failed")
            sys.exit(1)

        # Aggregate CV results for each trait/trait_set combination
        print("\nAggregating CV results...")

        trait_set_combos = set(
            (t["trait"], t["trait_set"])
            for t in successful
            if t["task_type"] == "cv_fold"
        )

        for trait, trait_set in trait_set_combos:
            # Build path: models/{product_code}/{trait}/{arch}/{trait_set}
            trait_models_dir = Path(cfg.models.dir_fp) / trait / cfg.train.arch
            runs_dir = trait_models_dir / "debug" if debug else trait_models_dir
            training_dir = runs_dir / trait_set

            if training_dir.exists():
                aggregate_cv_results(training_dir, cfg)

        print("\n✓ Model training completed successfully!")
    else:
        print("\nJobs submitted. Not waiting for completion (--no-wait flag set).")
        print(
            "To check status and aggregate results later, run this script again with --wait flag."
        )


if __name__ == "__main__":
    main()
