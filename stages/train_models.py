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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dask.dataframe as dd
from simple_slurm import Slurm

from src.conf.conf import get_config
from src.models.run_utils import generate_run_id, get_latest_run_id

# Setup environment and path
from src.pipeline.entrypoint_utils import (
    PartitionDistributor,
    add_common_args,
    add_partition_args,
    add_resource_args,
    add_retry_args,
    build_base_command,
    check_job_exists_by_name,
    determine_execution_mode,
    get_existing_job_names,
    is_node_failure,
    resolve_partitions,
    setup_environment,
    wait_for_job_completion,
)

project_root = setup_environment()


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train models locally or on Slurm.")
    add_common_args(parser, include_partition=False)
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
    parser.add_argument(
        "--cv-only",
        action="store_true",
        help="Train only CV fold models (excludes full models).",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Train only full models (excludes CV fold models).",
    )

    # Add retry arguments
    add_retry_args(parser, max_retries_default=2)

    return parser.parse_args()


def main() -> None:
    """Main function to run model training locally or on Slurm."""
    args = cli()

    # Validate mutually exclusive flags
    if args.cv_only and args.full_only:
        print("Error: --cv-only and --full-only cannot be used together")
        sys.exit(1)

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
        all_cols = (
            dd.read_parquet(y_fp).columns.difference(["x", "y", "source"]).to_list()
        )
        # Filter out reliability columns
        traits = [t for t in all_cols if not t.endswith("_reliability")]
        print(f"Found {len(traits)} traits to train: {', '.join(traits)}")

    # Handle debug mode
    if args.debug:
        print("\n⚠️  DEBUG MODE: Training only 1 trait")
        traits = [traits[0]]
        print(f"Debug trait: {traits[0]}")

    # Get trait sets to train
    trait_sets = args.trait_sets or cfg.train.trait_sets
    print(f"Training trait sets: {', '.join(trait_sets)}")

    # Get number of CV folds
    n_folds = cfg.train.cv_splits.n_splits
    print(f"Number of CV folds: {n_folds}")

    # Determine training mode
    if args.cv_only:
        print("\n⚠️  Training mode: CV-only (excluding full models)")
    elif args.full_only:
        print("\n⚠️  Training mode: Full-only (excluding CV fold models)")
    else:
        print("\nTraining mode: All models (CV folds + full models)")

    # Generate all training tasks
    tasks = []
    for trait in traits:
        for trait_set in trait_sets:
            # Add CV fold tasks (unless --full-only is specified)
            if not args.full_only:
                for fold in range(n_folds):
                    tasks.append(
                        {
                            "trait": trait,
                            "trait_set": trait_set,
                            "fold": fold,
                            "task_type": "cv_fold",
                        }
                    )
                # Add CV stats task (depends on all CV fold tasks)
                tasks.append(
                    {
                        "trait": trait,
                        "trait_set": trait_set,
                        "fold": None,
                        "task_type": "cv_stats",
                    }
                )
            # Add full model task (unless --cv-only is specified)
            if not args.cv_only:
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
    print(
        f"  - CV stats tasks: {len([t for t in tasks if t['task_type'] == 'cv_stats'])}"
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
            max_retries=args.max_retries,
        )




def run_single_task(
    task: dict,
    params_path: str | None,
    debug: bool,
    sample: float,
    resume: bool,
    overwrite: bool,
    product_code: str,
    run_id: str | None = None,
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
        product_code: Product code for log directory
        run_id: Run ID for this training session

    Returns:
        Tuple of (task, success)
    """
    trait = task["trait"]
    trait_set = task["trait_set"]
    fold = task["fold"]
    task_type = task["task_type"]

    # Handle cv_stats task differently
    if task_type == "cv_stats":
        extra_args: dict[str, str | None] = {
            "--trait": trait,
            "--trait-set": trait_set,
        }

        if run_id is not None:
            extra_args["--run-id"] = run_id

        if overwrite:
            extra_args["--overwrite"] = None

        cmd = build_base_command(
            "src.models.calculate_cv_stats",
            params_path=params_path,
            overwrite=False,
            extra_args=extra_args,
        )
    else:
        # Handle cv_fold and full_model tasks
        extra_args: dict[str, str | None] = {
            "--trait": trait,
            "--trait-set": trait_set,
        }

        if run_id is not None:
            extra_args["--run-id"] = run_id

        if fold is not None:
            extra_args["--fold"] = str(fold)

        if debug:
            extra_args["--debug"] = None

        if sample < 1.0:
            extra_args["--sample"] = str(sample)

        cmd = build_base_command(
            "src.models.autogluon",
            params_path=params_path,
            overwrite=False,
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
    # Generate or find run ID for this training session
    # --resume: use most recent run ID (error if none exists)
    # default: create a new run ID
    training_tasks = [t for t in tasks if t["task_type"] in ["cv_fold", "full_model"]]
    if training_tasks:
        sample_trait = training_tasks[0]["trait"]
        base_dir = project_root / cfg.models.dir_fp / sample_trait / cfg.train.arch
        print(f"\nSearching for existing runs in: {base_dir}")
        if resume:
            run_id = get_latest_run_id(base_dir)
            if run_id is None:
                run_id = generate_run_id()
                print(f"No existing runs found. Creating new run: {run_id}")
            else:
                print(f"Found existing run. Resuming: {run_id}")
        else:
            run_id = generate_run_id()
            print(f"Creating new run: {run_id}")
    else:
        run_id = generate_run_id()
        print(f"\nCreating new run: {run_id}")
    print(f"\nRunning model training locally (max {max_parallel} tasks in parallel)...")

    # Separate tasks into training tasks and cv_stats tasks
    training_tasks = [t for t in tasks if t["task_type"] in ["cv_fold", "full_model"]]
    cv_stats_tasks = [t for t in tasks if t["task_type"] == "cv_stats"]

    print(f"\nProcessing {len(training_tasks)} training tasks with up to {max_parallel} workers...")
    successful = []
    failed = []

    product_code = cfg.product_code

    # First, process all training tasks
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
                run_id,
            ): task
            for task in training_tasks
        }

        # Wait for completion
        for future in as_completed(futures):
            task, success = future.result()
            if success:
                successful.append(task)
            else:
                failed.append(task)

    # Report training results
    print(f"\n{'=' * 60}")
    print(f"Training completed: {len(successful)}/{len(training_tasks)} tasks")
    if failed:
        print(f"Failed: {len(failed)} tasks")
        for task in failed:
            task_name = f"{task['trait']}/{task['trait_set']}"
            if task["fold"] is not None:
                task_name += f"/fold_{task['fold']}"
            else:
                task_name += f"/{task['task_type']}"
            print(f"  - {task_name}")
        print("✗ Some training tasks failed to process")
        sys.exit(1)

    # Now process cv_stats tasks (they depend on CV folds being complete)
    print(f"\nProcessing {len(cv_stats_tasks)} CV stats tasks...")
    cv_stats_successful = []
    cv_stats_failed = []

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all cv_stats tasks
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
                run_id,
            ): task
            for task in cv_stats_tasks
        }

        # Wait for completion
        for future in as_completed(futures):
            task, success = future.result()
            if success:
                cv_stats_successful.append(task)
            else:
                cv_stats_failed.append(task)

    # Report cv_stats results
    print(f"\nCV stats completed: {len(cv_stats_successful)}/{len(cv_stats_tasks)} tasks")
    if cv_stats_failed:
        print(f"Failed: {len(cv_stats_failed)} tasks")
        for task in cv_stats_failed:
            task_name = f"{task['trait']}/{task['trait_set']}/cv_stats"
            print(f"  - {task_name}")
        print("✗ Some CV stats tasks failed to process")
        sys.exit(1)

    print("\n✓ Model training and CV stats calculation completed successfully!")


def resubmit_failed_job(
    task: dict,
    params_path: str | None,
    debug: bool,
    sample: float,
    resume: bool,
    overwrite: bool,
    partition: str,
    time_limit: str,
    cpus: int,
    mem: str,
    log_dir: Path,
    product_code: str,
) -> int:
    """
    Resubmit a failed training task.

    Args:
        task: Task dictionary with trait, trait_set, fold, and task_type
        params_path: Path to parameters file
        debug: Enable debug mode
        sample: Fraction of data to sample
        resume: Resume from last run
        overwrite: Overwrite existing models
        partition: Slurm partition to use
        time_limit: Time limit for job
        cpus: Number of CPUs
        mem: Memory allocation
        log_dir: Directory for log files
        product_code: Product code for namespacing job names

    Returns:
        New job ID
    """
    trait = task["trait"]
    trait_set = task["trait_set"]
    fold = task["fold"]
    task_type = task["task_type"]

    # Construct command
    if task_type == "cv_stats":
        extra_args: dict[str, str | None] = {
            "--trait": trait,
            "--trait-set": trait_set,
        }
        if overwrite:
            extra_args["--overwrite"] = None

        cmd_parts = build_base_command(
            "src.models.calculate_cv_stats",
            params_path=params_path,
            overwrite=False,
            extra_args=extra_args,
        )
    else:
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

        cmd_parts = build_base_command(
            "src.models.autogluon",
            params_path=params_path,
            overwrite=False,
            extra_args=extra_args,
        )

    command = " ".join(cmd_parts)

    # Create job name (include product_code to avoid conflicts across products)
    job_name_parts = ["train", trait[:8], product_code[:12], trait_set[:4]]
    if fold is not None:
        job_name_parts.append(f"f{fold}")
    else:
        job_name_parts.append(task_type[:4])
    job_name = "_".join(job_name_parts) + "_retry"

    # Create log file names
    task_name = f"{trait}_{trait_set}"
    if fold is not None:
        task_name += f"_fold_{fold}"
    else:
        task_name += f"_{task_type}"
    task_name += "_retry"

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
    return job_id


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
    max_retries: int = 2,
) -> None:
    """Submit separate Slurm jobs for each training task."""
    # Separate tasks into training and cv_stats
    training_tasks = [t for t in tasks if t["task_type"] in ["cv_fold", "full_model"]]
    cv_stats_tasks = [t for t in tasks if t["task_type"] == "cv_stats"]

    # Generate or find run ID for this training session
    # All jobs in this session will use the same run_id
    # --resume: use most recent run ID (error if none exists)
    # default: create a new run ID
    if training_tasks:
        sample_trait = training_tasks[0]["trait"]
        base_dir = (
            project_root / cfg.models.dir_fp / sample_trait / cfg.train.arch
        )
        print(f"\nSearching for existing runs in: {base_dir}")
        if resume:
            run_id = get_latest_run_id(base_dir)
            if run_id is None:
                run_id = generate_run_id()
                print(f"No existing runs found. Creating new run: {run_id}")
            else:
                print(f"Found existing run. Resuming: {run_id}")
        else:
            run_id = generate_run_id()
            print(f"Creating new run: {run_id}")
    else:
        run_id = generate_run_id()
        print(f"\nCreating new run: {run_id}")

    print(f"\nSubmitting {len(training_tasks)} training jobs...")
    print(f"Will submit {len(cv_stats_tasks)} CV stats jobs with dependencies...")

    # Create log directory
    product_code = cfg.product_code
    log_dir = Path("logs/train_models") / product_code
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create partition distributor for round-robin distribution
    distributor = PartitionDistributor(partitions)

    # Check for existing jobs in the queue to avoid duplicate submissions
    print("\nChecking for existing jobs in queue...")
    existing_jobs = get_existing_job_names()
    if existing_jobs:
        print(f"Found {len(existing_jobs)} existing jobs in queue")
    else:
        print("No existing jobs found")

    # Track job IDs for each trait/trait_set combination (for CV fold dependencies)
    trait_set_fold_jobs: dict[tuple[str, str], list[int]] = {}

    # Submit training jobs
    job_ids = []
    task_to_job = {}
    skipped_jobs = []

    for task in training_tasks:
        trait = task["trait"]
        trait_set = task["trait_set"]
        fold = task["fold"]
        task_type = task["task_type"]

        # Construct command for this task
        extra_args: dict[str, str | None] = {
            "--trait": trait,
            "--trait-set": trait_set,
            "--run-id": run_id,
        }

        if fold is not None:
            extra_args["--fold"] = str(fold)

        if debug:
            extra_args["--debug"] = None

        if sample < 1.0:
            extra_args["--sample"] = str(sample)

        cmd_parts = build_base_command(
            "src.models.autogluon",
            params_path=params_path,
            overwrite=False,
            extra_args=extra_args,
        )
        command = " ".join(cmd_parts)

        # Create job name (include product_code to avoid conflicts across products)
        job_name_parts = ["train", trait[:8], product_code[:12], trait_set[:4]]
        if fold is not None:
            job_name_parts.append(f"f{fold}")
        else:
            job_name_parts.append("full")
        job_name = "_".join(job_name_parts)

        # Check if job already exists in queue
        if job_name in existing_jobs:
            existing_job_id, existing_state = existing_jobs[job_name]
            task_desc = f"{trait}/{trait_set}"
            if fold is not None:
                task_desc += f"/fold_{fold}"
            else:
                task_desc += f"/{task_type}"

            print(
                f"  Skipping {task_desc}: job '{job_name}' already in queue "
                f"(job {existing_job_id}, state {existing_state})"
            )

            # Track this existing job for dependency management
            job_id = int(existing_job_id)
            job_ids.append(job_id)
            task_to_job[job_id] = task
            skipped_jobs.append((job_name, existing_job_id, existing_state))

            # Track CV fold jobs for dependency management
            if task_type == "cv_fold":
                key = (trait, trait_set)
                if key not in trait_set_fold_jobs:
                    trait_set_fold_jobs[key] = []
                trait_set_fold_jobs[key].append(job_id)

            continue

        # Create log file names
        task_name = f"{trait}_{trait_set}"
        if fold is not None:
            task_name += f"_fold_{fold}"
        else:
            task_name += f"_{task_type}"

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

        # Track CV fold jobs for dependency management
        if task_type == "cv_fold":
            key = (trait, trait_set)
            if key not in trait_set_fold_jobs:
                trait_set_fold_jobs[key] = []
            trait_set_fold_jobs[key].append(job_id)

        task_desc = f"{trait}/{trait_set}"
        if fold is not None:
            task_desc += f"/fold_{fold}"
        else:
            task_desc += f"/{task_type}"

        # Show partition in output if using multiple partitions
        partition_info = f" [{partition}]" if len(distributor) > 1 else ""
        print(f"  Submitted job {job_id} for: {task_desc}{partition_info}")

        # Add delay to avoid overwhelming Slurm scheduler
        time.sleep(0.5)

    num_submitted = len(training_tasks) - len(skipped_jobs)
    print(f"\nSubmitted {num_submitted} new training jobs")
    if skipped_jobs:
        print(f"Skipped {len(skipped_jobs)} jobs already in queue")

    # Wait for training jobs to complete and handle retries if enabled
    if wait and max_retries > 0:
        print(f"\nWaiting for {len(training_tasks)} training jobs to complete...")
        print(f"Will automatically retry failed jobs up to {max_retries} times")

        retry_count = 0
        while retry_count <= max_retries:
            print(f"\n--- Attempt {retry_count + 1}/{max_retries + 1} ---")

            # Wait for all training jobs
            successful_jobs = []
            failed_jobs = []

            for job_id in list(job_ids):
                if job_id not in task_to_job:
                    continue

                task = task_to_job[job_id]
                task_type = task["task_type"]

                # Skip cv_stats jobs (they will be submitted later)
                if task_type == "cv_stats":
                    continue

                task_name = f"{task['trait']}/{task['trait_set']}"
                if task["fold"] is not None:
                    task_name += f"/fold_{task['fold']}"
                else:
                    task_name += f"/{task_type}"

                success = wait_for_job_completion(job_id, poll_interval=5)

                if success:
                    successful_jobs.append((job_id, task))
                else:
                    failed_jobs.append((job_id, task))
                    print(f"✗ Job {job_id} ({task_name}) failed")

            # Check if any failures are due to node issues and can be retried
            retryable_jobs = []
            permanent_failures = []

            for job_id, task in failed_jobs:
                if is_node_failure(job_id):
                    retryable_jobs.append((job_id, task))
                else:
                    permanent_failures.append((job_id, task))

            # Report status
            print(f"\nTraining round {retry_count + 1} results:")
            print(f"  Successful: {len(successful_jobs)}")
            print(f"  Failed (retryable - node issues): {len(retryable_jobs)}")
            print(f"  Failed (permanent - code/data issues): {len(permanent_failures)}")

            # If we have permanent failures, abort
            if permanent_failures:
                print("\n⚠️  Permanent failures detected. These jobs failed due to code or data issues:")
                for job_id, task in permanent_failures:
                    task_name = f"{task['trait']}/{task['trait_set']}"
                    if task["fold"] is not None:
                        task_name += f"/fold_{task['fold']}"
                    else:
                        task_name += f"/{task['task_type']}"
                    print(f"  - Job {job_id}: {task_name}")
                    print(f"    Check logs: {log_dir.absolute()}/{job_id}_*.err")
                print("\n✗ Cannot proceed with automatic retry. Please fix the issues and rerun.")
                sys.exit(1)

            # If no retryable jobs, we're done
            if not retryable_jobs:
                print(f"\n✓ All training jobs completed successfully!")
                break

            # If we've exhausted retries, abort
            if retry_count >= max_retries:
                print(f"\n✗ Maximum retries ({max_retries}) exhausted. Still have {len(retryable_jobs)} failed jobs.")
                for job_id, task in retryable_jobs:
                    task_name = f"{task['trait']}/{task['trait_set']}"
                    if task["fold"] is not None:
                        task_name += f"/fold_{task['fold']}"
                    else:
                        task_name += f"/{task['task_type']}"
                    print(f"  - Job {job_id}: {task_name}")
                sys.exit(1)

            # Resubmit failed jobs
            print(f"\n🔄 Resubmitting {len(retryable_jobs)} failed jobs (node failures)...")
            retry_count += 1

            for old_job_id, task in retryable_jobs:
                task_name = f"{task['trait']}/{task['trait_set']}"
                if task["fold"] is not None:
                    task_name += f"/fold_{task['fold']}"
                else:
                    task_name += f"/{task['task_type']}"

                # Get partition for this job
                partition = distributor.get_next()

                # Resubmit the job
                new_job_id = resubmit_failed_job(
                    task,
                    params_path,
                    debug,
                    sample,
                    resume,
                    overwrite,
                    partition,
                    time_limit,
                    cpus,
                    mem,
                    log_dir,
                    product_code,
                )

                # Update tracking
                task_to_job[new_job_id] = task
                del task_to_job[old_job_id]
                job_ids.remove(old_job_id)
                job_ids.append(new_job_id)

                # Update CV fold tracking if this is a fold job
                if task["task_type"] == "cv_fold":
                    key = (task["trait"], task["trait_set"])
                    if key in trait_set_fold_jobs:
                        fold_jobs = trait_set_fold_jobs[key]
                        if old_job_id in fold_jobs:
                            fold_jobs.remove(old_job_id)
                        fold_jobs.append(new_job_id)

                partition_info = f" [{partition}]" if len(distributor) > 1 else ""
                print(f"  Resubmitted job {new_job_id} for: {task_name}{partition_info} (retry {retry_count})")

    # Now submit cv_stats jobs
    # If we already waited for training jobs (max_retries > 0 and wait=True),
    # don't use dependencies since those jobs are already completed and gone from queue
    training_already_completed = wait and max_retries > 0
    if training_already_completed:
        print(f"\nSubmitting {len(cv_stats_tasks)} CV stats jobs (training already completed)...")
    else:
        print(f"\nSubmitting {len(cv_stats_tasks)} CV stats jobs with dependencies...")
    skipped_cvstats_jobs = []

    for task in cv_stats_tasks:
        trait = task["trait"]
        trait_set = task["trait_set"]

        # Get the fold job IDs that this cv_stats task depends on
        key = (trait, trait_set)
        dependency_job_ids = trait_set_fold_jobs.get(key, [])

        if not dependency_job_ids and not training_already_completed:
            print(f"  Warning: No CV fold jobs found for {trait}/{trait_set}")
            continue

        # Construct command for cv_stats task
        extra_args: dict[str, str | None] = {
            "--trait": trait,
            "--trait-set": trait_set,
            "--run-id": run_id,
        }

        if overwrite:
            extra_args["--overwrite"] = None

        cmd_parts = build_base_command(
            "src.models.calculate_cv_stats",
            params_path=params_path,
            overwrite=False,
            extra_args=extra_args,
        )
        command = " ".join(cmd_parts)

        # Create job name (include product_code to avoid conflicts across products)
        job_name = f"cvstats_{trait[:8]}_{product_code[:12]}_{trait_set[:4]}"

        # Check if job already exists in queue
        if job_name in existing_jobs:
            existing_job_id, existing_state = existing_jobs[job_name]
            task_desc = f"{trait}/{trait_set}/cv_stats"

            print(
                f"  Skipping {task_desc}: job '{job_name}' already in queue "
                f"(job {existing_job_id}, state {existing_state})"
            )

            # Track this existing job
            job_id = int(existing_job_id)
            job_ids.append(job_id)
            task_to_job[job_id] = task
            skipped_cvstats_jobs.append((job_name, existing_job_id, existing_state))

            continue

        # Create log file names
        task_name = f"{trait}_{trait_set}_cv_stats"

        # Get partition using round-robin distribution
        partition = distributor.get_next()

        # Create Slurm job configuration
        # Only use dependencies if training jobs are still running (not already completed)
        if training_already_completed:
            slurm = Slurm(
                job_name=job_name,
                output=str(log_dir / f"%j_{task_name}.log"),
                error=str(log_dir / f"%j_{task_name}.err"),
                time=time_limit,
                cpus_per_task=cpus,
                mem=mem,
                partition=partition,
            )
        else:
            dependency_str = ":".join(str(jid) for jid in dependency_job_ids)
            slurm = Slurm(
                job_name=job_name,
                output=str(log_dir / f"%j_{task_name}.log"),
                error=str(log_dir / f"%j_{task_name}.err"),
                time=time_limit,
                cpus_per_task=cpus,
                mem=mem,
                partition=partition,
                dependency=f"afterok:{dependency_str}",
            )

        # Submit the job
        job_id = slurm.sbatch(command)
        job_ids.append(job_id)
        task_to_job[job_id] = task

        task_desc = f"{trait}/{trait_set}/cv_stats"
        partition_info = f" [{partition}]" if len(distributor) > 1 else ""
        if training_already_completed:
            print(f"  Submitted job {job_id} for: {task_desc}{partition_info}")
        else:
            print(
                f"  Submitted job {job_id} for: {task_desc}{partition_info} "
                f"(depends on {len(dependency_job_ids)} fold jobs)"
            )

        # Add delay to avoid overwhelming Slurm scheduler
        time.sleep(0.5)

    num_cvstats_submitted = len(cv_stats_tasks) - len(skipped_cvstats_jobs)
    print(f"\nSubmitted {num_cvstats_submitted} new CV stats jobs")
    if skipped_cvstats_jobs:
        print(f"Skipped {len(skipped_cvstats_jobs)} CV stats jobs already in queue")

    total_submitted = num_submitted + num_cvstats_submitted
    total_skipped = len(skipped_jobs) + len(skipped_cvstats_jobs)
    print(
        f"Total: {total_submitted} new jobs submitted, {total_skipped} existing jobs found "
        f"(Grand total: {len(job_ids)} jobs tracked)"
    )

    # Show distribution summary if using multiple partitions
    if len(distributor) > 1:
        summary = distributor.get_summary()
        print("Job distribution across partitions:")
        for partition, count in summary.items():
            print(f"  {partition}: {count} jobs")

    print(f"Logs directory: {log_dir.absolute()}")

    if wait:
        # If we already waited for training jobs (with retries), now wait for cvstats jobs
        if max_retries > 0:
            print(f"\nWaiting for {len(cv_stats_tasks)} CV stats jobs to complete...")
            cvstats_successful = []
            cvstats_failed = []

            for job_id in job_ids:
                if job_id not in task_to_job:
                    continue

                task = task_to_job[job_id]
                task_type = task["task_type"]

                # Only wait for cvstats jobs (training jobs already completed)
                if task_type != "cv_stats":
                    continue

                task_name = f"{task['trait']}/{task['trait_set']}/cv_stats"

                success = wait_for_job_completion(job_id, poll_interval=5)

                if success:
                    cvstats_successful.append(task)
                else:
                    cvstats_failed.append(task)
                    print(f"✗ Job {job_id} ({task_name}) failed. Check logs:")
                    print(
                        f"  {log_dir.absolute()}/{job_id}_{task_name.replace('/', '_')}.err"
                    )

            # Report final results
            print(f"\n{'=' * 60}")
            print(f"Training jobs: {len(training_tasks)}/{len(training_tasks)} completed")
            print(f"CV stats jobs: {len(cvstats_successful)}/{len(cv_stats_tasks)} completed")

            if cvstats_failed:
                print(f"\n✗ {len(cvstats_failed)} CV stats jobs failed:")
                for task in cvstats_failed:
                    t_name = f"{task['trait']}/{task['trait_set']}/cv_stats"
                    print(f"  - {t_name}")
                sys.exit(1)

            print("\n✓ All training and CV stats jobs completed successfully!")
        else:
            # No retries enabled, wait for all jobs (original behavior)
            print(f"\nWaiting for {len(job_ids)} jobs to complete...")
            successful = []
            failed = []

            for job_id in job_ids:
                task = task_to_job[job_id]
                task_type = task["task_type"]
                task_name = f"{task['trait']}/{task['trait_set']}"
                if task["fold"] is not None:
                    task_name += f"/fold_{task['fold']}"
                else:
                    task_name += f"/{task_type}"

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
                    task_type = task["task_type"]
                    t_name = f"{task['trait']}/{task['trait_set']}"
                    if task["fold"] is not None:
                        t_name += f"/fold_{task['fold']}"
                    else:
                        t_name += f"/{task_type}"
                    print(f"  - {t_name}")
                print("✗ Some jobs failed")
                sys.exit(1)

            print("\n✓ Model training and CV stats calculation completed successfully!")
    else:
        print("\nJobs submitted. Not waiting for completion (--no-wait flag set).")
        if max_retries > 0:
            print("⚠️  Note: Automatic retry is only available when waiting for jobs (without --no-wait).")
        print("CV stats jobs will run automatically after their dependencies complete.")
        print(
            "To check status, use: squeue -u $USER"
        )


if __name__ == "__main__":
    main()
