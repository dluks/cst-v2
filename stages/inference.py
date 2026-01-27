#!/usr/bin/env python
"""
Entry point script to run inference (prediction, CoV, AoA, and final product building) either locally or on Slurm.

This script provides a unified interface to run all inference tasks for traits and trait sets
in parallel either on the local machine or submit them as Slurm jobs.

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

# Setup environment and path FIRST
from src.pipeline.entrypoint_utils import (
    PartitionDistributor,
    add_common_args,
    add_execution_args,
    add_partition_args,
    build_base_command,
    determine_execution_mode,
    get_existing_job_names,
    resolve_partitions,
    setup_environment,
    setup_log_directory,
    wait_for_job_completion,
)

project_root = setup_environment()

# Import config AFTER setup_environment
from src.conf.conf import get_config  # noqa: E402


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference tasks locally or on Slurm."
    )
    add_common_args(parser, include_partition=False)
    add_partition_args(parser, enable_multi_partition=True)

    # Resource arguments for different task types
    parser.add_argument(
        "--predict-time",
        type=str,
        default="02:00:00",
        help="Time limit for prediction jobs (default: 02:00:00)",
    )
    parser.add_argument(
        "--predict-cpus",
        type=int,
        default=56,
        help="Number of CPUs for prediction jobs (default: 56)",
    )
    parser.add_argument(
        "--predict-mem",
        type=str,
        default="64GB",
        help="Memory for prediction jobs (default: 64GB)",
    )
    parser.add_argument(
        "--aoa-time",
        type=str,
        default="04:00:00",
        help="Time limit for AoA jobs (default: 04:00:00)",
    )
    parser.add_argument(
        "--aoa-cpus",
        type=int,
        default=112,
        help="Number of CPUs for AoA jobs (default: 112)",
    )
    parser.add_argument(
        "--aoa-mem",
        type=str,
        default="128GB",
        help="Memory for AoA jobs (default: 128GB)",
    )
    parser.add_argument(
        "--aoa-gpus",
        type=str,
        default="1",
        help="Number of GPUs for AoA jobs (default: 1)",
    )
    parser.add_argument(
        "--final-time",
        type=str,
        default="01:00:00",
        help="Time limit for final product jobs (default: 01:00:00)",
    )
    parser.add_argument(
        "--final-cpus",
        type=int,
        default=28,
        help="Number of CPUs for final product jobs (default: 28)",
    )
    parser.add_argument(
        "--final-mem",
        type=str,
        default="32GB",
        help="Memory for final product jobs (default: 32GB)",
    )

    # Task selection flags
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Run only prediction tasks (no CoV, AoA, or final product)",
    )
    parser.add_argument(
        "--cov-only",
        action="store_true",
        help="Run only CoV calculation tasks (no prediction, AoA, or final product)",
    )
    parser.add_argument(
        "--aoa-only",
        action="store_true",
        help="Run only AoA calculation tasks (no prediction, CoV, or final product)",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Run only final product building (no prediction, CoV, or AoA)",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip prediction tasks",
    )
    parser.add_argument(
        "--skip-cov",
        action="store_true",
        help="Skip CoV calculation tasks",
    )
    parser.add_argument(
        "--skip-aoa",
        action="store_true",
        help="Skip AoA calculation tasks",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Skip final product building tasks",
    )

    # Other options
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for Slurm jobs to complete (submit and exit)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help=(
            "Maximum number of tasks to run in parallel (default: 4). "
            "Only applies to local execution."
        ),
    )
    parser.add_argument(
        "--traits",
        type=str,
        nargs="+",
        default=None,
        help="Specific trait(s) to process. If not specified, processes all traits.",
    )
    parser.add_argument(
        "--trait-sets",
        type=str,
        nargs="+",
        default=None,
        choices=["splot", "gbif", "splot_gbif"],
        help="Specific trait set(s) to process. If not specified, processes all trait sets.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="local",
        choices=["local", "sftp", "both"],
        help="Destination for final products (default: local)",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments.

    Args:
        args: Parsed command-line arguments

    Raises:
        SystemExit: If arguments are invalid
    """
    # Check for mutually exclusive task flags
    only_flags = [args.predict_only, args.cov_only, args.aoa_only, args.final_only]
    if sum(only_flags) > 1:
        print("Error: Only one of --predict-only, --cov-only, --aoa-only, --final-only can be used")
        sys.exit(1)

    # Check for conflicting skip flags
    if args.predict_only and any([args.skip_predict]):
        print("Error: --predict-only conflicts with --skip-predict")
        sys.exit(1)
    if args.cov_only and any([args.skip_cov]):
        print("Error: --cov-only conflicts with --skip-cov")
        sys.exit(1)
    if args.aoa_only and any([args.skip_aoa]):
        print("Error: --aoa-only conflicts with --skip-aoa")
        sys.exit(1)
    if args.final_only and any([args.skip_final]):
        print("Error: --final-only conflicts with --skip-final")
        sys.exit(1)


def determine_tasks_to_run(args: argparse.Namespace) -> dict[str, bool]:
    """Determine which tasks to run based on command-line flags.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dictionary with task names as keys and boolean values indicating whether to run them
    """
    # If any --*-only flag is set, only run that task
    if args.predict_only:
        return {"predict": True, "cov": False, "aoa": False, "final": False}
    if args.cov_only:
        return {"predict": False, "cov": True, "aoa": False, "final": False}
    if args.aoa_only:
        return {"predict": False, "cov": False, "aoa": True, "final": False}
    if args.final_only:
        return {"predict": False, "cov": False, "aoa": False, "final": True}

    # Otherwise, run all tasks except those explicitly skipped
    return {
        "predict": not args.skip_predict,
        "cov": not args.skip_cov,
        "aoa": not args.skip_aoa,
        "final": not args.skip_final,
    }


def get_traits_to_process(
    args: argparse.Namespace, cfg, project_root: Path
) -> list[str]:
    """Get list of traits to process.

    Args:
        args: Parsed command-line arguments
        cfg: Configuration object
        project_root: Project root path

    Returns:
        List of trait names
    """
    if args.traits:
        traits = args.traits
        print(f"Processing specified traits: {', '.join(traits)}")
    else:
        y_fp = Path(project_root, cfg.train.Y.fp).resolve()
        all_cols = (
            dd.read_parquet(y_fp).columns.difference(["x", "y", "source"]).to_list()
        )
        # Filter out reliability columns
        traits = [t for t in all_cols if not t.endswith("_reliability")]
        print(f"Found {len(traits)} traits to process: {', '.join(traits)}")

    return traits


def generate_tasks(
    traits: list[str],
    trait_sets: list[str],
    tasks_to_run: dict[str, bool],
) -> list[dict]:
    """Generate list of inference tasks.

    Args:
        traits: List of trait names
        trait_sets: List of trait set names
        tasks_to_run: Dictionary indicating which task types to run

    Returns:
        List of task dictionaries
    """
    tasks = []

    for trait in traits:
        for trait_set in trait_sets:
            if tasks_to_run["predict"]:
                tasks.append({
                    "trait": trait,
                    "trait_set": trait_set,
                    "task_type": "predict",
                })
            if tasks_to_run["cov"]:
                tasks.append({
                    "trait": trait,
                    "trait_set": trait_set,
                    "task_type": "cov",
                })
            if tasks_to_run["aoa"]:
                tasks.append({
                    "trait": trait,
                    "trait_set": trait_set,
                    "task_type": "aoa",
                })
            if tasks_to_run["final"]:
                tasks.append({
                    "trait": trait,
                    "trait_set": trait_set,
                    "task_type": "final",
                })

    return tasks


def run_task_local(
    task: dict,
    params_path: str | None,
    overwrite: bool,
    dest: str,
) -> tuple[dict, int]:
    """Run a single inference task locally.

    Args:
        task: Task dictionary with trait, trait_set, and task_type
        params_path: Path to params.yaml file
        overwrite: Whether to overwrite existing outputs
        dest: Destination for final products

    Returns:
        Tuple of (task dict, return code)
    """
    trait = task["trait"]
    trait_set = task["trait_set"]
    task_type = task["task_type"]

    # Build command based on task type
    if task_type == "predict":
        script_path = "src/models/predict_single_trait.py"
        cmd = ["python", script_path, "--trait", trait, "--trait-set", trait_set]
    elif task_type == "cov":
        script_path = "src/models/predict_single_trait.py"
        cmd = ["python", script_path, "--trait", trait, "--trait-set", trait_set, "--cov"]
    elif task_type == "aoa":
        script_path = "src/analysis/aoa_single_trait.py"
        cmd = ["python", script_path, "--trait", trait, "--trait-set", trait_set]
    elif task_type == "final":
        script_path = "src/data/build_final_product_single_trait.py"
        cmd = ["python", script_path, "--trait", trait, "--trait-set", trait_set, "--dest", dest]
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Add common options
    if overwrite:
        cmd.append("--overwrite")
    if params_path:
        cmd.extend(["--params", params_path])

    print(f"Running {task_type} for {trait} ({trait_set})...")
    result = subprocess.run(cmd, cwd=project_root)

    return task, result.returncode


def run_local(
    params_path: str | None,
    overwrite: bool,
    tasks: list[dict],
    max_parallel: int,
    dest: str,
) -> None:
    """Run all inference tasks locally with parallel execution.

    Args:
        params_path: Path to params.yaml file
        overwrite: Whether to overwrite existing outputs
        tasks: List of task dictionaries
        max_parallel: Maximum number of parallel tasks
        dest: Destination for final products
    """
    print(f"\n{'='*80}")
    print("LOCAL EXECUTION")
    print(f"{'='*80}\n")

    failed_tasks = []

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(
                run_task_local,
                task,
                params_path,
                overwrite,
                dest,
            ): task
            for task in tasks
        }

        for future in as_completed(futures):
            task, returncode = future.result()
            if returncode != 0:
                failed_tasks.append(task)
                print(
                    f"❌ FAILED: {task['task_type']} for {task['trait']} ({task['trait_set']})"
                )
            else:
                print(
                    f"✅ COMPLETED: {task['task_type']} for {task['trait']} ({task['trait_set']})"
                )

    # Print summary
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Successful: {len(tasks) - len(failed_tasks)}")
    print(f"Failed: {len(failed_tasks)}")

    if failed_tasks:
        print("\nFailed tasks:")
        for task in failed_tasks:
            print(f"  - {task['task_type']}: {task['trait']} ({task['trait_set']})")
        sys.exit(1)


def get_task_resources(task_type: str, args: argparse.Namespace) -> dict:
    """Get resource requirements for a task type.

    Args:
        task_type: Type of task (predict, cov, aoa, final)
        args: Command-line arguments with resource specifications

    Returns:
        Dictionary with time, cpus, mem, and gres keys
    """
    if task_type in ("predict", "cov"):
        return {
            "time": args.predict_time,
            "cpus": args.predict_cpus,
            "mem": args.predict_mem,
            "gres": None,
        }
    elif task_type == "aoa":
        return {
            "time": args.aoa_time,
            "cpus": args.aoa_cpus,
            "mem": args.aoa_mem,
            "gres": f"gpu:{args.aoa_gpus}" if args.aoa_gpus != "0" else None,
        }
    elif task_type == "final":
        return {
            "time": args.final_time,
            "cpus": args.final_cpus,
            "mem": args.final_mem,
            "gres": None,
        }
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# Task type to module path mapping
TASK_MODULES = {
    "predict": "src.models.predict_single_trait",
    "cov": "src.models.predict_single_trait",
    "aoa": "src.analysis.aoa_single_trait",
    "final": "src.data.build_final_product_single_trait",
}


def run_slurm(
    params_path: str | None,
    overwrite: bool,
    partitions: list[str],
    tasks: list[dict],
    args: argparse.Namespace,
    cfg,
    no_wait: bool,
) -> None:
    """Submit all inference tasks to Slurm.

    Args:
        params_path: Path to params.yaml file
        overwrite: Whether to overwrite existing outputs
        partitions: List of Slurm partitions to use
        tasks: List of task dictionaries
        args: Command-line arguments with resource specifications
        cfg: Configuration object
        no_wait: Whether to wait for jobs to complete
    """
    print(f"\n{'='*80}")
    print("SLURM SUBMISSION")
    print(f"{'='*80}\n")

    # Setup log directory
    log_dir = setup_log_directory("inference")
    product_log_dir = log_dir / cfg.product_code
    product_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be written to: {product_log_dir.absolute()}")

    # Initialize partition distributor
    distributor = PartitionDistributor(partitions)

    # Check for existing jobs in queue
    print("\nChecking for existing jobs in queue...")
    existing_jobs = get_existing_job_names()
    if existing_jobs:
        print(f"Found {len(existing_jobs)} existing jobs in queue")
    else:
        print("No existing jobs found in queue")

    # Track jobs by trait/trait_set for dependencies
    job_tracker: dict[tuple[str, str], dict[str, int]] = {}
    skipped_jobs = []

    for task in tasks:
        trait = task["trait"]
        trait_set = task["trait_set"]
        task_type = task["task_type"]

        # Build job name (include product_code to avoid conflicts)
        job_name = (
            f"inf_{task_type[:4]}_{trait[:8]}_{cfg.product_code[:12]}_{trait_set[:4]}"
        )

        # Check if job already exists in queue
        if job_name in existing_jobs:
            existing_job_id, existing_state = existing_jobs[job_name]
            print(
                f"  Skipping {task_type}/{trait}/{trait_set}: "
                f"job already in queue ({existing_state})"
            )
            skipped_jobs.append((job_name, existing_job_id, existing_state))
            # Track for dependencies
            key = (trait, trait_set)
            if key not in job_tracker:
                job_tracker[key] = {}
            job_tracker[key][task_type] = int(existing_job_id)
            continue

        # Build extra_args for build_base_command
        extra_args: dict[str, str | None] = {
            "--trait": trait,
            "--trait-set": trait_set,
        }
        if task_type == "cov":
            extra_args["--cov"] = None
        if task_type == "final":
            extra_args["--dest"] = args.dest

        # Build command using module path
        cmd_parts = build_base_command(
            TASK_MODULES[task_type],
            params_path=params_path,
            overwrite=overwrite,
            extra_args=extra_args,
        )
        command = " ".join(cmd_parts)

        # Get resources for task type
        resources = get_task_resources(task_type, args)

        # Determine dependencies for final tasks
        dependency = None
        if task_type == "final":
            key = (trait, trait_set)
            if key in job_tracker:
                dep_job_ids = list(job_tracker[key].values())
                if dep_job_ids:
                    dependency = "afterok:" + ":".join(str(j) for j in dep_job_ids)

        # Get partition using round-robin distribution
        partition = distributor.get_next()

        # Build Slurm job kwargs (only include optional params if they have values)
        slurm_kwargs = {
            "job_name": job_name,
            "output": str(product_log_dir / f"%j_{task_type}_{trait}_{trait_set}.log"),
            "error": str(product_log_dir / f"%j_{task_type}_{trait}_{trait_set}.err"),
            "partition": partition,
            "time": resources["time"],
            "cpus_per_task": resources["cpus"],
            "mem": resources["mem"],
        }
        if resources["gres"]:
            slurm_kwargs["gres"] = resources["gres"]
        if dependency:
            slurm_kwargs["dependency"] = dependency

        # Create Slurm job
        slurm = Slurm(**slurm_kwargs)

        # Submit job
        job_id = slurm.sbatch(command)

        # Track job for dependencies
        key = (trait, trait_set)
        if key not in job_tracker:
            job_tracker[key] = {}
        job_tracker[key][task_type] = job_id

        # Print submission info
        dep_info = ""
        if dependency:
            n_deps = len(dependency.split(":")) - 1
            dep_info = f" (depends on {n_deps} jobs)"
        partition_info = f" [{partition}]" if len(distributor) > 1 else ""
        print(
            f"  Submitted {task_type}/{trait}/{trait_set}: "
            f"job {job_id}{partition_info}{dep_info}"
        )

        # Small delay to avoid overwhelming scheduler
        time.sleep(0.5)

    # Collect all job IDs for summary
    all_job_ids = []
    task_counts = {"predict": 0, "cov": 0, "aoa": 0, "final": 0}
    for trait_jobs in job_tracker.values():
        for task_type, job_id in trait_jobs.items():
            all_job_ids.append(job_id)
            task_counts[task_type] += 1

    num_submitted = len(all_job_ids) - len(skipped_jobs)
    print(f"\n{'='*60}")
    print(f"Submitted {num_submitted} new jobs")
    if skipped_jobs:
        print(f"Skipped {len(skipped_jobs)} jobs already in queue")
    print(f"  - Predict: {task_counts['predict']}")
    print(f"  - CoV: {task_counts['cov']}")
    print(f"  - AoA: {task_counts['aoa']}")
    print(f"  - Final: {task_counts['final']}")

    # Show partition distribution if using multiple partitions
    if len(distributor) > 1:
        summary = distributor.get_summary()
        print("\nJob distribution across partitions:")
        for partition, count in summary.items():
            print(f"  {partition}: {count} jobs")

    if not no_wait:
        print("\nWaiting for jobs to complete...")
        # Wait for final jobs (they depend on others, so waiting for them waits for all)
        final_job_ids = [
            job_tracker[key]["final"]
            for key in job_tracker
            if "final" in job_tracker[key]
        ]
        if final_job_ids:
            for job_id in final_job_ids:
                success = wait_for_job_completion(job_id, poll_interval=10)
                if not success:
                    print(f"✗ Job {job_id} failed. Check logs in {product_log_dir}")
                    sys.exit(1)
            print("\n✓ All jobs completed successfully!")
        else:
            # No final jobs, wait for all submitted jobs
            for job_id in all_job_ids:
                success = wait_for_job_completion(job_id, poll_interval=10)
                if not success:
                    print(f"✗ Job {job_id} failed. Check logs in {product_log_dir}")
                    sys.exit(1)
            print("\n✓ All jobs completed successfully!")
    else:
        print("\nJobs submitted. Not waiting for completion (--no-wait specified).")
        print("Monitor with: squeue -u $USER")
        print(f"Logs: {product_log_dir.absolute()}")


def main() -> None:
    """Main function to run inference locally or on Slurm."""
    args = cli()

    # Validate arguments
    validate_args(args)

    # Convert paths to absolute if provided
    params_path = str(Path(args.params).resolve()) if args.params else None

    # Set CONFIG_PATH BEFORE importing dataset_utils to avoid module-level load error
    if params_path is not None:
        os.environ["CONFIG_PATH"] = params_path

    # Get configuration
    cfg = get_config(params_path=params_path)

    # Determine execution mode
    use_local, mode = determine_execution_mode(args.local)

    # Get traits to process
    traits = get_traits_to_process(args, cfg, project_root)

    # Get trait sets to process
    trait_sets = args.trait_sets or cfg.train.trait_sets
    print(f"Processing trait sets: {', '.join(trait_sets)}")

    # Determine which tasks to run
    tasks_to_run = determine_tasks_to_run(args)
    print(f"\nTasks to run:")
    print(f"  - Predict: {tasks_to_run['predict']}")
    print(f"  - CoV: {tasks_to_run['cov']}")
    print(f"  - AoA: {tasks_to_run['aoa']}")
    print(f"  - Final: {tasks_to_run['final']}")

    # Generate all inference tasks
    tasks = generate_tasks(traits, trait_sets, tasks_to_run)

    print(f"\nTotal inference tasks: {len(tasks)}")
    print(f"  - Predict tasks: {len([t for t in tasks if t['task_type'] == 'predict'])}")
    print(f"  - CoV tasks: {len([t for t in tasks if t['task_type'] == 'cov'])}")
    print(f"  - AoA tasks: {len([t for t in tasks if t['task_type'] == 'aoa'])}")
    print(f"  - Final tasks: {len([t for t in tasks if t['task_type'] == 'final'])}")
    print(f"\nExecution mode: {mode}")

    if use_local:
        # Run locally
        run_local(
            params_path,
            args.overwrite,
            tasks,
            args.max_parallel,
            args.dest,
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
            args.overwrite,
            partitions,
            tasks,
            args,
            cfg,
            args.no_wait,
        )


if __name__ == "__main__":
    main()
