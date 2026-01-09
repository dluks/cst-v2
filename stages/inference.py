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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dask.dataframe as dd
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


def run_slurm(
    params_path: str | None,
    overwrite: bool,
    partitions: list[str],
    tasks: list[dict],
    args: argparse.Namespace,
    no_wait: bool,
) -> None:
    """Submit all inference tasks to Slurm.

    Args:
        params_path: Path to params.yaml file
        overwrite: Whether to overwrite existing outputs
        partitions: List of Slurm partitions to use
        tasks: List of task dictionaries
        args: Command-line arguments with resource specifications
        no_wait: Whether to wait for jobs to complete
    """
    print(f"\n{'='*80}")
    print("SLURM SUBMISSION")
    print(f"{'='*80}\n")

    # Distribute tasks across partitions
    distributor = PartitionDistributor(partitions)

    # Group tasks by type for dependency management
    predict_jobs = []
    cov_jobs = []
    aoa_jobs = []
    final_jobs = []

    for task in tasks:
        trait = task["trait"]
        trait_set = task["trait_set"]
        task_type = task["task_type"]
        partition = distributor.next_partition()

        # Build command based on task type
        if task_type == "predict":
            script_path = "src/models/predict_single_trait.py"
            cmd_args = ["--trait", trait, "--trait-set", trait_set]
            time = args.predict_time
            cpus = args.predict_cpus
            mem = args.predict_mem
            gres = None
        elif task_type == "cov":
            script_path = "src/models/predict_single_trait.py"
            cmd_args = ["--trait", trait, "--trait-set", trait_set, "--cov"]
            time = args.predict_time
            cpus = args.predict_cpus
            mem = args.predict_mem
            gres = None
        elif task_type == "aoa":
            script_path = "src/analysis/aoa_single_trait.py"
            cmd_args = ["--trait", trait, "--trait-set", trait_set]
            time = args.aoa_time
            cpus = args.aoa_cpus
            mem = args.aoa_mem
            gres = f"gpu:{args.aoa_gpus}"
        elif task_type == "final":
            script_path = "src/data/build_final_product_single_trait.py"
            cmd_args = ["--trait", trait, "--trait-set", trait_set, "--dest", args.dest]
            time = args.final_time
            cpus = args.final_cpus
            mem = args.final_mem
            gres = None
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Add common options
        if overwrite:
            cmd_args.append("--overwrite")
        if params_path:
            cmd_args.extend(["--params", params_path])

        # Build full command
        base_cmd = build_base_command(script_path, cmd_args)

        # Create Slurm job
        job_name = f"{task_type}_{trait}_{trait_set}"
        slurm = Slurm(
            job_name=job_name,
            output=f"logs/{job_name}_%j.out",
            error=f"logs/{job_name}_%j.err",
            partition=partition,
            time=time,
            cpus_per_task=cpus,
            mem=mem,
            gres=gres,
        )

        # Determine dependencies
        dependency = None
        if task_type == "final":
            # Final product depends on predict, cov, and aoa for the same trait/trait_set
            dep_jobs = []
            for job_info in predict_jobs + cov_jobs + aoa_jobs:
                if job_info["trait"] == trait and job_info["trait_set"] == trait_set:
                    dep_jobs.append(str(job_info["job_id"]))
            if dep_jobs:
                dependency = f"afterok:{':'.join(dep_jobs)}"

        # Submit job
        if dependency:
            job_id = slurm.sbatch(base_cmd, dependency=dependency)
        else:
            job_id = slurm.sbatch(base_cmd)

        print(
            f"Submitted {task_type} for {trait} ({trait_set}): "
            f"Job ID {job_id} on {partition}"
            + (f" with dependency {dependency}" if dependency else "")
        )

        # Store job info
        job_info = {
            "job_id": job_id,
            "trait": trait,
            "trait_set": trait_set,
            "task_type": task_type,
        }

        if task_type == "predict":
            predict_jobs.append(job_info)
        elif task_type == "cov":
            cov_jobs.append(job_info)
        elif task_type == "aoa":
            aoa_jobs.append(job_info)
        elif task_type == "final":
            final_jobs.append(job_info)

    all_jobs = predict_jobs + cov_jobs + aoa_jobs + final_jobs
    all_job_ids = [str(job["job_id"]) for job in all_jobs]

    print(f"\nSubmitted {len(all_jobs)} jobs")
    print(f"  - Predict: {len(predict_jobs)}")
    print(f"  - CoV: {len(cov_jobs)}")
    print(f"  - AoA: {len(aoa_jobs)}")
    print(f"  - Final: {len(final_jobs)}")

    if not no_wait:
        print("\nWaiting for jobs to complete...")
        wait_for_job_completion(all_job_ids)
        print("All jobs completed!")
    else:
        print("\nJobs submitted. Not waiting for completion (--no-wait specified).")
        print(f"Job IDs: {', '.join(all_job_ids)}")


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
            args.no_wait,
        )


if __name__ == "__main__":
    main()
