#!/usr/bin/env python
"""
Entry point script to build CV splits either locally or on Slurm.

This script provides a unified interface to assign spatial k-fold cross-validation
splits for all traits in parallel either on the local machine or submit it as Slurm
jobs.

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
    parser = argparse.ArgumentParser(description="Build CV splits locally or on Slurm.")
    add_common_args(parser, include_partition=False)
    add_partition_args(parser, enable_multi_partition=True)
    add_resource_args(
        parser,
        time_default="02:00:00",
        cpus_default=40,
        mem_default="64GB",
        include_gpus=False,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for Slurm job to complete (submit and exit).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help=(
            "Maximum number of traits to process in parallel (default: 4). "
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
    return parser.parse_args()


def main() -> None:
    """Main function to run CV splits building locally or on Slurm."""
    args = cli()

    # Convert paths to absolute if provided
    params_path = str(Path(args.params).absolute()) if args.params else None

    # Set CONFIG_PATH BEFORE importing to avoid module-level load error
    if params_path is not None:
        os.environ["CONFIG_PATH"] = params_path

    # Get configuration
    cfg = get_config(params_path=params_path)

    # Determine execution mode
    use_local, mode = determine_execution_mode(args.local)

    # Get traits to process
    if args.traits:
        traits = args.traits
        print(f"Processing specified traits: {', '.join(traits)}")
    else:
        y_fp = Path(project_root, cfg.train.Y.fp).resolve()
        traits = (
            dd.read_parquet(y_fp).columns.difference(["x", "y", "source"]).to_list()
        )
        print(f"Found {len(traits)} traits to process: {', '.join(traits)}")

    # Handle debug mode
    if args.debug:
        print("\n⚠️  DEBUG MODE: Processing only 1 trait")
        traits = [traits[0]]
        print(f"Debug trait: {traits[0]}")

    print(f"Execution mode: {mode}")

    if use_local:
        # Run locally
        run_local(
            params_path,
            args.debug,
            args.overwrite,
            traits,
            args.max_parallel,
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
            args.overwrite,
            partitions,
            args.time,
            args.cpus,
            args.mem,
            wait=not args.no_wait,
            traits=traits,
        )


def run_single_trait(
    trait: str,
    params_path: str | None,
    debug: bool,
    overwrite: bool,
) -> tuple[str, bool]:
    """
    Run CV splits assignment for a single trait.

    Args:
        trait: Trait name
        params_path: Path to parameters file
        debug: Enable debug mode
        overwrite: Overwrite existing splits

    Returns:
        Tuple of (trait_name, success)
    """
    extra_args: dict[str, str | None] = {
        "--trait": trait,
    }

    if debug:
        extra_args["--debug"] = None

    if overwrite:
        extra_args["--overwrite"] = None

    cmd = build_base_command(
        "src.features.build_cv_splits",
        params_path=params_path,
        overwrite=False,  # Handled by --overwrite flag in extra_args
        extra_args=extra_args,
    )

    # Set CONFIG_PATH in subprocess environment to avoid import-time errors
    env = os.environ.copy()
    if params_path:
        env["CONFIG_PATH"] = params_path

    print(f"  Starting: {trait}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode == 0:
        print(f"  ✓ Completed: {trait}")
        return (trait, True)
    else:
        print(f"  ✗ Failed: {trait}")
        print(f"    Exit code: {result.returncode}")
        if result.stderr:
            # Print last 20 lines of stderr for better error visibility
            stderr_lines = result.stderr.strip().split("\n")
            print(f"    Last {min(20, len(stderr_lines))} lines of stderr:")
            for line in stderr_lines[-20:]:
                print(f"      {line}")
        return (trait, False)


def run_local(
    params_path: str | None,
    debug: bool,
    overwrite: bool,
    traits: list[str],
    max_parallel: int,
) -> None:
    """Run CV splits building locally with parallel processing."""
    print(
        f"\nRunning CV splits building locally "
        f"(max {max_parallel} traits in parallel)..."
    )

    # Process traits in parallel
    print(f"\nProcessing {len(traits)} traits with up to {max_parallel} workers...")
    successful = []
    failed = []

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all trait jobs
        futures = {
            executor.submit(
                run_single_trait, trait, params_path, debug, overwrite
            ): trait
            for trait in traits
        }

        # Wait for completion
        for future in as_completed(futures):
            trait_name, success = future.result()
            if success:
                successful.append(trait_name)
            else:
                failed.append(trait_name)

    # Report results
    print(f"\n{'=' * 60}")
    print(f"Completed: {len(successful)}/{len(traits)} traits")
    if failed:
        print(f"Failed: {len(failed)} traits: {', '.join(failed)}")
        print("✗ Some traits failed to process")
        sys.exit(1)

    print("\n✓ CV splits building completed successfully!")


def run_slurm(
    params_path: str | None,
    debug: bool,
    overwrite: bool,
    partitions: list[str],
    time_limit: str,
    cpus: int,
    mem: str,
    wait: bool,
    traits: list[str],
) -> None:
    """Submit separate Slurm jobs for each trait."""
    print(f"\nSubmitting {len(traits)} Slurm jobs (one per trait)...")

    # Create log directory
    log_dir = Path("logs/build_cv_splits")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create partition distributor for round-robin distribution
    distributor = PartitionDistributor(partitions)

    # Submit a job for each trait
    job_ids = []
    trait_to_job = {}

    for trait in traits:
        # Construct command for this trait
        extra_args: dict[str, str | None] = {
            "--trait": trait,
        }

        if debug:
            extra_args["--debug"] = None

        if overwrite:
            extra_args["--overwrite"] = None

        cmd_parts = build_base_command(
            "src.features.build_cv_splits",
            params_path=params_path,
            overwrite=False,  # Handled by --overwrite flag in extra_args
            extra_args=extra_args,
        )
        command = " ".join(cmd_parts)

        # Get partition using round-robin distribution
        partition = distributor.get_next()

        # Create Slurm job configuration
        slurm = Slurm(
            job_name=f"cv_{trait[:12]}",
            output=str(log_dir / f"%j_{trait}.log"),
            error=str(log_dir / f"%j_{trait}.err"),
            time=time_limit,
            cpus_per_task=cpus,
            mem=mem,
            partition=partition,
        )

        # Submit the job
        job_id = slurm.sbatch(command)
        job_ids.append(job_id)
        trait_to_job[job_id] = trait
        print(f"  Submitted job {job_id} for trait: {trait}")

    print(f"\nSubmitted {len(job_ids)} jobs")
    print(f"Logs directory: {log_dir.absolute()}")

    # Show distribution summary if using multiple partitions
    if len(distributor) > 1:
        summary = distributor.get_summary()
        print("Job distribution across partitions:")
        for partition, count in summary.items():
            print(f"  {partition}: {count} jobs")

    if wait:
        # Wait for all jobs to complete
        print(f"\nWaiting for {len(job_ids)} jobs to complete...")
        successful = []
        failed = []

        for job_id in job_ids:
            trait = trait_to_job[job_id]
            success = wait_for_job_completion(job_id)

            if success:
                successful.append(trait)
            else:
                failed.append(trait)
                print(f"✗ Job {job_id} ({trait}) failed. Check logs:")
                print(f"  {log_dir.absolute()}/{job_id}_{trait}.err")

        # Report results
        print(f"\n{'=' * 60}")
        print(f"Completed: {len(successful)}/{len(traits)} traits")
        if failed:
            print(f"Failed: {len(failed)} traits: {', '.join(failed)}")
            print("✗ Some jobs failed")
            sys.exit(1)

        print("\n✓ CV splits building completed successfully!")
    else:
        print("\nJobs submitted. Not waiting for completion (--no-wait flag set).")
        print("To check status later, run this script again with --wait flag.")


if __name__ == "__main__":
    main()
