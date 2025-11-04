#!/usr/bin/env python
"""
Entry point script to generate Slurm jobs for building sPlot trait maps.

This script reads trait names from the configuration file and submits a separate
Slurm job for each trait to process it in parallel, or runs them locally in parallel
if Slurm is unavailable.

Execution mode is determined by the USE_SLURM environment variable (from .env file).
Set USE_SLURM=false to use local execution by default, or use --local flag to override.
"""

import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from simple_slurm import Slurm

# Setup environment and path
from src.pipeline.entrypoint_utils import (
    setup_environment,
    determine_execution_mode,
    setup_log_directory,
    build_base_command,
    add_common_args,
    add_execution_args,
    add_partition_args,
    add_resource_args,
    resolve_partitions,
    PartitionDistributor,
    wait_for_job_completion,
)

project_root = setup_environment()

from src.conf.conf import get_config  # noqa: E402


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Submit Slurm jobs to build sPlot trait maps for all configured traits."
        )
    )
    add_common_args(parser, include_partition=False)
    add_execution_args(parser, multi_job=True, n_jobs_default=4)
    add_partition_args(parser, enable_multi_partition=True)
    add_resource_args(
        parser,
        time_default="00:15:00",
        cpus_default=10,
        mem_default="30GB",
        include_gpus=False,
    )
    return parser.parse_args()


def run_trait_job(
    trait: str, params_path: str | None, overwrite: bool
) -> tuple[str, int]:
    """Run a single trait job locally."""
    cmd = build_base_command(
        "src.data.build_splot_map",
        params_path=params_path,
        overwrite=overwrite,
        extra_args={"--trait": trait}
    )

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return trait, result.returncode


def run_local(
    trait_names: list[str], params_path: str | None, overwrite: bool, n_jobs: int
) -> None:
    """Run trait jobs locally in parallel."""
    print(f"\nRunning {len(trait_names)} traits locally with {n_jobs} parallel jobs")

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all jobs
        futures = {
            executor.submit(run_trait_job, trait, params_path, overwrite): trait
            for trait in trait_names
        }

        # Track results
        completed = 0
        failed = []

        # Process results as they complete
        for future in as_completed(futures):
            trait = futures[future]
            try:
                trait_name, returncode = future.result()
                completed += 1
                if returncode == 0:
                    print(f"✓ [{completed}/{len(trait_names)}] Completed: {trait_name}")
                else:
                    print(
                        f"✗ [{completed}/{len(trait_names)}] Failed: {trait_name} "
                        f"(exit code: {returncode})"
                    )
                    failed.append(trait_name)
            except Exception as e:
                completed += 1
                print(f"✗ [{completed}/{len(trait_names)}] Error in {trait}: {e}")
                failed.append(trait)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Completed: {len(trait_names) - len(failed)}/{len(trait_names)}")
    if failed:
        print(f"Failed traits: {', '.join(failed)}")
    else:
        print("All traits processed successfully!")


def run_slurm(
    trait_names: list[str],
    params_path: str | None,
    overwrite: bool,
    partitions: list[str],
    log_dir: Path,
    time_limit: str,
    cpus: int,
    mem: str,
) -> None:
    """Submit trait jobs to Slurm and wait for completion."""
    # Create partition distributor for round-robin distribution
    distributor = PartitionDistributor(partitions)

    job_ids = []
    trait_to_job = {}
    for trait in trait_names:
        # Construct the command
        cmd_parts = build_base_command(
            "src.data.build_splot_map",
            params_path=params_path,
            overwrite=overwrite,
            extra_args={"--trait": trait}
        )
        command = " ".join(cmd_parts)

        # Get partition using round-robin distribution
        partition = distributor.get_next()

        # Create Slurm job configuration
        slurm = Slurm(
            job_name=f"splot_{trait}",
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

        # Show partition in output if using multiple partitions
        partition_info = f" [{partition}]" if len(distributor) > 1 else ""
        print(f"Submitted job {job_id} for trait '{trait}'{partition_info}")

    print(f"\nSubmitted {len(job_ids)} jobs successfully")

    # Show distribution summary if using multiple partitions
    if len(distributor) > 1:
        summary = distributor.get_summary()
        print("Job distribution across partitions:")
        for partition, count in summary.items():
            print(f"  {partition}: {count} jobs")

    print("Job IDs and traits:")
    for job_id in job_ids:
        print(f"  {job_id}: {trait_to_job[job_id]}")

    # Wait for all jobs to complete
    print(f"\nWaiting for {len(job_ids)} jobs to complete...")
    successful = []
    failed = []

    for job_id in job_ids:
        trait = trait_to_job[job_id]
        success = wait_for_job_completion(job_id, poll_interval=5)

        if success:
            successful.append(trait)
            print(f"✓ Job {job_id} ({trait}) completed successfully")
        else:
            failed.append(trait)
            print(f"✗ Job {job_id} ({trait}) failed. Check logs:")
            print(f"  {log_dir.absolute()}/{job_id}_{trait}.err")

    # Report results
    print(f"\n{'=' * 60}")
    print(f"Completed: {len(successful)}/{len(trait_names)} traits")
    if failed:
        print(f"Failed traits: {', '.join(failed)}")
        import sys
        sys.exit(1)
    else:
        print("All traits processed successfully!")


def main() -> None:
    """Main function to submit Slurm jobs or run locally for each trait."""
    args = cli()
    params_path = Path(args.params).absolute()
    cfg = get_config(params_path=params_path)

    # Get traits from configuration
    trait_names = cfg.traits.names
    print(f"Found {len(trait_names)} traits to process: {', '.join(trait_names)}")

    # Determine execution mode
    use_local, mode = determine_execution_mode(args.local)
    print(f"Execution mode: {mode}")

    if use_local:
        # Run locally in parallel
        run_local(trait_names, str(params_path), args.overwrite, args.n_jobs)
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
        log_dir = setup_log_directory("build_splot_maps")
        print(f"Logs will be written to: {log_dir.absolute()}")
        run_slurm(
            trait_names,
            str(params_path),
            args.overwrite,
            partitions,
            log_dir,
            args.time,
            args.cpus,
            args.mem,
        )


if __name__ == "__main__":
    main()
