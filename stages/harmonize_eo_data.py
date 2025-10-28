#!/usr/bin/env python
"""
Entry point script to harmonize EO data either locally or on Slurm.

This script provides a unified interface to run the harmonization pipeline
either on the local machine or submit it as a Slurm job.

Execution mode is determined by the USE_SLURM environment variable (from .env file).
Set USE_SLURM=false to use local execution by default, or use --local flag to override.
"""

import argparse
import subprocess
import sys
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
    add_resource_args,
    wait_for_job_completion,
)

project_root = setup_environment()


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Harmonize EO data locally or on Slurm."
    )
    add_common_args(parser)
    add_execution_args(parser, multi_job=False)
    add_resource_args(
        parser,
        time_default="04:00:00",
        cpus_default=16,
        mem_default="64GB"
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Dry run without writing output files.",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run harmonization locally or on Slurm."""
    args = cli()

    # Convert paths to absolute if provided
    params_path = str(Path(args.params).absolute()) if args.params else None

    # Determine execution mode
    use_local, mode = determine_execution_mode(args.local)
    print(f"Execution mode: {mode}")

    if use_local:
        # Run locally
        run_local(params_path, args.dry_run, args.overwrite)
    else:
        # Submit to Slurm
        run_slurm(
            params_path,
            args.dry_run,
            args.overwrite,
            args.partition,
            args.time,
            args.cpus,
            args.mem,
            wait=not args.no_wait,
        )


def run_local(
    params_path: str | None,
    dry_run: bool,
    overwrite: bool,
) -> None:
    """Run harmonization locally."""
    print("\nRunning harmonization locally...")

    extra_args: dict[str, str | None] = {}
    if dry_run:
        extra_args["--dry-run"] = None

    cmd = build_base_command(
        "src.data.harmonize_eo_data",
        params_path=params_path,
        overwrite=overwrite,
        extra_args=extra_args
    )

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print("\n✓ Harmonization completed successfully!")
    else:
        print(f"\n✗ Harmonization failed (exit code: {result.returncode})")
        sys.exit(1)


def run_slurm(
    params_path: str | None,
    dry_run: bool,
    overwrite: bool,
    partition: str,
    time_limit: str,
    cpus: int,
    mem: str,
    wait: bool,
) -> None:
    """Submit harmonization job to Slurm."""
    print("\nSubmitting harmonization job to Slurm...")

    # Create log directory
    log_dir = setup_log_directory("harmonize_eo_data")

    # Construct command
    extra_args: dict[str, str | None] = {}
    if dry_run:
        extra_args["--dry-run"] = None

    cmd_parts = build_base_command(
        "src.data.harmonize_eo_data",
        params_path=params_path,
        overwrite=overwrite,
        extra_args=extra_args
    )
    command = " ".join(cmd_parts)

    # Create Slurm job configuration
    slurm = Slurm(
        job_name="harmonize_eo",
        output=str(log_dir / "%j_harmonize_eo.log"),
        error=str(log_dir / "%j_harmonize_eo.err"),
        time=time_limit,
        cpus_per_task=cpus,
        mem=mem,
        partition=partition,
    )

    # Submit the job
    job_id = slurm.sbatch(command)
    print(f"Submitted job {job_id}")
    print(f"Logs will be written to: {log_dir.absolute()}")
    print(f"Monitor progress: tail -f {log_dir.absolute()}/{job_id}_harmonize_eo.log")

    if wait:
        # Wait for job to complete
        success = wait_for_job_completion(job_id, poll_interval=30)

        if success:
            print("\n✓ Harmonization completed successfully!")
        else:
            print("\n✗ Job failed. Check logs:")
            print(f"  {log_dir.absolute()}/{job_id}_harmonize_eo.err")
            sys.exit(1)
    else:
        print("\nJob submitted. Not waiting for completion (--no-wait flag set).")


if __name__ == "__main__":
    main()
