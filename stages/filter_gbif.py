#!/usr/bin/env python
"""
Entry point script to run GBIF filtering either locally or via Slurm.

This script filters GBIF observations by trait data and calculates resurvey weights.
It uses Dask internally for parallel processing, so only a single job is needed.

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
    add_common_args,
    add_execution_args,
    build_base_command,
    determine_execution_mode,
    setup_environment,
    setup_log_directory,
    wait_for_job_completion,
)

project_root = setup_environment()

from src.conf.conf import get_config  # noqa: E402


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter GBIF data by traits and calculate resurvey weights."
    )
    add_common_args(parser)
    add_execution_args(parser, multi_job=False)
    parser.add_argument(
        "--country",
        type=str,
        required=False,
        default=None,
        help="Optional country code to filter GBIF data (e.g., 'PT' for Portugal)",
    )
    return parser.parse_args()


def run_local(params_path: str | None, overwrite: bool, country: str | None) -> None:
    """Run GBIF filtering locally."""
    print("\nRunning GBIF filtering locally...")

    extra_args = {}
    if country:
        extra_args["--country"] = country

    cmd = build_base_command(
        "src.data.filter_gbif",
        params_path=params_path,
        overwrite=overwrite,
        extra_args=extra_args,
    )

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✓ GBIF filtering completed successfully!")
    else:
        print("\n" + "=" * 60)
        print(f"✗ GBIF filtering failed (exit code: {result.returncode})")
        sys.exit(1)


def run_slurm(
    params_path: str | None,
    overwrite: bool,
    partition: str,
    log_dir: Path,
    country: str | None,
) -> None:
    """Submit GBIF filtering job to Slurm."""
    # Construct the command
    extra_args = {}
    if country:
        extra_args["--country"] = country

    cmd_parts = build_base_command(
        "src.data.filter_gbif",
        params_path=params_path,
        overwrite=overwrite,
        extra_args=extra_args,
    )
    command = " ".join(cmd_parts)

    # Create Slurm job configuration
    slurm = Slurm(
        job_name="filter_gbif",
        output=str(log_dir / "%j_filter_gbif.log"),
        error=str(log_dir / "%j_filter_gbif.err"),
        time="00:30:00",  # Should only take about 5 minutes
        cpus_per_task=60,
        mem="350GB",
        partition=partition,
    )

    # Submit the job
    job_id = slurm.sbatch(command)
    print(f"Submitted job {job_id} for GBIF filtering")
    print(f"Logs will be written to: {log_dir.absolute()}")

    # Wait for job to complete
    print("\n" + "=" * 60)
    print("Waiting for job to complete...")
    log_file = f"{log_dir.absolute()}/{job_id}_filter_gbif.log"
    print(f"Monitor progress: tail -f {log_file}")

    success = wait_for_job_completion(job_id, "filter_gbif", poll_interval=5)

    if success:
        print("\n" + "=" * 60)
        print("✓ GBIF filtering completed successfully!")
    else:
        print("\n" + "=" * 60)
        print("✗ Job failed. Check logs:")
        print(f"  {log_dir.absolute()}/{job_id}_filter_gbif.err")
        sys.exit(1)


def main() -> None:
    """Main function to run GBIF filtering locally or via Slurm."""
    args = cli()
    params_path = Path(args.params).resolve() if args.params else None

    print("=" * 60)
    print("GBIF Data Filtering")
    print("=" * 60)
    if args.country:
        print(f"Country filter: {args.country}")

    # Determine execution mode
    use_local, mode = determine_execution_mode(args.local)

    if use_local:
        # Run locally
        print(f"Execution mode: {mode}")
        run_local(
            str(params_path) if params_path else None,
            args.overwrite,
            args.country,
        )
    else:
        # Submit to Slurm
        print("Execution mode: Slurm")
        log_dir = setup_log_directory("filter_gbif")
        run_slurm(
            str(params_path) if params_path else None,
            args.overwrite,
            args.partition,
            log_dir,
            args.country,
        )


if __name__ == "__main__":
    main()
