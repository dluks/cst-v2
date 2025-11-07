#!/usr/bin/env python
"""
Entry point script to run sPlot filtering either locally or via Slurm.

This script filters sPlot observations by trait data and calculates resurvey weights.
It processes data sequentially (no internal parallelization), so only a single job is needed.

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
    add_resource_args,
    build_base_command,
    determine_execution_mode,
    setup_environment,
    setup_log_directory,
    wait_for_job_completion,
)

project_root = setup_environment()


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter sPlot data by traits and calculate resurvey weights."
    )
    add_common_args(parser)
    add_execution_args(parser, multi_job=False)
    add_resource_args(
        parser, time_default="01:00:00", cpus_default=4, mem_default="64GB"
    )
    return parser.parse_args()


def run_local(params_path: str | None, overwrite: bool) -> None:
    """Run sPlot filtering locally."""
    print("\nRunning sPlot filtering locally...")

    cmd = build_base_command(
        "src.data.filter_splot",
        params_path=params_path,
        overwrite=overwrite,
    )

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✓ sPlot filtering completed successfully!")
    else:
        print("\n" + "=" * 60)
        print(f"✗ sPlot filtering failed (exit code: {result.returncode})")
        sys.exit(1)


def run_slurm(
    params_path: str | None,
    overwrite: bool,
    partition: str,
    time_limit: str,
    cpus: int,
    mem: str,
    log_dir: Path,
) -> None:
    """Submit sPlot filtering job to Slurm."""
    # Construct the command
    cmd_parts = build_base_command(
        "src.data.filter_splot",
        params_path=params_path,
        overwrite=overwrite,
    )
    command = " ".join(cmd_parts)

    # Create Slurm job configuration
    # sPlot filtering includes R data extraction and pandas processing
    slurm = Slurm(
        job_name="filter_splot",
        output=str(log_dir / "%j_filter_splot.log"),
        error=str(log_dir / "%j_filter_splot.err"),
        time=time_limit,
        cpus_per_task=cpus,
        mem=mem,
        partition=partition,
    )

    # Submit the job
    job_id = slurm.sbatch(command)
    print(f"Submitted job {job_id} for sPlot filtering")
    print(f"Logs will be written to: {log_dir.absolute()}")

    # Wait for job to complete
    print("\n" + "=" * 60)
    print("Waiting for job to complete...")
    log_file = f"{log_dir.absolute()}/{job_id}_filter_splot.log"
    print(f"Monitor progress: tail -f {log_file}")

    success = wait_for_job_completion(job_id, "filter_splot", poll_interval=5)

    if success:
        print("\n" + "=" * 60)
        print("✓ sPlot filtering completed successfully!")
    else:
        print("\n" + "=" * 60)
        print("✗ Job failed. Check logs:")
        print(f"  {log_dir.absolute()}/{job_id}_filter_splot.err")
        sys.exit(1)


def main() -> None:
    """Main function to run sPlot filtering locally or via Slurm."""
    args = cli()
    params_path = Path(args.params).resolve() if args.params else None

    print("=" * 60)
    print("sPlot Data Filtering")
    print("=" * 60)

    # Determine execution mode
    use_local, mode = determine_execution_mode(args.local)

    if use_local:
        # Run locally
        print(f"Execution mode: {mode}")
        run_local(
            str(params_path) if params_path else None,
            args.overwrite,
        )
    else:
        # Submit to Slurm
        print("Execution mode: Slurm")
        log_dir = setup_log_directory("filter_splot")
        run_slurm(
            str(params_path) if params_path else None,
            args.overwrite,
            args.partition,
            args.time,
            args.cpus,
            args.mem,
            log_dir,
        )


if __name__ == "__main__":
    main()
