#!/usr/bin/env python
"""
Entry point script to build prediction features either locally or on Slurm.

This script provides a unified interface to run the prediction feature building pipeline
either on the local machine or submit it as a Slurm job.

Execution mode is determined by the USE_SLURM environment variable (from .env file).
Set USE_SLURM=false to use local execution by default, or use --local flag to override.
"""

import argparse
import os
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
        description="Build prediction features locally or on Slurm."
    )
    add_common_args(parser)
    add_execution_args(parser, multi_job=False)
    add_resource_args(
        parser, time_default="02:00:00", cpus_default=8, mem_default="32GB"
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run prediction building locally or on Slurm."""
    args = cli()

    # Convert paths to absolute if provided
    params_path = str(Path(args.params).absolute()) if args.params else None

    # Determine execution mode
    use_local, mode = determine_execution_mode(args.local)
    print(f"Execution mode: {mode}")

    if use_local:
        # Run locally
        run_local(params_path, args.debug, args.overwrite)
    else:
        # Submit to Slurm
        run_slurm(
            params_path,
            args.debug,
            args.overwrite,
            args.partition,
            args.time,
            args.cpus,
            args.mem,
            wait=not args.no_wait,
        )


def run_local(
    params_path: str | None,
    debug: bool,
    overwrite: bool,
) -> None:
    """Run prediction building locally."""
    print("\nRunning prediction building locally...")

    # Set CONFIG_PATH environment variable
    if params_path:
        os.environ["CONFIG_PATH"] = params_path

    extra_args: dict[str, str | None] = {}
    if debug:
        extra_args["--debug"] = None

    cmd = build_base_command(
        "src.features.build_predict",
        params_path=params_path,
        overwrite=overwrite,
        extra_args=extra_args,
    )

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print("\n✓ Prediction building completed successfully!")
    else:
        print(f"\n✗ Prediction building failed (exit code: {result.returncode})")
        sys.exit(1)


def run_slurm(
    params_path: str | None,
    debug: bool,
    overwrite: bool,
    partition: str,
    time_limit: str,
    cpus: int,
    mem: str,
    wait: bool,
) -> None:
    """Submit prediction building job to Slurm."""
    print("\nSubmitting prediction building job to Slurm...")

    # Create log directory
    log_dir = setup_log_directory("build_predict")

    # Construct command
    extra_args: dict[str, str | None] = {}
    if debug:
        extra_args["--debug"] = None

    cmd_parts = build_base_command(
        "src.features.build_predict",
        params_path=params_path,
        overwrite=overwrite,
        extra_args=extra_args,
    )

    # Prepend CONFIG_PATH export if params_path is provided
    # This ensures the environment variable is set before Python imports
    if params_path:
        command = f"export CONFIG_PATH={params_path} && {' '.join(cmd_parts)}"
    else:
        command = " ".join(cmd_parts)

    # Create Slurm job configuration
    slurm = Slurm(
        job_name="build_predict",
        output=str(log_dir / "%j_build_predict.log"),
        error=str(log_dir / "%j_build_predict.err"),
        time=time_limit,
        cpus_per_task=cpus,
        mem=mem,
        partition=partition,
    )

    # Submit the job
    job_id = slurm.sbatch(command)
    print(f"Submitted job {job_id}")
    print(f"Logs will be written to: {log_dir.absolute()}")
    print(f"Monitor progress: tail -f {log_dir.absolute()}/{job_id}_build_predict.err")

    if wait:
        # Wait for job to complete
        success = wait_for_job_completion(job_id)

        if success:
            print("\n✓ Prediction building completed successfully!")
        else:
            print("\n✗ Job failed. Check logs:")
            print(f"  {log_dir.absolute()}/{job_id}_build_predict.err")
            sys.exit(1)
    else:
        print("\nJob submitted. Not waiting for completion (--no-wait flag set).")


if __name__ == "__main__":
    main()
