#!/usr/bin/env python
"""
Entry point script to merge histogram targets with EO features.

This script submits a Slurm job (or runs locally) to build the merged
training dataset (train.zarr) that combines GBIF/sPlot histogram targets
with Earth Observation predictor features.

Execution mode is determined by the USE_SLURM environment variable (from .env file).
Set USE_SLURM=false to use local execution by default, or use --local flag to override.
"""

import argparse
import subprocess
from pathlib import Path

from simple_slurm import Slurm

from src.pipeline.entrypoint_utils import (
    add_common_args,
    add_execution_args,
    add_partition_args,
    add_resource_args,
    build_base_command,
    determine_execution_mode,
    resolve_partitions,
    setup_environment,
    setup_log_directory,
    wait_for_job_completion,
)

project_root = setup_environment()


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge histogram targets with EO features for training."
    )
    add_common_args(parser, include_partition=False)
    add_execution_args(parser, multi_job=False, n_jobs_default=1)
    add_partition_args(parser, enable_multi_partition=True)
    add_resource_args(
        parser,
        time_default="00:30:00",
        cpus_default=8,
        mem_default="64GB",
        include_gpus=False,
    )
    return parser.parse_args()


def main() -> None:
    """Main function to submit Slurm job or run locally."""
    args = cli()
    params_path = Path(args.params).resolve()

    print("Building histogram XY training data")

    use_local, mode = determine_execution_mode(args.local)
    print(f"Execution mode: {mode}")

    if use_local:
        run_local(str(params_path), args.overwrite)
    else:
        partitions = resolve_partitions(args.partition, args.partitions)
        partition = partitions[0]
        print(f"Using partition: {partition}")

        log_dir = setup_log_directory("build_histogram_xy")
        print(f"Logs will be written to: {log_dir.absolute()}")

        run_slurm(
            params_path=str(params_path),
            overwrite=args.overwrite,
            partition=partition,
            log_dir=log_dir,
            time_limit=args.time,
            cpus=args.cpus,
            mem=args.mem,
        )


def run_local(params_path: str, overwrite: bool) -> None:
    """Run histogram XY merge locally."""
    cmd = build_base_command(
        "src.features.build_histogram_xy",
        params_path=params_path,
        overwrite=overwrite,
    )

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print("\n✓ Histogram XY training data built successfully")
    else:
        print("\n✗ Failed to build histogram XY training data")
        import sys
        sys.exit(result.returncode)


def run_slurm(
    params_path: str,
    overwrite: bool,
    partition: str,
    log_dir: Path,
    time_limit: str,
    cpus: int,
    mem: str,
) -> None:
    """Submit histogram XY merge job to Slurm."""
    cmd_parts = build_base_command(
        "src.features.build_histogram_xy",
        params_path=params_path,
        overwrite=overwrite,
    )
    command = " ".join(cmd_parts)

    slurm = Slurm(
        job_name="hist_xy_merge",
        output=str(log_dir / "%j_hist_xy.log"),
        error=str(log_dir / "%j_hist_xy.err"),
        time=time_limit,
        cpus_per_task=cpus,
        mem=mem,
        partition=partition,
    )

    job_id = slurm.sbatch(command)
    print(f"Submitted job {job_id} for histogram XY merge")

    print(f"\nWaiting for job {job_id} to complete...")
    success = wait_for_job_completion(job_id, poll_interval=10)

    if success:
        print(f"\n✓ Job {job_id} completed successfully")
    else:
        print(f"\n✗ Job {job_id} failed. Check logs:")
        print(f"  {log_dir.absolute()}/{job_id}_hist_xy.err")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
