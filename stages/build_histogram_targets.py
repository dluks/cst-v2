#!/usr/bin/env python
"""
Entry point script to build histogram targets from GBIF or sPlot observations.

This script submits a Slurm job (or runs locally) to construct histogram targets
for a specified data source.

Execution mode is determined by the USE_SLURM environment variable (from .env file).
Set USE_SLURM=false to use local execution by default, or use --local flag to override.
"""

import argparse
import subprocess
from pathlib import Path

from simple_slurm import Slurm

# Setup environment and path
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
        description="Build histogram targets from GBIF or sPlot observations."
    )
    add_common_args(parser, include_partition=False)
    add_execution_args(parser, multi_job=False, n_jobs_default=1)
    add_partition_args(parser, enable_multi_partition=True)
    add_resource_args(
        parser,
        time_default="02:00:00",
        cpus_default=16,
        mem_default="64GB",
        include_gpus=False,
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["gbif", "splot"],
        required=True,
        help="Data source: 'gbif' or 'splot'.",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to submit Slurm job or run locally."""
    args = cli()
    params_path = Path(args.params).resolve()
    source = args.source

    print(f"Building histogram targets from {source.upper()} data")

    # Determine execution mode
    use_local, mode = determine_execution_mode(args.local)
    print(f"Execution mode: {mode}")

    if use_local:
        run_local(source, str(params_path), args.overwrite)
    else:
        partitions = resolve_partitions(args.partition, args.partitions)
        partition = partitions[0]  # Use first partition for single job
        print(f"Using partition: {partition}")

        log_dir = setup_log_directory("build_histogram_targets")
        print(f"Logs will be written to: {log_dir.absolute()}")

        run_slurm(
            source=source,
            params_path=str(params_path),
            overwrite=args.overwrite,
            partition=partition,
            log_dir=log_dir,
            time_limit=args.time,
            cpus=args.cpus,
            mem=args.mem,
        )


def run_local(source: str, params_path: str, overwrite: bool) -> None:
    """Run histogram target construction locally."""
    cmd = build_base_command(
        "src.data.build_histogram_targets",
        params_path=params_path,
        overwrite=overwrite,
        extra_args={"--source": source},
    )

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print(f"\n✓ Histogram targets for {source.upper()} built successfully")
    else:
        print(f"\n✗ Failed to build histogram targets for {source.upper()}")
        import sys
        sys.exit(result.returncode)


def run_slurm(
    source: str,
    params_path: str,
    overwrite: bool,
    partition: str,
    log_dir: Path,
    time_limit: str,
    cpus: int,
    mem: str,
) -> None:
    """Submit histogram target construction job to Slurm."""
    # Construct the command
    cmd_parts = build_base_command(
        "src.data.build_histogram_targets",
        params_path=params_path,
        overwrite=overwrite,
        extra_args={"--source": source},
    )
    command = " ".join(cmd_parts)

    # Create Slurm job configuration
    slurm = Slurm(
        job_name=f"hist_targets_{source}",
        output=str(log_dir / f"%j_{source}.log"),
        error=str(log_dir / f"%j_{source}.err"),
        time=time_limit,
        cpus_per_task=cpus,
        mem=mem,
        partition=partition,
    )

    # Submit the job
    job_id = slurm.sbatch(command)
    print(f"Submitted job {job_id} for {source.upper()} histogram targets")

    # Wait for job to complete
    print(f"\nWaiting for job {job_id} to complete...")
    success = wait_for_job_completion(job_id, poll_interval=10)

    if success:
        print(f"\n✓ Job {job_id} completed successfully")
    else:
        print(f"\n✗ Job {job_id} failed. Check logs:")
        print(f"  {log_dir.absolute()}/{job_id}_{source}.err")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
