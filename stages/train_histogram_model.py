#!/usr/bin/env python
"""
Entry point script to train the histogram MLP model.

This script submits a Slurm job (or runs locally) to train a PyTorch MLP
that predicts trait probability distributions from EO features.

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
        description="Train histogram MLP for trait distribution prediction."
    )
    add_common_args(parser, include_partition=False)
    add_execution_args(parser, multi_job=False, n_jobs_default=1)
    add_partition_args(parser, enable_multi_partition=False)
    add_resource_args(
        parser,
        time_default="04:00:00",
        cpus_default=16,
        mem_default="64GB",
        include_gpus=True,
        gpus_default="1",
    )

    # Training-specific arguments (forwarded to train.py)
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Train a specific fold only (0-indexed). If None, run full CV.",
    )
    parser.add_argument(
        "--full-only", action="store_true",
        help="Train only the full model (skip CV).",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run ID (format: run_YYYYMMDD_HHMMSS). Auto-generated if not provided.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from most recent run.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: 3 epochs, batch_size=64.",
    )
    return parser.parse_args()


def _build_train_extra_args(args: argparse.Namespace) -> dict[str, str | None]:
    """Build extra arguments to forward to train.py."""
    extra: dict[str, str | None] = {}
    if args.fold is not None:
        extra["--fold"] = str(args.fold)
    if args.full_only:
        extra["--full-only"] = None
    if args.run_id:
        extra["--run-id"] = args.run_id
    if args.resume:
        extra["--resume"] = None
    if args.debug:
        extra["--debug"] = None
    return extra


def main() -> None:
    """Main function to submit Slurm job or run locally."""
    args = cli()
    params_path = Path(args.params).resolve()

    print("Training histogram MLP model")

    use_local, mode = determine_execution_mode(args.local)
    print(f"Execution mode: {mode}")

    extra_args = _build_train_extra_args(args)

    if use_local:
        run_local(str(params_path), args.overwrite, extra_args)
    else:
        partitions = resolve_partitions(args.partition, None)
        partition = partitions[0]
        print(f"Using partition: {partition}")

        log_dir = setup_log_directory("train_histogram_model")
        print(f"Logs will be written to: {log_dir.absolute()}")

        run_slurm(
            params_path=str(params_path),
            overwrite=args.overwrite,
            extra_args=extra_args,
            partition=partition,
            log_dir=log_dir,
            time_limit=args.time,
            cpus=args.cpus,
            mem=args.mem,
            gpus=args.gpus,
        )


def run_local(
    params_path: str,
    overwrite: bool,
    extra_args: dict[str, str | None],
) -> None:
    """Run histogram MLP training locally."""
    cmd = build_base_command(
        "src.models.histogram_mlp.train",
        params_path=params_path,
        overwrite=overwrite,
        extra_args=extra_args,
    )

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print("\n✓ Histogram MLP training completed successfully")
    else:
        print("\n✗ Histogram MLP training failed")
        import sys
        sys.exit(result.returncode)


def run_slurm(
    params_path: str,
    overwrite: bool,
    extra_args: dict[str, str | None],
    partition: str,
    log_dir: Path,
    time_limit: str,
    cpus: int,
    mem: str,
    gpus: str,
) -> None:
    """Submit histogram MLP training job to Slurm."""
    cmd_parts = build_base_command(
        "src.models.histogram_mlp.train",
        params_path=params_path,
        overwrite=overwrite,
        extra_args=extra_args,
    )
    command = " ".join(cmd_parts)

    slurm = Slurm(
        job_name="hist_mlp_train",
        output=str(log_dir / "%j_hist_mlp_train.log"),
        error=str(log_dir / "%j_hist_mlp_train.err"),
        time=time_limit,
        cpus_per_task=cpus,
        mem=mem,
        partition=partition,
        gres=f"gpu:{gpus}",
    )

    job_id = slurm.sbatch(command)
    print(f"Submitted job {job_id} for histogram MLP training")

    print(f"\nWaiting for job {job_id} to complete...")
    success = wait_for_job_completion(job_id, poll_interval=30)

    if success:
        print(f"\n✓ Job {job_id} completed successfully")
    else:
        print(f"\n✗ Job {job_id} failed. Check logs:")
        print(f"  {log_dir.absolute()}/{job_id}_hist_mlp_train.err")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
