#!/usr/bin/env python
"""
Entry point to run Optuna HPO for the histogram MLP model.

Submits N parallel Slurm jobs that share an Optuna study via file-based
journal storage, or runs locally for testing.

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
    submit_job_with_retry,
    wait_for_job_completion,
)

project_root = setup_environment()


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch Optuna HPO workers for histogram MLP."
    )
    add_common_args(parser, include_partition=False)
    add_execution_args(parser, multi_job=False, n_jobs_default=1)
    add_partition_args(parser, enable_multi_partition=True)
    add_resource_args(
        parser,
        time_default="06:00:00",
        cpus_default=16,
        mem_default="64GB",
        include_gpus=True,
        gpus_default="1",
    )

    # HPO-specific arguments
    parser.add_argument(
        "--n-workers", type=int, default=4,
        help="Number of parallel HPO workers (Slurm jobs) (default: 4).",
    )
    parser.add_argument(
        "--n-trials-per-worker", type=int, default=25,
        help="Number of trials each worker runs (default: 25).",
    )
    parser.add_argument(
        "--study-name", type=str, default="histogram_mlp_hpo",
        help="Optuna study name (default: histogram_mlp_hpo).",
    )
    parser.add_argument(
        "--hpo-fold", type=int, default=0,
        help="Fold ID to use for HPO validation (default: 0).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: 5 epochs per trial.",
    )
    return parser.parse_args()


def _build_hpo_extra_args(args: argparse.Namespace) -> dict[str, str | None]:
    """Build extra arguments to forward to hpo.py."""
    extra: dict[str, str | None] = {
        "--study-name": args.study_name,
        "--n-trials": str(args.n_trials_per_worker),
        "--hpo-fold": str(args.hpo_fold),
    }
    if args.debug:
        extra["--debug"] = None
    return extra


def main() -> None:
    """Main function to submit HPO workers or run locally."""
    args = cli()
    params_path = Path(args.params).resolve()

    total_trials = args.n_workers * args.n_trials_per_worker
    print(f"HPO: {args.n_workers} workers x {args.n_trials_per_worker} trials = {total_trials} total")
    print(f"Study name: {args.study_name}")

    use_local, mode = determine_execution_mode(args.local)
    print(f"Execution mode: {mode}")

    extra_args = _build_hpo_extra_args(args)

    if use_local:
        run_local(str(params_path), extra_args)
    else:
        partitions = resolve_partitions(
            args.partition, getattr(args, "partitions", None),
        )
        log_dir = setup_log_directory("histogram_hpo")
        print(f"Logs will be written to: {log_dir.absolute()}")

        run_slurm(
            params_path=str(params_path),
            extra_args=extra_args,
            n_workers=args.n_workers,
            partitions=partitions,
            log_dir=log_dir,
            time_limit=args.time,
            cpus=args.cpus,
            mem=args.mem,
            gpus=args.gpus,
            study_name=args.study_name,
        )


def run_local(
    params_path: str,
    extra_args: dict[str, str | None],
) -> None:
    """Run HPO locally (single worker)."""
    cmd = build_base_command(
        "src.models.histogram_mlp.hpo",
        params_path=params_path,
        extra_args=extra_args,
    )

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print("\n✓ HPO completed successfully")
    else:
        print("\n✗ HPO failed")
        import sys
        sys.exit(result.returncode)


def run_slurm(
    params_path: str,
    extra_args: dict[str, str | None],
    n_workers: int,
    partitions: list[str],
    log_dir: Path,
    time_limit: str,
    cpus: int,
    mem: str,
    gpus: str,
    study_name: str,
) -> None:
    """Submit parallel HPO workers to Slurm."""
    cmd_parts = build_base_command(
        "src.models.histogram_mlp.hpo",
        params_path=params_path,
        extra_args=extra_args,
    )
    command = " ".join(cmd_parts)

    job_ids = []
    for worker_id in range(n_workers):
        partition = partitions[worker_id % len(partitions)]

        slurm = Slurm(
            job_name=f"hpo_w{worker_id}_{study_name}",
            output=str(log_dir / f"%j_hpo_w{worker_id}.log"),
            error=str(log_dir / f"%j_hpo_w{worker_id}.err"),
            time=time_limit,
            cpus_per_task=cpus,
            mem=mem,
            partition=partition,
            gres=f"gpu:{gpus}",
        )

        job_id = submit_job_with_retry(slurm, command)
        job_ids.append(job_id)
        print(f"  Worker {worker_id}: job {job_id} on {partition}")

    print(f"\nSubmitted {len(job_ids)} HPO workers")
    print("Waiting for all workers to complete...")

    all_success = True
    for job_id in job_ids:
        success = wait_for_job_completion(job_id, poll_interval=30)
        if not success:
            all_success = False
            print(f"  Worker job {job_id} failed")

    if all_success:
        print(f"\n✓ All {n_workers} HPO workers completed successfully")
        print(f"  Check best_params.json in models directory")
    else:
        print("\n✗ Some HPO workers failed. Check logs:")
        print(f"  {log_dir.absolute()}/")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
