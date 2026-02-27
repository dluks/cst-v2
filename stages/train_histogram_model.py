#!/usr/bin/env python
"""
Entry point script to train the histogram MLP model.

Submits parallel Slurm jobs for CV folds (one GPU per fold), then a
full-model job after all folds complete.  Falls back to sequential
``run_cv()`` when running locally.

Execution mode is determined by the USE_SLURM environment variable (from .env file).
Set USE_SLURM=false to use local execution by default, or use --local flag to override.
"""

import argparse
import json
import subprocess
from pathlib import Path

from simple_slurm import Slurm

from src.conf.conf import get_config
from src.models.run_utils import generate_run_id
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
        description="Train histogram MLP for trait distribution prediction."
    )
    add_common_args(parser, include_partition=False)
    add_execution_args(parser, multi_job=False, n_jobs_default=1)
    add_partition_args(parser, enable_multi_partition=True)
    add_resource_args(
        parser,
        time_default="01:30:00",
        cpus_default=4,
        mem_default="8GB",
        include_gpus=True,
        gpus_default="1",
    )

    # Training-specific arguments
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


def main() -> None:
    """Main function to submit Slurm jobs or run locally."""
    args = cli()
    params_path = Path(args.params).resolve()
    cfg = get_config(params_path=str(params_path))
    n_folds = cfg.train.n_folds

    run_id = args.run_id or generate_run_id()
    print(f"Training histogram MLP model (run: {run_id}, {n_folds} folds)")

    use_local, mode = determine_execution_mode(args.local)
    print(f"Execution mode: {mode}")

    if use_local:
        extra_args = _build_extra_args(run_id=run_id, resume=args.resume, debug=args.debug)
        run_local(str(params_path), extra_args)
    else:
        partitions = resolve_partitions(
            args.partition, getattr(args, "partitions", None),
        )
        log_dir = setup_log_directory("train_histogram_model")
        print(f"Logs will be written to: {log_dir.absolute()}")

        run_slurm(
            params_path=str(params_path),
            run_id=run_id,
            n_folds=n_folds,
            partitions=partitions,
            log_dir=log_dir,
            time_limit=args.time,
            cpus=args.cpus,
            mem=args.mem,
            gpus=args.gpus,
            exclude=args.exclude,
            debug=args.debug,
            resume=args.resume,
        )


def _build_extra_args(
    *,
    run_id: str,
    fold: int | None = None,
    full_only: bool = False,
    resume: bool = False,
    debug: bool = False,
) -> dict[str, str | None]:
    """Build extra arguments to forward to train.py."""
    extra: dict[str, str | None] = {"--run-id": run_id}
    if fold is not None:
        extra["--fold"] = str(fold)
    if full_only:
        extra["--full-only"] = None
    if resume:
        extra["--resume"] = None
    if debug:
        extra["--debug"] = None
    return extra


def run_local(
    params_path: str,
    extra_args: dict[str, str | None],
) -> None:
    """Run histogram MLP training locally (sequential folds)."""
    cmd = build_base_command(
        "src.models.histogram_mlp.train",
        params_path=params_path,
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


def _submit_job(
    command: str,
    job_name: str,
    log_dir: Path,
    partitions: list[str],
    worker_id: int,
    time_limit: str,
    cpus: int,
    mem: str,
    gpus: str,
    exclude: str | None,
) -> int:
    """Submit a single Slurm job and return its job ID."""
    partition = partitions[worker_id % len(partitions)]
    slurm_kwargs = dict(
        job_name=job_name,
        output=str(log_dir / f"%j_{job_name}.log"),
        error=str(log_dir / f"%j_{job_name}.err"),
        time=time_limit,
        cpus_per_task=cpus,
        mem=mem,
        partition=partition,
        gres=f"gpu:{gpus}",
    )
    if exclude:
        slurm_kwargs["exclude"] = exclude

    slurm = Slurm(**slurm_kwargs)
    return submit_job_with_retry(slurm, command)


def run_slurm(
    params_path: str,
    run_id: str,
    n_folds: int,
    partitions: list[str],
    log_dir: Path,
    time_limit: str,
    cpus: int,
    mem: str,
    gpus: str,
    exclude: str | None = None,
    debug: bool = False,
    resume: bool = False,
) -> None:
    """Submit parallel CV fold jobs, then full model job."""

    # ── Phase 1: Submit parallel fold jobs ──────────────────────────────
    print(f"\nSubmitting {n_folds} CV fold jobs...")
    fold_job_ids = []
    for fold_id in range(n_folds):
        extra = _build_extra_args(
            run_id=run_id, fold=fold_id, resume=resume, debug=debug,
        )
        cmd = " ".join(build_base_command(
            "src.models.histogram_mlp.train",
            params_path=params_path,
            extra_args=extra,
        ))
        job_id = _submit_job(
            cmd, f"fold_{fold_id}_{run_id}", log_dir,
            partitions, fold_id, time_limit, cpus, mem, gpus, exclude,
        )
        fold_job_ids.append(job_id)
        partition = partitions[fold_id % len(partitions)]
        print(f"  Fold {fold_id}: job {job_id} on {partition}")

    # Wait for all fold jobs
    print(f"\nWaiting for {n_folds} fold jobs to complete...")
    all_folds_ok = True
    for job_id in fold_job_ids:
        if not wait_for_job_completion(job_id, poll_interval=30):
            all_folds_ok = False
            print(f"  Fold job {job_id} failed")

    if not all_folds_ok:
        print("\n✗ Some fold jobs failed. Check logs:")
        print(f"  {log_dir.absolute()}/")
        import sys
        sys.exit(1)

    print(f"✓ All {n_folds} fold jobs completed")

    # ── Phase 2: Aggregate CV metrics ───────────────────────────────────
    cfg = get_config(params_path=params_path)
    models_base = Path(project_root) / cfg.models.dir_fp / "training"
    run_dir = models_base / run_id
    cv_dir = run_dir / "cv"

    print("\nAggregating CV metrics...")
    from src.models.histogram_mlp.train import (
        aggregate_cv_metrics,
        generate_performance_summary,
    )

    fold_metrics = []
    for fold_id in range(n_folds):
        metrics_path = cv_dir / f"fold_{fold_id}" / "fold_metrics.json"
        with open(metrics_path) as f:
            fold_metrics.append(json.load(f))

    summary = aggregate_cv_metrics(fold_metrics)
    with open(cv_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  KL={summary.get('kl_divergence_mean', float('nan')):.6f} "
          f"(baseline={summary.get('baseline_kl_divergence_mean', float('nan')):.6f}), "
          f"CRPS={summary.get('crps_mean', float('nan')):.6f}, "
          f"HI={summary.get('histogram_intersection_mean', float('nan')):.4f}")
    generate_performance_summary(summary, cv_dir / "performance_summary.csv")

    # ── Phase 3: Submit full model job ──────────────────────────────────
    print("\nSubmitting full model job...")
    extra = _build_extra_args(
        run_id=run_id, full_only=True, resume=resume, debug=debug,
    )
    cmd = " ".join(build_base_command(
        "src.models.histogram_mlp.train",
        params_path=params_path,
        extra_args=extra,
    ))
    full_job_id = _submit_job(
        cmd, f"full_{run_id}", log_dir,
        partitions, 0, time_limit, cpus, mem, gpus, exclude,
    )
    print(f"  Full model: job {full_job_id}")

    print(f"\nWaiting for full model job {full_job_id}...")
    if wait_for_job_completion(full_job_id, poll_interval=30):
        print(f"\n✓ Training complete: {run_dir}")
    else:
        print(f"\n✗ Full model job {full_job_id} failed. Check logs:")
        print(f"  {log_dir.absolute()}/")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
