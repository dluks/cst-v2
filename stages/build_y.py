#!/usr/bin/env python
"""
Entry point script to process Y data per trait in parallel.

This script reads trait names from the configuration file and submits a separate
job for each trait to process it in parallel (via Slurm or local execution),
then merges all results into a single Y.parquet file.

In Slurm mode, uses job dependencies to automatically run the merge step after
all trait jobs complete, and waits for final completion before returning.

Execution mode is determined by the USE_SLURM environment variable (from .env file).
Set USE_SLURM=false to use local execution by default, or use --local flag to override.
"""

import argparse
import subprocess
import sys
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
    wait_for_job_completion,
)

project_root = setup_environment()

from src.conf.conf import get_config  # noqa: E402


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Process Y data per trait in parallel, then merge into Y.parquet.")
    )
    add_common_args(parser)
    add_execution_args(parser, multi_job=True, n_jobs_default=4)
    return parser.parse_args()


def run_trait_job(
    trait: str, params_path: str | None, overwrite: bool
) -> tuple[str, int]:
    """Run a single trait job locally."""
    cmd = build_base_command(
        "src.features.build_y_trait",
        params_path=params_path,
        overwrite=overwrite,
        extra_args={"--trait": trait}
    )

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return trait, result.returncode


def run_merge_step(params_path: str | None, overwrite: bool) -> int:
    """Run the merge step to combine all trait files."""
    cmd = build_base_command(
        "src.features.merge_y_traits",
        params_path=params_path,
        overwrite=overwrite
    )

    print(f"\nRunning merge step: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


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
        sys.exit(1)
    else:
        print("All traits processed successfully!")

    # Run merge step
    print(f"\n{'=' * 60}")
    print("Running merge step...")
    merge_returncode = run_merge_step(params_path, overwrite)
    if merge_returncode == 0:
        print("✓ Merge completed successfully!")
    else:
        print(f"✗ Merge failed (exit code: {merge_returncode})")
        sys.exit(1)


def run_slurm(
    trait_names: list[str],
    params_path: str | None,
    overwrite: bool,
    partition: str,
    log_dir: Path,
) -> None:
    """Submit trait jobs to Slurm with automatic merge via job dependencies."""
    # Submit trait processing jobs
    trait_job_ids = []
    for trait in trait_names:
        # Construct the command
        cmd_parts = build_base_command(
            "src.features.build_y_trait",
            params_path=params_path,
            overwrite=overwrite,
            extra_args={"--trait": trait}
        )
        command = " ".join(cmd_parts)

        # Create Slurm job configuration
        slurm = Slurm(
            job_name=f"y_{trait}",
            output=str(log_dir / f"%j_{trait}.log"),
            error=str(log_dir / f"%j_{trait}.err"),
            time="00:15:00",
            cpus_per_task=4,
            mem="16GB",
            partition=partition,
        )

        # Submit the job
        job_id = slurm.sbatch(command)
        trait_job_ids.append(job_id)
        print(f"Submitted job {job_id} for trait '{trait}'")

    print(f"\n{'=' * 60}")
    print(f"Submitted {len(trait_job_ids)} trait processing jobs")

    # Submit merge job with dependency on all trait jobs
    print(f"\n{'=' * 60}")
    print("Submitting merge job with dependencies on all trait jobs...")

    # Create dependency string: afterok:job1:job2:job3...
    dependency_str = "afterok:" + ":".join(str(jid) for jid in trait_job_ids)

    # Construct merge command
    merge_cmd_parts = build_base_command(
        "src.features.merge_y_traits",
        params_path=params_path,
        overwrite=overwrite
    )
    merge_command = " ".join(merge_cmd_parts)

    # Create Slurm job for merge with dependencies
    merge_slurm = Slurm(
        job_name="merge_y_traits",
        output=str(log_dir / "%j_merge_y.log"),
        error=str(log_dir / "%j_merge_y.err"),
        time="00:30:00",
        cpus_per_task=8,
        mem="32GB",
        partition=partition,
        dependency=dependency_str,
    )

    merge_job_id = merge_slurm.sbatch(merge_command)
    num_deps = len(trait_job_ids)
    print(f"Submitted merge job {merge_job_id} (depends on: {num_deps} trait jobs)")
    print(f"Logs will be written to: {log_dir.absolute()}")

    # Wait for merge job to complete
    print(f"\n{'=' * 60}")
    print("Waiting for all jobs to complete...")
    print("Merge job will start automatically after all trait jobs finish.")
    merge_log = f"{log_dir.absolute()}/{merge_job_id}_merge_y.log"
    print(f"Monitor progress: tail -f {merge_log}")

    success = wait_for_job_completion(merge_job_id, "merge", poll_interval=30)

    if success:
        print(f"\n{'=' * 60}")
        print("✓ All jobs completed successfully!")
        print("Y.parquet is ready.")
    else:
        print(f"\n{'=' * 60}")
        print("✗ Merge job failed. Check logs:")
        print(f"  {log_dir.absolute()}/{merge_job_id}_merge_y.err")
        sys.exit(1)


def main() -> None:
    """Main function to submit jobs or run locally for each trait."""
    args = cli()
    params_path = Path(args.params).absolute() if args.params else None
    cfg = get_config(params_path=params_path)

    # Get traits from configuration
    trait_names = cfg.traits.names
    print(f"Found {len(trait_names)} traits to process: {', '.join(trait_names)}")

    # Determine execution mode
    use_local, mode = determine_execution_mode(args.local)

    if use_local:
        # Run locally in parallel
        print(f"Execution mode: {mode}")
        run_local(
            trait_names,
            str(params_path) if params_path else None,
            args.overwrite,
            args.n_jobs,
        )
    else:
        # Submit to Slurm
        print("Execution mode: Slurm with job dependencies")
        log_dir = setup_log_directory("build_y")
        print(f"Logs will be written to: {log_dir.absolute()}")
        run_slurm(
            trait_names,
            str(params_path) if params_path else None,
            args.overwrite,
            args.partition,
            log_dir,
        )


if __name__ == "__main__":
    main()
