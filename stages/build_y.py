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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from simple_slurm import Slurm

# Setup environment and path
from src.pipeline.entrypoint_utils import (
    PartitionDistributor,
    add_common_args,
    add_execution_args,
    add_partition_args,
    build_base_command,
    determine_execution_mode,
    resolve_partitions,
    setup_environment,
    setup_log_directory,
    wait_for_job_completion,
)

project_root = setup_environment()

from src.conf.conf import get_config  # noqa: E402


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Process Y data per trait in parallel, then merge into Y.parquet.")
    )
    add_common_args(parser, include_partition=False)
    add_execution_args(parser, multi_job=True, n_jobs_default=4)
    add_partition_args(parser, enable_multi_partition=True)

    # Resource arguments for individual trait jobs
    parser.add_argument(
        "--time",
        type=str,
        default="00:15:00",
        help="Time limit for individual trait jobs (default: 00:15:00)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="Number of CPUs for individual trait jobs (default: 4)",
    )
    parser.add_argument(
        "--mem",
        type=str,
        default="16GB",
        help="Memory for individual trait jobs (default: 16GB)",
    )

    # Resource arguments for merge job
    parser.add_argument(
        "--merge-time",
        type=str,
        default="00:30:00",
        help="Time limit for merge job (default: 00:30:00)",
    )
    parser.add_argument(
        "--merge-cpus",
        type=int,
        default=8,
        help="Number of CPUs for merge job (default: 8)",
    )
    parser.add_argument(
        "--merge-mem",
        type=str,
        default="32GB",
        help="Memory for merge job (default: 32GB)",
    )

    # Debug mode
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode: process only the first trait.",
    )

    return parser.parse_args()


def run_trait_job(
    trait: str, params_path: str | None, overwrite: bool
) -> tuple[str, int]:
    """Run a single trait job locally."""
    cmd = build_base_command(
        "src.features.build_y_trait",
        params_path=params_path,
        overwrite=overwrite,
        extra_args={"--trait": trait},
    )

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return trait, result.returncode


def run_merge_step(params_path: str | None, overwrite: bool) -> int:
    """Run the merge step to combine all trait files."""
    cmd = build_base_command(
        "src.features.merge_y_traits", params_path=params_path, overwrite=overwrite
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
    partitions: list[str],
    log_dir: Path,
    time_limit: str = "00:15:00",
    cpus: int = 4,
    mem: str = "16GB",
    merge_time: str = "00:30:00",
    merge_cpus: int = 8,
    merge_mem: str = "32GB",
) -> None:
    """Submit trait jobs to Slurm with automatic merge via job dependencies."""
    # Initialize partition distributor for round-robin distribution
    distributor = PartitionDistributor(partitions)

    # Submit trait processing jobs
    trait_job_ids = []
    for trait in trait_names:
        # Construct the command
        cmd_parts = build_base_command(
            "src.features.build_y_trait",
            params_path=params_path,
            overwrite=overwrite,
            extra_args={"--trait": trait},
        )
        command = " ".join(cmd_parts)

        # Get next partition for round-robin distribution
        partition = distributor.get_next()

        # Create Slurm job configuration
        slurm = Slurm(
            job_name=f"y_{trait}",
            output=str(log_dir / f"%j_{trait}.log"),
            error=str(log_dir / f"%j_{trait}.err"),
            time=time_limit,
            cpus_per_task=cpus,
            mem=mem,
            partition=partition,
        )

        # Submit the job
        job_id = slurm.sbatch(command)
        trait_job_ids.append(job_id)
        print(f"Submitted job {job_id} for trait '{trait}'")

        # Add small delay to avoid overwhelming the Slurm scheduler
        time.sleep(0.5)

    print(f"\n{'=' * 60}")
    print(f"Submitted {len(trait_job_ids)} trait processing jobs")

    # Submit merge job with dependency on all trait jobs
    print(f"\n{'=' * 60}")
    print("Submitting merge job with dependencies on all trait jobs...")

    # Create dependency string: afterok:job1:job2:job3...
    dependency_str = "afterok:" + ":".join(str(jid) for jid in trait_job_ids)

    # Construct merge command
    merge_cmd_parts = build_base_command(
        "src.features.merge_y_traits", params_path=params_path, overwrite=overwrite
    )
    merge_command = " ".join(merge_cmd_parts)

    # Use first partition for merge job (could be any partition)
    merge_partition = partitions[0]

    # Create Slurm job for merge with dependencies
    merge_slurm = Slurm(
        job_name="merge_y_traits",
        output=str(log_dir / "%j_merge_y.log"),
        error=str(log_dir / "%j_merge_y.err"),
        time=merge_time,
        cpus_per_task=merge_cpus,
        mem=merge_mem,
        partition=merge_partition,
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

    success = wait_for_job_completion(merge_job_id, "merge")

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

    # Check if Y.parquet and report already exist
    out_fn = Path(cfg.train.Y.fp)
    report_fp = out_fn.parent / "report.md"
    figure_fp = out_fn.parent / "trait_distributions.png"

    y_exists = out_fn.exists()
    report_exists = report_fp.exists() and figure_fp.exists()

    # Smart overwrite logic
    if not args.overwrite and y_exists and report_exists:
        print("All outputs already exist. Use --overwrite to regenerate.")
        print(f"  Y data: {out_fn}")
        print(f"  Report: {report_fp}")
        print(f"  Figure: {figure_fp}")
        return

    if not args.overwrite and y_exists:
        print(f"Y data already exists: {out_fn}")
        print("Skipping trait processing. Will only generate report if needed.")
        # Skip to merge step which will handle report generation
        from src.features.merge_y_traits import main as merge_main

        merge_main(args)
        return

    # Get traits from configuration
    trait_names = cfg.traits.names
    print(f"Found {len(trait_names)} traits to process: {', '.join(trait_names)}")

    # In debug mode, only process the first trait
    if args.debug:
        trait_names = trait_names[:1]
        print(f"DEBUG MODE: Limited to 1 trait: {trait_names[0]}")

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
        partitions = resolve_partitions(args.partition, args.partitions)
        if len(partitions) > 1:
            print(
                f"Distributing jobs across {len(partitions)} partitions: "
                f"{', '.join(partitions)}"
            )
        print("Execution mode: Slurm with job dependencies")
        log_dir = setup_log_directory("build_y")
        print(f"Logs will be written to: {log_dir.absolute()}")
        run_slurm(
            trait_names,
            str(params_path) if params_path else None,
            args.overwrite,
            partitions,
            log_dir,
            time_limit=args.time,
            cpus=args.cpus,
            mem=args.mem,
            merge_time=args.merge_time,
            merge_cpus=args.merge_cpus,
            merge_mem=args.merge_mem,
        )


if __name__ == "__main__":
    main()
