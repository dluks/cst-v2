#!/usr/bin/env python
"""
Entry point script to generate Slurm jobs for building GBIF trait maps.

This script reads trait names from the configuration file and submits a separate
Slurm job for each trait to process it in parallel, or runs them locally in parallel
if Slurm is unavailable.

Execution mode is determined by the USE_SLURM environment variable (from .env file).
Set USE_SLURM=false to use local execution by default, or use --local flag to override.
"""

import argparse
import subprocess
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
)

project_root = setup_environment()

from src.conf.conf import get_config  # noqa: E402


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Submit Slurm jobs to build GBIF trait maps for all configured traits."
        )
    )
    add_common_args(parser)
    add_execution_args(parser, multi_job=True, n_jobs_default=4)
    return parser.parse_args()


def run_trait_job(
    trait: str, params_path: str | None, overwrite: bool
) -> tuple[str, int]:
    """Run a single trait job locally."""
    cmd = build_base_command(
        "src.data.build_gbif_map",
        params_path=params_path,
        overwrite=overwrite,
        extra_args={"--trait": trait}
    )

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return trait, result.returncode


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
    else:
        print("All traits processed successfully!")


def run_slurm(
    trait_names: list[str],
    params_path: str | None,
    overwrite: bool,
    partition: str,
    log_dir: Path,
) -> None:
    """Submit trait jobs to Slurm."""
    job_ids = []
    for trait in trait_names:
        # Construct the command
        cmd_parts = build_base_command(
            "src.data.build_gbif_map",
            params_path=params_path,
            overwrite=overwrite,
            extra_args={"--trait": trait}
        )
        command = " ".join(cmd_parts)

        # Create Slurm job configuration
        slurm = Slurm(
            job_name=f"gbif_{trait}",
            output=str(log_dir / f"%j_{trait}.log"),
            error=str(log_dir / f"%j_{trait}.err"),
            time="00:10:00",
            cpus_per_task=10,
            mem="20GB",
            partition=partition,
        )

        # Submit the job
        job_id = slurm.sbatch(command)
        job_ids.append((trait, job_id))
        print(f"Submitted job {job_id} for trait '{trait}'")

    print(f"\nSubmitted {len(job_ids)} jobs successfully")
    print("Job IDs and traits:")
    for trait, job_id in job_ids:
        print(f"  {job_id}: {trait}")


def main() -> None:
    """Main function to submit Slurm jobs or run locally for each trait."""
    args = cli()
    params_path = Path(args.params).absolute()
    cfg = get_config(params_path=params_path)

    # Get traits from configuration
    trait_names = cfg.traits.names
    print(f"Found {len(trait_names)} traits to process: {', '.join(trait_names)}")

    # Determine execution mode
    use_local, mode = determine_execution_mode(args.local)
    print(f"Execution mode: {mode}")

    if use_local:
        # Run locally in parallel
        run_local(trait_names, str(params_path), args.overwrite, args.n_jobs)
    else:
        # Submit to Slurm
        log_dir = setup_log_directory("build_gbif_maps")
        print(f"Logs will be written to: {log_dir.absolute()}")
        run_slurm(
            trait_names, str(params_path), args.overwrite, args.partition, log_dir
        )


if __name__ == "__main__":
    main()
