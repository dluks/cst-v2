"""
Entry point script to generate Slurm jobs for building GBIF trait maps.

This script reads trait names from the configuration file and submits a separate
Slurm job for each trait to process it in parallel, or runs them locally in parallel
if Slurm is unavailable.

Execution mode is determined by the USE_SLURM environment variable (from .env file).
Set USE_SLURM=false to use local execution by default, or use --local flag to override.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from simple_slurm import Slurm

# Load environment variables from .env file
load_dotenv()

# Add project root to path to import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.conf.conf import get_config  # noqa: E402


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Submit Slurm jobs to build GBIF trait maps for all configured traits."
        )
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        help="Path to parameters file.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="main",
        help="Slurm partition to use.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Force local parallel execution instead of Slurm "
            "(overrides USE_SLURM env variable)."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of parallel jobs to run locally (default: 4).",
    )
    return parser.parse_args()


def run_trait_job(
    trait: str, params_path: str | None, overwrite: bool
) -> tuple[str, int]:
    """Run a single trait job locally."""
    cmd = ["python", "-m", "src.data.build_gbif_map", "--trait", trait]
    if params_path:
        cmd.extend(["--params", params_path])
    if overwrite:
        cmd.append("--overwrite")

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
        cmd_parts = ["python", "-m", "src.data.build_gbif_map"]
        if params_path:
            cmd_parts.extend(["--params", params_path])
        if overwrite:
            cmd_parts.append("--overwrite")
        cmd_parts.extend(["--trait", trait])
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
    cfg = get_config(params_path=args.params)

    # Get traits from configuration
    trait_names = cfg.traits.names
    print(f"Found {len(trait_names)} traits to process: {', '.join(trait_names)}")

    # Determine execution mode
    # Check environment variable USE_SLURM (default to False if not set)
    use_slurm_env = os.getenv("USE_SLURM", "false").lower() in ("true", "1", "yes", "t")

    # Command-line flag --local overrides environment variable
    use_local = args.local or not use_slurm_env

    if use_local:
        # Run locally in parallel
        mode = "local (--local flag)" if args.local else "local (USE_SLURM=false)"
        print(f"Execution mode: {mode}")
        run_local(trait_names, args.params, args.overwrite, args.n_jobs)
    else:
        # Submit to Slurm
        print("Execution mode: Slurm")
        log_dir = Path("logs/build_gbif_maps")
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logs will be written to: {log_dir.absolute()}")
        run_slurm(trait_names, args.params, args.overwrite, args.partition, log_dir)


if __name__ == "__main__":
    main()
