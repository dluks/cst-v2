#!/usr/bin/env python
"""
Entry point script to calculate spatial autocorrelation either locally or on Slurm.

This script provides a unified interface to run the spatial autocorrelation calculation
either on the local machine or submit it as a Slurm job.

Execution mode is determined by the USE_SLURM environment variable (from .env file).
Set USE_SLURM=false to use local execution by default, or use --local flag to override.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from dotenv import load_dotenv
from simple_slurm import Slurm

from src.conf.conf import get_config

# Load environment variables from .env file
load_dotenv()

# Add project root to path to import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate spatial autocorrelation locally or on Slurm."
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        help="Path to parameters file.",
    )
    parser.add_argument(
        "-s",
        "--sys-params",
        type=str,
        help="Path to system parameters file.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode.",
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
        help="Slurm partition to use (default: main).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Force local execution instead of Slurm (overrides USE_SLURM env variable)."
        ),
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for Slurm job to complete (submit and exit).",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="06:00:00",
        help="Time limit for Slurm job (default: 06:00:00).",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=8,
        help="Number of CPUs for Slurm job (default: 8).",
    )
    parser.add_argument(
        "--mem",
        type=str,
        default="64GB",
        help="Memory for Slurm job (default: 64GB).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="1",
        help="Number of GPUs for Slurm job (default: 1).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help=(
            "Maximum number of traits to process in parallel (default: 4). "
            "Only applies to local execution."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run spatial autocorrelation calculation locally or on Slurm."""
    args = cli()

    # Convert paths to absolute if provided
    params_path = str(Path(args.params).absolute()) if args.params else None
    sys_params_path = str(Path(args.sys_params).absolute()) if args.sys_params else None

    # Set CONFIG_PATH BEFORE importing dataset_utils to avoid module-level load error
    if params_path is not None:
        os.environ["CONFIG_PATH"] = params_path

    # Import after setting CONFIG_PATH
    from src.utils.dataset_utils import get_y_fn

    # Determine execution mode
    # Check environment variable USE_SLURM (default to False if not set)
    use_slurm_env = os.getenv("USE_SLURM", "false").lower() in ("true", "1", "yes", "t")

    # Command-line flag --local overrides environment variable
    use_local = args.local or not use_slurm_env
    ranges_fp = Path(
        get_config(params_path=params_path).spatial_autocorr.ranges_fp
    ).resolve()

    traits = (
        dd.read_parquet(get_y_fn(get_config(params_path=params_path)))
        .columns.difference(["x", "y", "source"])
        .to_list()
    )
    print(f"Found {len(traits)} traits to process: {', '.join(traits)}")

    if use_local:
        # Run locally
        mode = "local (--local flag)" if args.local else "local (USE_SLURM=false)"
        print(f"Execution mode: {mode}")
        run_local(
            params_path,
            sys_params_path,
            args.debug,
            args.overwrite,
            traits,
            args.max_parallel,
            ranges_fp,
        )
    else:
        # Submit to Slurm
        print("Execution mode: Slurm")
        run_slurm(
            params_path,
            sys_params_path,
            args.debug,
            args.overwrite,
            args.partition,
            args.time,
            args.cpus,
            args.mem,
            args.gpus,
            wait=not args.no_wait,
            traits=traits,
            ranges_fp=ranges_fp,
        )


def check_job_status(job_id: int | str) -> str:
    """
    Check the status of a Slurm job.

    Args:
        job_id: Slurm job ID

    Returns:
        Job state: 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'NOT_FOUND'
    """
    try:
        result = subprocess.run(
            ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

        # Job not in queue - check sacct for completed/failed jobs
        result = subprocess.run(
            ["sacct", "-j", str(job_id), "-n", "-X", "-o", "State"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            state = result.stdout.strip().split()[0]
            return state

        return "NOT_FOUND"

    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"Warning: Could not check job status: {e}")
        return "UNKNOWN"


def wait_for_job_completion(job_id: int | str, poll_interval: int = 30) -> bool:
    """
    Wait for a Slurm job to complete.

    Args:
        job_id: Slurm job ID
        poll_interval: Seconds to wait between status checks

    Returns:
        True if job completed successfully, False otherwise
    """
    print(f"\nWaiting for job {job_id} to complete...")
    print(f"Checking status every {poll_interval} seconds...")

    start_time = time.time()
    last_status = None

    while True:
        status = check_job_status(job_id)

        if status != last_status:
            elapsed = int(time.time() - start_time)
            print(f"[{elapsed}s] Job {job_id} status: {status}")
            last_status = status

        if status in ["COMPLETED"]:
            print(f"✓ Job {job_id} completed successfully!")
            return True
        elif status in [
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
        ]:
            print(f"✗ Job {job_id} failed with status: {status}")
            return False
        elif status in ["PENDING", "RUNNING", "CONFIGURING"]:
            # Job still running, continue waiting
            time.sleep(poll_interval)
        else:
            # Unknown or not found - could mean it completed before polling started
            time.sleep(5)
            status = check_job_status(job_id)
            if status == "NOT_FOUND":
                print(
                    f"Warning: Job {job_id} not found in queue. "
                    "May have already completed."
                )
                return True
            time.sleep(poll_interval)


def combine_results(temp_dir: Path, output_fp: Path, cfg, traits: list[str]) -> None:
    """
    Combine individual trait results into a single output file.

    Args:
        temp_dir: Directory containing individual trait parquet files
        output_fp: Final output file path
        cfg: Configuration object
        traits: List of trait names
    """
    print("\nCombining results from all traits...")

    # Read all individual trait results
    trait_dfs = []
    missing_traits = []

    for trait in traits:
        trait_fp = temp_dir / f"spatial_autocorr_{trait}.parquet"
        if trait_fp.exists():
            trait_dfs.append(pd.read_parquet(trait_fp))
        else:
            missing_traits.append(trait)

    if missing_traits:
        print(f"Warning: Missing results for traits: {', '.join(missing_traits)}")

    if not trait_dfs:
        print("Error: No trait results found to combine!")
        sys.exit(1)

    # Combine all results
    ranges_df = pd.concat(trait_dfs, ignore_index=True)

    # Save to final output location
    print(f"Saving combined results to {output_fp}...")
    if output_fp.exists():
        print("Overwriting existing output file...")
        output_fp.unlink()

    output_fp.parent.mkdir(parents=True, exist_ok=True)
    ranges_df.to_parquet(output_fp)

    print(f"✓ Combined results saved to {output_fp}")

    # Clean up temporary files
    print("Cleaning up temporary files...")
    for trait in traits:
        trait_fp = temp_dir / f"spatial_autocorr_{trait}.parquet"
        if trait_fp.exists():
            trait_fp.unlink()

    # Remove temp directory if empty
    try:
        temp_dir.rmdir()
        print(f"✓ Removed temporary directory: {temp_dir}")
    except OSError:
        print(f"Note: Temporary directory not empty: {temp_dir}")


def run_single_trait(
    trait: str,
    params_path: str | None,
    sys_params_path: str | None,
    debug: bool,
    output_dir: Path,
) -> tuple[str, bool]:
    """
    Run spatial autocorrelation calculation for a single trait.

    Args:
        trait: Trait name
        params_path: Path to parameters file
        sys_params_path: Path to system parameters file
        debug: Enable debug mode
        output_dir: Output directory for results

    Returns:
        Tuple of (trait_name, success)
    """
    cmd = [
        "python",
        "-m",
        "src.features.calc_spatial_autocorr_gpu",
        "--trait",
        trait,
        "--output-dir",
        str(output_dir),
    ]
    if params_path:
        cmd.extend(["--params", params_path])
    if sys_params_path:
        cmd.extend(["--sys_params", sys_params_path])
    if debug:
        cmd.append("--debug")

    # Always overwrite when processing individual traits
    cmd.append("--overwrite")

    # Set CONFIG_PATH in subprocess environment to avoid import-time errors
    env = os.environ.copy()
    if params_path:
        env["CONFIG_PATH"] = params_path

    print(f"  Starting: {trait}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode == 0:
        print(f"  ✓ Completed: {trait}")
        return (trait, True)
    else:
        print(f"  ✗ Failed: {trait}")
        print(f"    Exit code: {result.returncode}")
        if result.stderr:
            # Print last 20 lines of stderr for better error visibility
            stderr_lines = result.stderr.strip().split('\n')
            print(f"    Last {min(20, len(stderr_lines))} lines of stderr:")
            for line in stderr_lines[-20:]:
                print(f"      {line}")
        if result.stdout:
            # Also check stdout for error messages
            stdout_lines = result.stdout.strip().split('\n')
            if stdout_lines:
                print(f"    Last {min(5, len(stdout_lines))} lines of stdout:")
                for line in stdout_lines[-5:]:
                    print(f"      {line}")
        return (trait, False)


def run_local(
    params_path: str | None,
    sys_params_path: str | None,
    debug: bool,
    overwrite: bool,
    traits: list[str],
    max_parallel: int,
    ranges_fp: Path,
) -> None:
    """Run spatial autocorrelation calculation locally with parallel processing."""
    print(
        f"\nRunning spatial autocorrelation calculation locally "
        f"(max {max_parallel} traits in parallel)..."
    )

    # Get configuration
    cfg = get_config(params_path=params_path)

    # Check if output already exists
    if ranges_fp.exists() and not overwrite:
        print(f"Output file already exists: {ranges_fp}")
        print("Use --overwrite flag to overwrite existing files.")
        sys.exit(0)

    # Create temporary directory for individual trait results
    temp_dir = ranges_fp.parent / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Process traits in parallel
    print(f"\nProcessing {len(traits)} traits with up to {max_parallel} workers...")
    successful = []
    failed = []

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all trait jobs
        futures = {
            executor.submit(
                run_single_trait, trait, params_path, sys_params_path, debug, temp_dir
            ): trait
            for trait in traits
        }

        # Wait for completion
        for future in as_completed(futures):
            trait_name, success = future.result()
            if success:
                successful.append(trait_name)
            else:
                failed.append(trait_name)

    # Report results
    print(f"\n{'=' * 60}")
    print(f"Completed: {len(successful)}/{len(traits)} traits")
    if failed:
        print(f"Failed: {len(failed)} traits: {', '.join(failed)}")
        print("✗ Some traits failed to process")
        sys.exit(1)

    # Combine results
    combine_results(temp_dir, ranges_fp, cfg, successful)

    print("\n✓ Spatial autocorrelation calculation completed successfully!")


def run_slurm(
    params_path: str | None,
    sys_params_path: str | None,
    debug: bool,
    overwrite: bool,
    partition: str,
    time_limit: str,
    cpus: int,
    mem: str,
    gpus: str,
    wait: bool,
    traits: list[str],
    ranges_fp: Path,
) -> None:
    """Submit separate Slurm jobs for each trait."""
    print(f"\nSubmitting {len(traits)} Slurm jobs (one per trait)...")

    # Get configuration
    cfg = get_config(params_path=params_path)

    # Check if output already exists
    if ranges_fp.exists() and not overwrite:
        print(f"Output file already exists: {ranges_fp}")
        print("Use --overwrite flag to overwrite existing files.")
        sys.exit(0)

    # Create log directory
    log_dir = Path("logs/calc_spatial_autocorr")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for individual trait results
    temp_dir = ranges_fp.parent / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Submit a job for each trait
    job_ids = []
    trait_to_job = {}

    for trait in traits:
        # Construct command for this trait
        cmd_parts = [
            "python",
            "-m",
            "src.features.calc_spatial_autocorr_gpu",
            "--trait",
            trait,
            "--output-dir",
            str(temp_dir),
            "--overwrite",
        ]
        if params_path:
            cmd_parts.extend(["--params", params_path])
        if sys_params_path:
            cmd_parts.extend(["--sys_params", sys_params_path])
        if debug:
            cmd_parts.append("--debug")
        command = " ".join(cmd_parts)

        # Create Slurm job configuration with GPU support
        slurm = Slurm(
            job_name=f"autocorr_{trait}",
            output=str(log_dir / f"%j_{trait}.log"),
            error=str(log_dir / f"%j_{trait}.err"),
            time=time_limit,
            cpus_per_task=cpus,
            mem=mem,
            partition=partition,
            gres=f"gpu:{gpus}",
        )

        # Submit the job
        job_id = slurm.sbatch(command)
        job_ids.append(job_id)
        trait_to_job[job_id] = trait
        print(f"  Submitted job {job_id} for trait: {trait}")

    print(f"\nSubmitted {len(job_ids)} jobs")
    print(f"Logs directory: {log_dir.absolute()}")

    if wait:
        # Wait for all jobs to complete
        print(f"\nWaiting for {len(job_ids)} jobs to complete...")
        successful = []
        failed = []

        for job_id in job_ids:
            trait = trait_to_job[job_id]
            success = wait_for_job_completion(job_id, poll_interval=30)

            if success:
                successful.append(trait)
            else:
                failed.append(trait)
                print(f"✗ Job {job_id} ({trait}) failed. Check logs:")
                print(f"  {log_dir.absolute()}/{job_id}_{trait}.err")

        # Report results
        print(f"\n{'=' * 60}")
        print(f"Completed: {len(successful)}/{len(traits)} traits")
        if failed:
            print(f"Failed: {len(failed)} traits: {', '.join(failed)}")
            print("✗ Some jobs failed")
            sys.exit(1)

        # Combine results
        combine_results(temp_dir, ranges_fp, cfg, successful)

        print("\n✓ Spatial autocorrelation calculation completed successfully!")
    else:
        print("\nJobs submitted. Not waiting for completion (--no-wait flag set).")
        print("To combine results later, run this script again with --wait flag.")


if __name__ == "__main__":
    main()
