#!/usr/bin/env python
"""
Entry point script to harmonize EO data either locally or on Slurm.

This script provides a unified interface to run the harmonization pipeline
either on the local machine or submit it as a Slurm job.

Execution mode is determined by the USE_SLURM environment variable (from .env file).
Set USE_SLURM=false to use local execution by default, or use --local flag to override.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from simple_slurm import Slurm

# Load environment variables from .env file
load_dotenv()

# Add project root to path to import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Harmonize EO data locally or on Slurm."
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
        "--dry-run",
        action="store_true",
        help="Dry run without writing output files.",
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
        default="04:00:00",
        help="Time limit for Slurm job (default: 04:00:00).",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=16,
        help="Number of CPUs for Slurm job (default: 16).",
    )
    parser.add_argument(
        "--mem",
        type=str,
        default="64GB",
        help="Memory for Slurm job (default: 64GB).",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to run harmonization locally or on Slurm."""
    args = cli()

    # Convert paths to absolute if provided
    params_path = str(Path(args.params).absolute()) if args.params else None
    sys_params_path = str(Path(args.sys_params).absolute()) if args.sys_params else None

    # Determine execution mode
    # Check environment variable USE_SLURM (default to False if not set)
    use_slurm_env = os.getenv("USE_SLURM", "false").lower() in ("true", "1", "yes", "t")

    # Command-line flag --local overrides environment variable
    use_local = args.local or not use_slurm_env

    if use_local:
        # Run locally
        mode = "local (--local flag)" if args.local else "local (USE_SLURM=false)"
        print(f"Execution mode: {mode}")
        run_local(params_path, sys_params_path, args.dry_run, args.overwrite)
    else:
        # Submit to Slurm
        print("Execution mode: Slurm")
        run_slurm(
            params_path,
            sys_params_path,
            args.dry_run,
            args.overwrite,
            args.partition,
            args.time,
            args.cpus,
            args.mem,
            wait=not args.no_wait,
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


def run_local(
    params_path: str | None,
    sys_params_path: str | None,
    dry_run: bool,
    overwrite: bool,
) -> None:
    """Run harmonization locally."""
    print("\nRunning harmonization locally...")

    cmd = ["python", "-m", "src.data.harmonize_eo_data"]
    if params_path:
        cmd.extend(["--params", params_path])
    if sys_params_path:
        cmd.extend(["--sys_params", sys_params_path])
    if dry_run:
        cmd.append("--dry-run")
    if overwrite:
        cmd.append("--overwrite")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print("\n✓ Harmonization completed successfully!")
    else:
        print(f"\n✗ Harmonization failed (exit code: {result.returncode})")
        sys.exit(1)


def run_slurm(
    params_path: str | None,
    sys_params_path: str | None,
    dry_run: bool,
    overwrite: bool,
    partition: str,
    time_limit: str,
    cpus: int,
    mem: str,
    wait: bool,
) -> None:
    """Submit harmonization job to Slurm."""
    print("\nSubmitting harmonization job to Slurm...")

    # Create log directory
    log_dir = Path("logs/harmonize_eo_data")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Construct command
    cmd_parts = ["python", "-m", "src.data.harmonize_eo_data"]
    if params_path:
        cmd_parts.extend(["--params", params_path])
    if sys_params_path:
        cmd_parts.extend(["--sys_params", sys_params_path])
    if dry_run:
        cmd_parts.append("--dry-run")
    if overwrite:
        cmd_parts.append("--overwrite")
    command = " ".join(cmd_parts)

    # Create Slurm job configuration
    slurm = Slurm(
        job_name="harmonize_eo",
        output=str(log_dir / "%j_harmonize_eo.log"),
        error=str(log_dir / "%j_harmonize_eo.err"),
        time=time_limit,
        cpus_per_task=cpus,
        mem=mem,
        partition=partition,
    )

    # Submit the job
    job_id = slurm.sbatch(command)
    print(f"Submitted job {job_id}")
    print(f"Logs will be written to: {log_dir.absolute()}")
    print(f"Monitor progress: tail -f {log_dir.absolute()}/{job_id}_harmonize_eo.log")

    if wait:
        # Wait for job to complete
        success = wait_for_job_completion(job_id, poll_interval=30)

        if success:
            print("\n✓ Harmonization completed successfully!")
        else:
            print("\n✗ Job failed. Check logs:")
            print(f"  {log_dir.absolute()}/{job_id}_harmonize_eo.err")
            sys.exit(1)
    else:
        print("\nJob submitted. Not waiting for completion (--no-wait flag set).")


if __name__ == "__main__":
    main()
