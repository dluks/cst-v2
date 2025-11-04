"""
Utility functions for entry point scripts.

This module provides common functionality for stage entry points including:
- Environment setup and execution mode determination
- Slurm job management (status checking, waiting)
- Command building and argument parsing helpers
- Log directory management
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Any

from dotenv import load_dotenv


# ====================
# Environment & Setup
# ====================

def setup_environment() -> Path:
    """
    Load environment variables and setup Python path.

    Returns:
        Path to project root directory
    """
    load_dotenv()

    # Get project root (assuming this file is in src/pipeline/)
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    return project_root


def determine_execution_mode(local_flag: bool) -> tuple[bool, str]:
    """
    Determine execution mode based on environment variable and CLI flag.

    Args:
        local_flag: Value of --local command-line flag

    Returns:
        Tuple of (use_local, mode_description)
        - use_local: True if should run locally, False if should use Slurm
        - mode_description: Human-readable description of the mode
    """
    use_slurm_env = os.getenv("USE_SLURM", "false").lower() in ("true", "1", "yes", "t")
    use_local = local_flag or not use_slurm_env

    if use_local:
        mode = "local (--local flag)" if local_flag else "local (USE_SLURM=false)"
    else:
        mode = "Slurm"

    return use_local, mode


# ====================
# Slurm Job Management
# ====================

def check_job_status(job_id: int | str) -> str:
    """
    Check the status of a Slurm job using squeue and sacct.

    Args:
        job_id: Slurm job ID

    Returns:
        Job state: 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED',
        'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'NOT_FOUND', or 'UNKNOWN'
    """
    try:
        # Try squeue first (for running/pending jobs)
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


def wait_for_job_completion(
    job_id: int | str,
    job_name: str = "",
    poll_interval: int = 30
) -> bool:
    """
    Wait for a Slurm job to complete by polling its status.

    Args:
        job_id: Slurm job ID
        job_name: Optional name of the job (for logging)
        poll_interval: Seconds to wait between status checks

    Returns:
        True if job completed successfully, False otherwise
    """
    job_desc = f"{job_name} (job {job_id})" if job_name else f"job {job_id}"
    print(f"\nWaiting for {job_desc} to complete...")
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


def setup_log_directory(stage_name: str) -> Path:
    """
    Create log directory for a stage and return its path.

    Args:
        stage_name: Name of the stage (e.g., "build_gbif_maps")

    Returns:
        Path to the log directory
    """
    log_dir = Path("logs") / stage_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# ====================
# Command Building
# ====================

def build_base_command(
    module_path: str,
    params_path: str | None = None,
    overwrite: bool = False,
    extra_args: dict[str, Any] | None = None
) -> list[str]:
    """
    Build a base command for running a Python module with common arguments.

    Args:
        module_path: Python module path (e.g., "src.data.build_gbif_map")
        params_path: Optional path to parameters file
        overwrite: Whether to include --overwrite flag
        extra_args: Optional dictionary of additional arguments.
                   Format: {"--arg-name": value} or {"--arg-name": None} for flags

    Returns:
        List of command parts ready to be joined or passed to subprocess

    Examples:
        >>> build_base_command("src.data.build_gbif_map", "/path/to/params.yaml", True)
        ['python', '-m', 'src.data.build_gbif_map', '--params', '/path/to/params.yaml', '--overwrite']

        >>> build_base_command("src.data.harmonize", extra_args={"--dry-run": None, "--cpus": 4})
        ['python', '-m', 'src.data.harmonize', '--dry-run', '--cpus', '4']
    """
    cmd = ["python", "-m", module_path]

    if params_path:
        cmd.extend(["--params", params_path])

    if overwrite:
        cmd.append("--overwrite")

    if extra_args:
        for arg_name, arg_value in extra_args.items():
            if arg_value is None:
                # Flag without value
                cmd.append(arg_name)
            else:
                # Argument with value
                cmd.extend([arg_name, str(arg_value)])

    return cmd


def format_command_string(cmd_parts: list[str]) -> str:
    """
    Format command parts into a string suitable for execution.

    Args:
        cmd_parts: List of command parts

    Returns:
        Space-joined command string
    """
    return " ".join(cmd_parts)


# ====================
# Argument Parser Helpers
# ====================

def add_common_args(
    parser: argparse.ArgumentParser,
    include_partition: bool = True
) -> None:
    """
    Add common arguments to an argument parser.

    Common arguments include:
    - -p/--params: Path to parameters file
    - -o/--overwrite: Overwrite existing files
    - --partition: Slurm partition (optional, controlled by include_partition)
    - --local: Force local execution

    Args:
        parser: ArgumentParser instance to add arguments to
        include_partition: If True, adds --partition argument. Set to False
            when using add_partition_args() instead.
    """
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
    if include_partition:
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
            "Force local execution instead of Slurm "
            "(overrides USE_SLURM env variable)."
        ),
    )


def add_resource_args(
    parser: argparse.ArgumentParser,
    time_default: str = "01:00:00",
    cpus_default: int = 4,
    mem_default: str = "16GB",
    include_gpus: bool = False,
    gpus_default: str = "1"
) -> None:
    """
    Add resource specification arguments to an argument parser.

    Args:
        parser: ArgumentParser instance to add arguments to
        time_default: Default time limit
        cpus_default: Default number of CPUs
        mem_default: Default memory
        include_gpus: Whether to include GPU argument
        gpus_default: Default number of GPUs (only used if include_gpus=True)
    """
    parser.add_argument(
        "--time",
        type=str,
        default=time_default,
        help=f"Time limit for Slurm job (default: {time_default}).",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=cpus_default,
        help=f"Number of CPUs for Slurm job (default: {cpus_default}).",
    )
    parser.add_argument(
        "--mem",
        type=str,
        default=mem_default,
        help=f"Memory for Slurm job (default: {mem_default}).",
    )

    if include_gpus:
        parser.add_argument(
            "--gpus",
            type=str,
            default=gpus_default,
            help=f"Number of GPUs for Slurm job (default: {gpus_default}).",
        )


def add_execution_args(
    parser: argparse.ArgumentParser,
    multi_job: bool = False,
    n_jobs_default: int = 4
) -> None:
    """
    Add execution control arguments to an argument parser.

    Args:
        parser: ArgumentParser instance to add arguments to
        multi_job: If True, add --n-jobs; if False, add --no-wait
        n_jobs_default: Default number of parallel jobs (only used if multi_job=True)
    """
    if multi_job:
        parser.add_argument(
            "--n-jobs",
            type=int,
            default=n_jobs_default,
            help=f"Number of parallel jobs to run locally (default: {n_jobs_default}).",
        )
    else:
        parser.add_argument(
            "--no-wait",
            action="store_true",
            help="Don't wait for Slurm job to complete (submit and exit).",
        )




# ====================
# Post-processing Helpers
# ====================

def apply_post_processing(
    results: list[Any],
    callback: Callable[[list[Any]], Any] | None = None
) -> Any:
    """
    Apply optional post-processing callback to results.

    Args:
        results: List of results from parallel jobs
        callback: Optional function to apply to results

    Returns:
        Callback result if callback provided, otherwise None
    """
    if callback is not None:
        return callback(results)
    return None


# ====================
# Multi-Partition Support
# ====================

def add_partition_args(
    parser: argparse.ArgumentParser,
    enable_multi_partition: bool = True,
) -> None:
    """
    Add partition argument(s) to an argument parser.

    Supports both single partition (--partition) and multiple partitions
    (--partitions) for distributing jobs across multiple Slurm partitions.

    Args:
        parser: ArgumentParser instance to add arguments to
        enable_multi_partition: If True, adds --partitions argument for
            multi-partition support
    """
    parser.add_argument(
        "--partition",
        type=str,
        default="main",
        help="Slurm partition to use (default: main).",
    )

    if enable_multi_partition:
        parser.add_argument(
            "--partitions",
            type=str,
            nargs="+",
            default=None,
            help=(
                "Multiple Slurm partitions to distribute jobs across "
                "(e.g., --partitions milan genoa). Jobs will be round-robin "
                "distributed. If specified, overrides --partition."
            ),
        )


def resolve_partitions(
    partition: str,
    partitions: list[str] | None,
) -> list[str]:
    """
    Resolve partition list from single partition or multi-partition arguments.

    Args:
        partition: Single partition from --partition argument
        partitions: List of partitions from --partitions argument (or None)

    Returns:
        List of partitions to use for job distribution
    """
    return partitions if partitions else [partition]


class PartitionDistributor:
    """
    Distributes jobs across multiple Slurm partitions using round-robin.

    Tracks partition usage and provides partition selection for job submission.

    Example:
        >>> distributor = PartitionDistributor(["milan", "genoa"])
        >>> for i in range(5):
        ...     partition = distributor.get_next()
        ...     print(f"Job {i}: {partition}")
        Job 0: milan
        Job 1: genoa
        Job 2: milan
        Job 3: genoa
        Job 4: milan
        >>> distributor.get_summary()
        {'milan': 3, 'genoa': 2}
    """

    def __init__(self, partitions: list[str]):
        """
        Initialize the partition distributor.

        Args:
            partitions: List of partition names to distribute across
        """
        self.partitions = partitions
        self.counts = {p: 0 for p in partitions}
        self.index = 0

    def get_next(self) -> str:
        """
        Get the next partition in round-robin order.

        Returns:
            Partition name
        """
        partition = self.partitions[self.index % len(self.partitions)]
        self.counts[partition] += 1
        self.index += 1
        return partition

    def get_summary(self) -> dict[str, int]:
        """
        Get summary of jobs distributed to each partition.

        Returns:
            Dictionary mapping partition name to job count
        """
        return self.counts.copy()

    def __len__(self) -> int:
        """Return number of partitions."""
        return len(self.partitions)
