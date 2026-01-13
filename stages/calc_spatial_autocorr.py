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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from simple_slurm import Slurm

from src.conf.conf import get_config

# Setup environment and path
from src.pipeline.entrypoint_utils import (
    add_common_args,
    add_resource_args,
    build_base_command,
    determine_execution_mode,
    setup_environment,
    setup_log_directory,
    wait_for_job_completion,
)

project_root = setup_environment()


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate spatial autocorrelation locally or on Slurm."
    )
    add_common_args(parser)
    add_resource_args(
        parser,
        time_default="06:00:00",
        cpus_default=8,
        mem_default="64GB",
        include_gpus=True,
        gpus_default="1",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help=(
            "Enable debug mode: process only 1 trait and enable verbose logging "
            "in the GPU calculation script."
        ),
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for Slurm job to complete (submit and exit).",
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

    # Set CONFIG_PATH BEFORE importing dataset_utils to avoid module-level load error
    if params_path is not None:
        os.environ["CONFIG_PATH"] = params_path

    # Import after setting CONFIG_PATH
    from src.utils.dataset_utils import get_y_fn

    # Determine execution mode
    use_local, mode = determine_execution_mode(args.local)
    ranges_fp = Path(
        get_config(params_path=params_path).spatial_autocorr.ranges_fp
    ).resolve()

    all_cols = (
        dd.read_parquet(get_y_fn(get_config(params_path=params_path)))
        .columns.difference(["x", "y", "source"])
        .to_list()
    )
    # Filter out reliability columns
    traits = [t for t in all_cols if not t.endswith("_reliability")]
    print(f"Found {len(traits)} traits to process: {', '.join(traits)}")

    # In debug mode, only process the first trait
    if args.debug:
        traits = traits[:1]
        print(f"DEBUG MODE: Limited to 1 trait: {traits[0]}")

    print(f"Execution mode: {mode}")

    if use_local:
        # Run locally
        run_local(
            params_path,
            args.debug,
            args.overwrite,
            traits,
            args.max_parallel,
            ranges_fp,
        )
    else:
        # Submit to Slurm
        run_slurm(
            params_path,
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



def filter_completed_traits(
    traits: list[str], temp_dir: Path, overwrite: bool
) -> tuple[list[str], list[str]]:
    """
    Filter out traits that already have results in the temp directory.

    Args:
        traits: List of all trait names to process
        temp_dir: Directory containing individual trait parquet files
        overwrite: If True, return all traits (don't skip any)

    Returns:
        Tuple of (traits_to_process, already_completed_traits)
    """
    if overwrite:
        return traits, []

    to_process = []
    already_done = []

    for trait in traits:
        trait_fp = temp_dir / f"spatial_autocorr_{trait}.parquet"
        if trait_fp.exists():
            already_done.append(trait)
        else:
            to_process.append(trait)

    if already_done:
        print(f"\nSkipping {len(already_done)} traits with existing results:")
        print(f"  {', '.join(already_done)}")
        print("  (use --overwrite to reprocess all traits)")

    return to_process, already_done

def run_single_trait(
    trait: str,
    params_path: str | None,
    debug: bool,
    output_dir: Path,
) -> tuple[str, bool]:
    """
    Run spatial autocorrelation calculation for a single trait.

    Args:
        trait: Trait name
        params_path: Path to parameters file
        debug: Enable debug mode
        output_dir: Output directory for results

    Returns:
        Tuple of (trait_name, success)
    """
    extra_args: dict[str, str | None] = {
        "--trait": trait,
        "--output-dir": str(output_dir),
    }
    if debug:
        extra_args["--debug"] = None

    cmd = build_base_command(
        "src.features.calc_spatial_autocorr_gpu",
        params_path=params_path,
        overwrite=True,  # Always overwrite when processing individual traits
        extra_args=extra_args,
    )

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
            stderr_lines = result.stderr.strip().split("\n")
            print(f"    Last {min(20, len(stderr_lines))} lines of stderr:")
            for line in stderr_lines[-20:]:
                print(f"      {line}")
        if result.stdout:
            # Also check stdout for error messages
            stdout_lines = result.stdout.strip().split("\n")
            if stdout_lines:
                print(f"    Last {min(5, len(stdout_lines))} lines of stdout:")
                for line in stdout_lines[-5:]:
                    print(f"      {line}")
        return (trait, False)


def run_local(
    params_path: str | None,
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

    # Filter out traits that already have results
    traits_to_process, already_done = filter_completed_traits(
        traits, temp_dir, overwrite
    )

    if not traits_to_process:
        print("\nAll traits already processed!")
        # Combine existing results
        combine_results(temp_dir, ranges_fp, cfg, already_done)
        print("\n✓ Spatial autocorrelation calculation completed successfully!")
        return

    # Process traits in parallel
    print(f"\nProcessing {len(traits_to_process)} traits with up to {max_parallel} workers...")
    successful = []
    failed = []

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all trait jobs
        futures = {
            executor.submit(
                run_single_trait, trait, params_path, debug, temp_dir
            ): trait
            for trait in traits_to_process
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
    print(f"Completed: {len(successful)}/{len(traits_to_process)} traits")
    if failed:
        print(f"Failed: {len(failed)} traits: {', '.join(failed)}")
        print("✗ Some traits failed to process")
        sys.exit(1)

    # Combine results (include previously completed traits)
    all_successful = already_done + successful
    combine_results(temp_dir, ranges_fp, cfg, all_successful)

    print("\n✓ Spatial autocorrelation calculation completed successfully!")


def run_slurm(
    params_path: str | None,
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
    print(f"\nSubmitting Slurm jobs for {len(traits)} traits...")

    # Get configuration
    cfg = get_config(params_path=params_path)

    # Check if output already exists
    if ranges_fp.exists() and not overwrite:
        print(f"Output file already exists: {ranges_fp}")
        print("Use --overwrite flag to overwrite existing files.")
        sys.exit(0)

    # Create log directory in product-specific location
    log_dir = setup_log_directory("calc_spatial_autocorr")

    # Create temporary directory for individual trait results
    temp_dir = ranges_fp.parent / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Filter out traits that already have results
    traits_to_process, already_done = filter_completed_traits(
        traits, temp_dir, overwrite
    )

    if not traits_to_process:
        print("\nAll traits already processed!")
        # Combine existing results
        combine_results(temp_dir, ranges_fp, cfg, already_done)
        print("\n✓ Spatial autocorrelation calculation completed successfully!")
        return

    print(f"\nSubmitting {len(traits_to_process)} Slurm jobs...")

    # Submit a job for each trait
    job_ids = []
    trait_to_job = {}

    for trait in traits_to_process:
        # Construct command for this trait
        extra_args: dict[str, str | None] = {
            "--trait": trait,
            "--output-dir": str(temp_dir),
        }
        if debug:
            extra_args["--debug"] = None

        cmd_parts = build_base_command(
            "src.features.calc_spatial_autocorr_gpu",
            params_path=params_path,
            overwrite=True,
            extra_args=extra_args,
        )
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
            success = wait_for_job_completion(job_id)

            if success:
                successful.append(trait)
            else:
                failed.append(trait)
                print(f"✗ Job {job_id} ({trait}) failed. Check logs:")
                print(f"  {log_dir.absolute()}/{job_id}_{trait}.err")

        # Report results
        print(f"\n{'=' * 60}")
        print(f"Completed: {len(successful)}/{len(traits_to_process)} traits")
        if failed:
            print(f"Failed: {len(failed)} traits: {', '.join(failed)}")
            print("✗ Some jobs failed")
            sys.exit(1)

        # Combine results (include previously completed traits)
        all_successful = already_done + successful
        combine_results(temp_dir, ranges_fp, cfg, all_successful)

        print("\n✓ Spatial autocorrelation calculation completed successfully!")
    else:
        print("\nJobs submitted. Not waiting for completion (--no-wait flag set).")
        print("To combine results later, run this script again with --wait flag.")


if __name__ == "__main__":
    main()
