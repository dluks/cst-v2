#!/usr/bin/env python
"""
Entry point script to generate Slurm jobs for harmonizing EO data files.

This script reads EO datasets from the configuration file and submits a separate
Slurm job for each file to process it in parallel, or runs them locally in parallel
if Slurm is unavailable. After all files are processed, it runs post-processing
steps (MODIS NDVI calculation and WorldClim pruning) if needed.

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
    PartitionDistributor,
    add_common_args,
    add_execution_args,
    add_partition_args,
    add_resource_args,
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
        description=(
            "Submit Slurm jobs to harmonize EO data files for all configured datasets."
        )
    )
    add_common_args(parser, include_partition=False)
    add_execution_args(parser, multi_job=True, n_jobs_default=4)
    add_partition_args(parser, enable_multi_partition=True)
    add_resource_args(
        parser,
        time_default="00:30:00",
        cpus_default=8,
        mem_default="8GB",
        include_gpus=False,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: process only one file and skip post-processing jobs.",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to submit Slurm jobs or run locally for each file."""
    args = cli()
    params_path = Path(args.params).resolve()
    cfg = get_config(params_path=params_path)

    # Collect all files from datasets
    print("Collecting files from datasets...")
    files_to_process = []
    for dataset, path in cfg.datasets.items():
        dataset_files = list(Path(project_root, path).glob("*.tif"))
        files_to_process.extend([(dataset, f) for f in dataset_files])
        print(f"  {dataset}: {len(dataset_files)} files")

    print(
        f"Found {len(files_to_process)} files to process "
        f"across {len(cfg.datasets)} datasets"
    )

    if not files_to_process:
        print("No files to process. Exiting.")
        return

    # Filter out files that already exist (unless overwrite is True)
    if not args.overwrite:
        out_dir = Path(cfg.interim.out_dir).resolve()
        files_to_skip = []
        files_filtered = []

        for dataset, file_path in files_to_process:
            out_path = out_dir / dataset / file_path.with_suffix(".tif").name
            if out_path.exists():
                files_to_skip.append((dataset, file_path))
            else:
                files_filtered.append((dataset, file_path))

        if files_to_skip:
            print(
                f"\nSkipping {len(files_to_skip)} files that already exist "
                f"(use --overwrite to reprocess)"
            )

        files_to_process = files_filtered

        if not files_to_process:
            print("All files already processed. Nothing to do.")
            return

        print(f"Processing {len(files_to_process)} new/updated files")

    # Handle debug mode
    if args.debug:
        print("\n⚠️  DEBUG MODE: Processing only 1 file, skipping post-processing")
        files_to_process = [files_to_process[0]]
        print(f"Debug file: {files_to_process[0][0]}/{files_to_process[0][1].name}")

    # Determine execution mode
    use_local, mode = determine_execution_mode(args.local)
    print(f"Execution mode: {mode}")

    if use_local:
        # Run locally in parallel
        run_local(
            files_to_process,
            str(params_path),
            args.overwrite,
            args.n_jobs,
            cfg,
            debug=args.debug,
        )
    else:
        # Determine partitions to use
        partitions = resolve_partitions(args.partition, args.partitions)
        if len(partitions) > 1:
            print(
                f"Distributing jobs across {len(partitions)} partitions: "
                f"{', '.join(partitions)}"
            )
        else:
            print(f"Using partition: {partitions[0]}")

        # Submit to Slurm
        log_dir = setup_log_directory("harmonize_eo_data")
        print(f"Logs will be written to: {log_dir.absolute()}")
        run_slurm(
            files_to_process,
            str(params_path),
            args.overwrite,
            partitions,
            log_dir,
            args.time,
            args.cpus,
            args.mem,
            cfg,
            debug=args.debug,
        )


def run_file_job(
    dataset: str, file_path: Path, params_path: str | None, overwrite: bool
) -> tuple[str, int]:
    """Run a single file job locally."""
    cmd = build_base_command(
        "src.data.harmonize_eo_file",
        params_path=params_path,
        overwrite=overwrite,
        extra_args={"--file": str(file_path), "--dataset": dataset},
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    return f"{dataset}/{file_path.name}", result.returncode


def run_postprocessing_job(
    module: str, params_path: str | None, overwrite: bool, name: str
) -> tuple[str, int]:
    """Run a post-processing job locally."""
    cmd = build_base_command(
        module,
        params_path=params_path,
        overwrite=overwrite,
    )

    print(f"Running post-processing: {name}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return name, result.returncode


def run_local(
    files: list[tuple[str, Path]],
    params_path: str | None,
    overwrite: bool,
    n_jobs: int,
    cfg,
    debug: bool = False,
) -> None:
    """Run file jobs locally in parallel."""
    print(f"\nProcessing {len(files)} files locally with {n_jobs} parallel jobs")

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all file jobs
        futures = {
            executor.submit(run_file_job, dataset, file_path, params_path, overwrite): (
                dataset,
                file_path,
            )
            for dataset, file_path in files
        }

        # Track results
        completed = 0
        failed = []

        # Process results as they complete
        for future in as_completed(futures):
            dataset, file_path = futures[future]
            try:
                file_name, returncode = future.result()
                completed += 1
                if returncode == 0:
                    print(f"✓ [{completed}/{len(files)}] Completed: {file_name}")
                else:
                    print(
                        f"✗ [{completed}/{len(files)}] Failed: {file_name} "
                        f"(exit code: {returncode})"
                    )
                    failed.append(file_name)
            except Exception as e:
                completed += 1
                file_name = f"{dataset}/{file_path.name}"
                print(f"✗ [{completed}/{len(files)}] Error in {file_name}: {e}")
                failed.append(file_name)

    # Report file processing results
    print(f"\n{'=' * 60}")
    print(f"File processing completed: {len(files) - len(failed)}/{len(files)}")

    if failed:
        print(f"Failed files ({len(failed)}):")
        for f in failed[:10]:  # Show first 10
            print(f"  {f}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
        sys.exit(1)
    else:
        print("All files processed successfully!")

    # Run post-processing jobs
    if debug:
        print(f"\n{'=' * 60}")
        print("⊘ Skipping post-processing in debug mode")
        print("All file processing completed successfully!")
        return

    print(f"\n{'=' * 60}")
    print("Running post-processing steps...")

    post_jobs = []
    if "modis" in cfg.datasets:
        post_jobs.append(("src.data.calculate_modis_ndvi", "MODIS NDVI"))
    if "worldclim" in cfg.datasets:
        post_jobs.append(("src.data.prune_worldclim_vars", "WorldClim pruning"))

    if not post_jobs:
        print("No post-processing steps needed.")
        return

    post_failed = []
    for module, name in post_jobs:
        _, returncode = run_postprocessing_job(module, params_path, overwrite, name)
        if returncode != 0:
            print(f"✗ Post-processing failed: {name}")
            post_failed.append(name)
        else:
            print(f"✓ Post-processing completed: {name}")

    # Final summary
    print(f"\n{'=' * 60}")
    if post_failed:
        print(f"Post-processing failed for: {', '.join(post_failed)}")
        sys.exit(1)
    else:
        print("All processing completed successfully!")


def run_slurm(
    files: list[tuple[str, Path]],
    params_path: str | None,
    overwrite: bool,
    partitions: list[str],
    log_dir: Path,
    time_limit: str,
    cpus: int,
    mem: str,
    cfg,
    debug: bool = False,
) -> None:
    """Submit file jobs to Slurm and wait for completion."""
    # Create partition distributor for round-robin distribution
    distributor = PartitionDistributor(partitions)

    job_ids = []
    file_to_job = {}

    print(f"\nSubmitting {len(files)} file processing jobs...")

    for dataset, file_path in files:
        # Construct the command
        cmd_parts = build_base_command(
            "src.data.harmonize_eo_file",
            params_path=params_path,
            overwrite=overwrite,
            extra_args={"--file": str(file_path), "--dataset": dataset},
        )
        command = " ".join(cmd_parts)

        # Get partition using round-robin distribution
        partition = distributor.get_next()

        # Create safe filename for log (remove special characters)
        safe_filename = file_path.stem.replace(".", "_")

        # Create Slurm job configuration
        slurm = Slurm(
            job_name=f"eo_{dataset}",
            output=str(log_dir / f"%j_{dataset}_{safe_filename}.log"),
            error=str(log_dir / f"%j_{dataset}_{safe_filename}.err"),
            time=time_limit,
            cpus_per_task=cpus,
            mem=mem,
            partition=partition,
        )

        # Submit the job
        job_id = slurm.sbatch(command)
        job_ids.append(job_id)
        file_to_job[job_id] = f"{dataset}/{file_path.name}"

    print(f"Submitted {len(job_ids)} jobs successfully")

    # Show distribution summary if using multiple partitions
    if len(distributor) > 1:
        summary = distributor.get_summary()
        print("Job distribution across partitions:")
        for partition, count in summary.items():
            print(f"  {partition}: {count} jobs")

    # Wait for all file jobs to complete
    print(f"\nWaiting for {len(job_ids)} file jobs to complete...")
    successful = []
    failed = []

    for i, job_id in enumerate(job_ids, 1):
        file_name = file_to_job[job_id]
        success = wait_for_job_completion(job_id, poll_interval=5)

        if success:
            successful.append(file_name)
            print(f"✓ [{i}/{len(job_ids)}] Job {job_id} ({file_name}) completed")
        else:
            failed.append(file_name)
            dataset, fname = file_name.split("/", 1)
            safe_fname = Path(fname).stem.replace(".", "_")
            print(
                f"✗ [{i}/{len(job_ids)}] Job {job_id} ({file_name}) failed. Check logs:"
            )
            print(f"  {log_dir.absolute()}/{job_id}_{dataset}_{safe_fname}.err")

    # Report file processing results
    print(f"\n{'=' * 60}")
    print(f"File processing completed: {len(successful)}/{len(files)}")

    if failed:
        print(f"Failed files ({len(failed)}):")
        for f in failed[:10]:  # Show first 10
            print(f"  {f}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
        sys.exit(1)
    else:
        print("All files processed successfully!")

    # Run post-processing jobs
    if debug:
        print(f"\n{'=' * 60}")
        print("⊘ Skipping post-processing in debug mode")
        print("All file processing completed successfully!")
        return

    print(f"\n{'=' * 60}")
    print("Running post-processing steps...")

    post_jobs = []
    if "modis" in cfg.datasets:
        post_jobs.append(("src.data.calculate_modis_ndvi", "modis_ndvi", "MODIS NDVI"))
    if "worldclim" in cfg.datasets:
        post_jobs.append(
            ("src.data.prune_worldclim_vars", "worldclim_prune", "WorldClim pruning")
        )

    if not post_jobs:
        print("No post-processing steps needed.")
        return

    post_failed = []
    for module, job_suffix, name in post_jobs:
        print(f"\nSubmitting post-processing job: {name}")

        cmd_parts = build_base_command(
            module,
            params_path=params_path,
            overwrite=overwrite,
        )
        command = " ".join(cmd_parts)

        # Use first partition for post-processing
        partition = partitions[0]

        # Set resources based on job type
        # MODIS NDVI processes 12 months in parallel, needs more resources
        if job_suffix == "modis_ndvi":
            job_cpus = 12
            job_mem = "96GB"
            job_time = "00:30:00"
        else:
            job_cpus = 2
            job_mem = "16GB"
            job_time = "00:15:00"

        slurm = Slurm(
            job_name=f"eo_{job_suffix}",
            output=str(log_dir / f"%j_{job_suffix}.log"),
            error=str(log_dir / f"%j_{job_suffix}.err"),
            time=job_time,
            cpus_per_task=job_cpus,
            mem=job_mem,
            partition=partition,
        )

        job_id = slurm.sbatch(command)
        print(f"Submitted post-processing job {job_id} for {name}")

        # Wait for this post-processing job to complete
        success = wait_for_job_completion(job_id, poll_interval=5)

        if success:
            print(f"✓ Post-processing completed: {name}")
        else:
            print(f"✗ Post-processing failed: {name}. Check logs:")
            print(f"  {log_dir.absolute()}/{job_id}_{job_suffix}.err")
            post_failed.append(name)

    # Final summary
    print(f"\n{'=' * 60}")
    if post_failed:
        print(f"Post-processing failed for: {', '.join(post_failed)}")
        sys.exit(1)
    else:
        print("All processing completed successfully!")


if __name__ == "__main__":
    main()
