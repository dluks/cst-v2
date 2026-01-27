# Inference Entrypoint Updates

This document outlines the patterns used in recent entrypoint scripts (`train_models.py`, `build_y.py`) and the changes needed to align `stages/inference.py` with these patterns.

---

## Current Entrypoint Patterns

### 1. Import Structure

Recent stages follow this import order:

```python
#!/usr/bin/env python
"""Module docstring."""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from simple_slurm import Slurm

# Setup environment and path FIRST
from src.pipeline.entrypoint_utils import (
    PartitionDistributor,
    add_common_args,
    add_execution_args,
    add_partition_args,
    add_retry_args,
    build_base_command,
    determine_execution_mode,
    get_existing_job_names,
    is_node_failure,
    resolve_partitions,
    setup_environment,
    setup_log_directory,
    wait_for_job_completion,
)

project_root = setup_environment()

# Import config AFTER setup_environment (needs PROJECT_ROOT set)
from src.conf.conf import get_config  # noqa: E402
```

**Key points:**
- `setup_environment()` is called before `get_config` import
- `get_config` is imported after `project_root` is set (with `# noqa: E402`)

### 2. CLI Argument Helpers

Use standardized helpers from `entrypoint_utils.py`:

| Helper | Purpose | Arguments |
|--------|---------|-----------|
| `add_common_args(parser, include_partition=False)` | Adds `--params`, `--overwrite`, `--local` | Set `include_partition=False` when using `add_partition_args` |
| `add_execution_args(parser, multi_job=True)` | Adds `--n-jobs` (multi_job=True) or `--no-wait` (multi_job=False) | `n_jobs_default` sets default parallel jobs |
| `add_partition_args(parser, enable_multi_partition=True)` | Adds `--partition` and/or `--partitions` | Enable multi for round-robin distribution |
| `add_retry_args(parser)` | Adds `--max-retries` | For automatic retry on node failures |

**Example CLI function:**

```python
def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="...")
    add_common_args(parser, include_partition=False)
    add_execution_args(parser, multi_job=True, n_jobs_default=4)
    add_partition_args(parser, enable_multi_partition=True)
    add_retry_args(parser)

    # Stage-specific arguments
    parser.add_argument("--time", type=str, default="02:00:00", ...)
    parser.add_argument("--cpus", type=int, default=48, ...)
    # ...

    return parser.parse_args()
```

### 3. Log Directory Setup

Use `setup_log_directory()` for consistent log organization:

```python
# Creates logs/{stage_name}/ directory
log_dir = setup_log_directory("inference")

# For product-specific logs:
log_dir = Path("logs/inference") / cfg.product_code
log_dir.mkdir(parents=True, exist_ok=True)
```

Log files follow pattern: `{job_id}_{task_name}.log` and `{job_id}_{task_name}.err`

### 4. build_base_command Usage

**CRITICAL:** `build_base_command` takes a **module path**, not a script path:

```python
# CORRECT - module path
cmd_parts = build_base_command(
    "src.models.predict_single_trait",  # Module path
    params_path=params_path,
    overwrite=overwrite,
    extra_args={"--trait": trait, "--trait-set": trait_set},
)
command = " ".join(cmd_parts)

# WRONG - script path
cmd_parts = build_base_command(
    "src/models/predict_single_trait.py",  # Script path - INCORRECT
    ...
)
```

### 5. Partition Distribution

Use `PartitionDistributor` with `get_next()` method:

```python
distributor = PartitionDistributor(partitions)

for task in tasks:
    partition = distributor.get_next()  # NOT next_partition()
    # Submit job to partition...
```

### 6. Check for Existing Jobs

Before submitting, check if jobs already exist in queue:

```python
print("\nChecking for existing jobs in queue...")
existing_jobs = get_existing_job_names()
if existing_jobs:
    print(f"Found {len(existing_jobs)} existing jobs in queue")

for task in tasks:
    job_name = f"inference_{task_type}_{trait}"

    # Skip if already in queue
    if job_name in existing_jobs:
        existing_job_id, existing_state = existing_jobs[job_name]
        print(f"  Skipping: job '{job_name}' already in queue (job {existing_job_id})")
        continue

    # Submit new job...
```

### 7. Job Dependencies

For dependent jobs, use `afterok` dependency string:

```python
# Phase 1 jobs (no dependencies)
phase1_job_ids = []
for task in phase1_tasks:
    job_id = slurm.sbatch(command)
    phase1_job_ids.append(job_id)

# Phase 2 jobs (depend on Phase 1)
dependency_str = "afterok:" + ":".join(str(jid) for jid in phase1_job_ids)

slurm = Slurm(
    ...,
    dependency=dependency_str,
)
job_id = slurm.sbatch(command)
```

### 8. Wait for Job Completion

`wait_for_job_completion` takes a **single job ID** (or with job name for logging):

```python
# Wait for single job
success = wait_for_job_completion(job_id, "task_name")

# For multiple jobs, wait for each
for job_id in job_ids:
    success = wait_for_job_completion(job_id, poll_interval=5)
```

---

## Issues in Current inference.py

### Issue 1: Import Structure
**Current:** Imports `get_config` at top level
**Fix:** Import after `setup_environment()`

### Issue 2: CLI Doesn't Use Helpers
**Current:** Manual argument definitions
**Fix:** Use `add_common_args`, `add_execution_args`, `add_partition_args`

### Issue 3: build_base_command with Script Path
**Current (line 512):**
```python
base_cmd = build_base_command(script_path, cmd_args)  # script_path = "src/models/predict_single_trait.py"
```
**Fix:**
```python
module_path = "src.models.predict_single_trait"  # Module path with dots
cmd_parts = build_base_command(
    module_path,
    params_path=params_path,
    overwrite=overwrite,
    extra_args={"--trait": trait, "--trait-set": trait_set},
)
```

### Issue 4: Incorrect Partition Distributor Method
**Current (line 471):**
```python
partition = distributor.next_partition()
```
**Fix:**
```python
partition = distributor.get_next()
```

### Issue 5: No Existing Job Check
**Current:** No check for existing jobs
**Fix:** Add `get_existing_job_names()` check before submission

### Issue 6: wait_for_job_completion with List
**Current (line 578):**
```python
wait_for_job_completion(all_job_ids)  # List of IDs
```
**Fix:** Wait for each job individually or wait for final dependent job

### Issue 7: Log Directory Structure
**Current:** Simple `logs/{job_name}_%j.out`
**Fix:** Use `setup_log_directory("inference")` and include product_code

---

## Proposed Changes

### 1. Update Imports

```python
from src.pipeline.entrypoint_utils import (
    PartitionDistributor,
    add_common_args,
    add_execution_args,
    add_partition_args,
    build_base_command,
    determine_execution_mode,
    get_existing_job_names,
    resolve_partitions,
    setup_environment,
    setup_log_directory,
    wait_for_job_completion,
)

project_root = setup_environment()

from src.conf.conf import get_config  # noqa: E402
```

### 2. Refactor CLI

```python
def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference tasks locally or on Slurm."
    )
    add_common_args(parser, include_partition=False)
    add_execution_args(parser, multi_job=True, n_jobs_default=4)
    add_partition_args(parser, enable_multi_partition=True)

    # Task-specific resource arguments
    # Predict resources
    parser.add_argument("--predict-time", type=str, default="02:00:00", ...)
    parser.add_argument("--predict-cpus", type=int, default=56, ...)
    parser.add_argument("--predict-mem", type=str, default="64GB", ...)

    # AoA resources (with GPU)
    parser.add_argument("--aoa-time", type=str, default="04:00:00", ...)
    parser.add_argument("--aoa-cpus", type=int, default=112, ...)
    parser.add_argument("--aoa-mem", type=str, default="128GB", ...)
    parser.add_argument("--aoa-gpus", type=str, default="1", ...)

    # Final product resources
    parser.add_argument("--final-time", type=str, default="01:00:00", ...)
    parser.add_argument("--final-cpus", type=int, default=28, ...)
    parser.add_argument("--final-mem", type=str, default="32GB", ...)

    # Task selection flags (keep existing)
    parser.add_argument("--predict-only", action="store_true", ...)
    # ... other --*-only and --skip-* flags ...

    # Other options
    parser.add_argument("--traits", type=str, nargs="+", ...)
    parser.add_argument("--trait-sets", type=str, nargs="+", ...)
    parser.add_argument("--dest", type=str, default="local", ...)

    return parser.parse_args()
```

### 3. Update run_slurm Function

```python
def run_slurm(
    params_path: str | None,
    overwrite: bool,
    partitions: list[str],
    tasks: list[dict],
    args: argparse.Namespace,
    cfg,
) -> None:
    """Submit all inference tasks to Slurm."""

    # Setup log directory
    log_dir = setup_log_directory("inference")
    product_log_dir = log_dir / cfg.product_code
    product_log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize partition distributor
    distributor = PartitionDistributor(partitions)

    # Check for existing jobs
    print("\nChecking for existing jobs in queue...")
    existing_jobs = get_existing_job_names()

    # Task type to module mapping
    TASK_MODULES = {
        "predict": "src.models.predict_single_trait",
        "cov": "src.models.predict_single_trait",
        "aoa": "src.analysis.aoa_single_trait",
        "final": "src.data.build_final_product_single_trait",
    }

    # Track jobs by trait/trait_set for dependencies
    job_tracker: dict[tuple[str, str], dict[str, int]] = {}

    for task in tasks:
        trait = task["trait"]
        trait_set = task["trait_set"]
        task_type = task["task_type"]

        # Build job name
        job_name = f"inf_{task_type[:4]}_{trait[:8]}_{cfg.product_code[:12]}"

        # Check if already in queue
        if job_name in existing_jobs:
            existing_job_id, existing_state = existing_jobs[job_name]
            print(f"  Skipping {task_type}/{trait}: already in queue ({existing_state})")
            # Track for dependencies
            key = (trait, trait_set)
            if key not in job_tracker:
                job_tracker[key] = {}
            job_tracker[key][task_type] = int(existing_job_id)
            continue

        # Build command
        extra_args = {"--trait": trait, "--trait-set": trait_set}
        if task_type == "cov":
            extra_args["--cov"] = None
        if task_type == "final":
            extra_args["--dest"] = args.dest

        cmd_parts = build_base_command(
            TASK_MODULES[task_type],
            params_path=params_path,
            overwrite=overwrite,
            extra_args=extra_args,
        )
        command = " ".join(cmd_parts)

        # Get resources for task type
        resources = get_task_resources(task_type, args)

        # Determine dependencies
        dependency = None
        if task_type == "final":
            key = (trait, trait_set)
            if key in job_tracker:
                dep_jobs = list(job_tracker[key].values())
                if dep_jobs:
                    dependency = "afterok:" + ":".join(str(j) for j in dep_jobs)

        # Get partition
        partition = distributor.get_next()

        # Create and submit Slurm job
        slurm = Slurm(
            job_name=job_name,
            output=str(product_log_dir / f"%j_{task_type}_{trait}.log"),
            error=str(product_log_dir / f"%j_{task_type}_{trait}.err"),
            partition=partition,
            time=resources["time"],
            cpus_per_task=resources["cpus"],
            mem=resources["mem"],
            gres=resources.get("gres"),
            dependency=dependency,
        )

        job_id = slurm.sbatch(command)

        # Track job
        key = (trait, trait_set)
        if key not in job_tracker:
            job_tracker[key] = {}
        job_tracker[key][task_type] = job_id

        dep_info = f" (depends on {len(dependency.split(':'))-1} jobs)" if dependency else ""
        print(f"  Submitted {task_type}/{trait}: job {job_id}{dep_info}")

        time.sleep(0.5)  # Avoid overwhelming scheduler

    # ... wait logic ...
```

### 4. Add Helper Function for Resources

```python
def get_task_resources(task_type: str, args: argparse.Namespace) -> dict:
    """Get resource requirements for a task type."""
    if task_type in ("predict", "cov"):
        return {
            "time": args.predict_time,
            "cpus": args.predict_cpus,
            "mem": args.predict_mem,
            "gres": None,
        }
    elif task_type == "aoa":
        return {
            "time": args.aoa_time,
            "cpus": args.aoa_cpus,
            "mem": args.aoa_mem,
            "gres": f"gpu:{args.aoa_gpus}" if args.aoa_gpus != "0" else None,
        }
    elif task_type == "final":
        return {
            "time": args.final_time,
            "cpus": args.final_cpus,
            "mem": args.final_mem,
            "gres": None,
        }
    else:
        raise ValueError(f"Unknown task type: {task_type}")
```

---

## Implementation Checklist

- [ ] Update import structure (setup_environment before get_config)
- [ ] Refactor CLI to use helper functions
- [ ] Fix `build_base_command` to use module paths
- [ ] Fix `PartitionDistributor.get_next()` method call
- [ ] Add existing job check with `get_existing_job_names()`
- [ ] Update log directory structure
- [ ] Fix `wait_for_job_completion` usage
- [ ] Add `get_task_resources` helper function
- [ ] Test local execution
- [ ] Test Slurm execution with dependencies

---

## Notes

- The underlying single-trait modules (`predict_single_trait.py`, `aoa_single_trait.py`, `build_final_product_single_trait.py`) need to exist and be callable as Python modules (`python -m src.models.predict_single_trait`)
- Ensure each module has proper CLI with `--trait`, `--trait-set`, `--params`, `--overwrite` arguments
- The `--cov` flag for predict_single_trait triggers CoV calculation in same module
