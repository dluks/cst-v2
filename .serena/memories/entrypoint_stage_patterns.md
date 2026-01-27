# Entrypoint Stage Patterns

Standard patterns for `stages/*.py` entrypoint scripts.

## Import Structure

```python
#!/usr/bin/env python
"""Module docstring."""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from simple_slurm import Slurm

# Setup environment FIRST
from src.pipeline.entrypoint_utils import (
    PartitionDistributor,
    add_common_args,
    add_execution_args,
    add_partition_args,
    add_retry_args,
    build_base_command,
    determine_execution_mode,
    get_existing_job_names,
    resolve_partitions,
    setup_environment,
    setup_log_directory,
    wait_for_job_completion,
)

project_root = setup_environment()

# Import config AFTER setup_environment
from src.conf.conf import get_config  # noqa: E402
```

## CLI Argument Helpers

| Helper | Purpose |
|--------|---------|
| `add_common_args(parser, include_partition=False)` | `--params`, `--overwrite`, `--local` |
| `add_execution_args(parser, multi_job=True)` | `--n-jobs` or `--no-wait` |
| `add_partition_args(parser, enable_multi_partition=True)` | `--partition`/`--partitions` |
| `add_retry_args(parser)` | `--max-retries` for node failures |

## build_base_command

Takes **module path** (dots), NOT script path (slashes):
```python
# CORRECT
build_base_command("src.features.build_y_trait", params_path=params_path, ...)

# WRONG
build_base_command("src/features/build_y_trait.py", ...)
```

## Partition Distribution

```python
distributor = PartitionDistributor(partitions)
partition = distributor.get_next()  # NOT next_partition()
```

## Check Existing Jobs Before Submission

```python
existing_jobs = get_existing_job_names()
if job_name in existing_jobs:
    existing_job_id, existing_state = existing_jobs[job_name]
    # Skip submission
```

## Log Directory

```python
log_dir = setup_log_directory("stage_name")  # Creates logs/{stage_name}/
# Or with product code:
log_dir = Path("logs/stage_name") / cfg.product_code
log_dir.mkdir(parents=True, exist_ok=True)
```

Log file pattern: `%j_{task_name}.log` and `%j_{task_name}.err`

## Job Dependencies

```python
dependency_str = "afterok:" + ":".join(str(jid) for jid in dep_job_ids)
slurm = Slurm(..., dependency=dependency_str)
```

## wait_for_job_completion

Takes single job ID, not a list:
```python
success = wait_for_job_completion(job_id, "task_name")
```

## Slurm Job Submission Pattern

```python
slurm = Slurm(
    job_name=job_name,
    output=str(log_dir / f"%j_{task_name}.log"),
    error=str(log_dir / f"%j_{task_name}.err"),
    time=time_limit,
    cpus_per_task=cpus,
    mem=mem,
    partition=partition,
    gres=f"gpu:{n_gpus}" if n_gpus else None,
    dependency=dependency_str,  # Optional
)
job_id = slurm.sbatch(command)
time.sleep(0.5)  # Avoid overwhelming scheduler
```
