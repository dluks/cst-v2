# Inference Pipeline Refactoring Plan

## Overview

Consolidate 4 separate DVC stages (aoa, cov, predict, build_final_product) into a single unified "inference" stage with per-trait parallel processing, supporting both local and Slurm execution with task-specific resource requirements.

---

## Phase 1: Move Existing Code to Deprecated Folders

### 1.1 Deprecate old modules

- Move `src/models/predict_traits.py` → `src/models/deprecated/predict_traits.py`
- Move `src/analysis/aoa.py` → `src/analysis/deprecated/aoa.py`
- Move `src/data/build_final_product.py` → `src/data/deprecated/build_final_product.py`
- Add deprecation notices to each file

---

## Phase 2: Create New Per-Trait Modules

### 2.1 Create `src/models/predict_single_trait.py`

**Purpose:** Predict trait values for a single trait

**Function signature:**
```python
def predict_trait(
    trait: str,
    trait_set: str,
    cfg: ConfigBox,
    output_dir: Path,
    calculate_cov: bool = False,
    overwrite: bool = False
) -> Path | tuple[Path, Path]
```

**Responsibilities:**
- Loads trained model for the trait
- Generates prediction raster
- Optionally calculates CoV if `calculate_cov=True`
- Returns path(s) to output raster(s)
- Handles both prediction and CoV in same module (they share logic)

### 2.2 Create `src/analysis/aoa_single_trait.py`

**Purpose:** Calculate Area of Applicability for a single trait

**Function signature:**
```python
def calculate_aoa_trait(
    trait: str,
    trait_set: str,
    cfg: ConfigBox,
    output_dir: Path,
    overwrite: bool = False
) -> Path
```

**Responsibilities:**
- Calculates Area of Applicability for single trait
- Returns path to AoA raster
- Refactored from existing `aoa.py` but processes only one trait

### 2.3 Create `src/data/build_final_product_single_trait.py`

**Purpose:** Combine prediction, CoV, and AoA into final multi-band raster

**Function signature:**
```python
def build_final_product_trait(
    trait: str,
    trait_set: str,
    cfg: ConfigBox,
    predict_path: Path,
    cov_path: Path | None,
    aoa_path: Path | None,
    output_dir: Path,
    overwrite: bool = False
) -> Path
```

**Responsibilities:**
- Combines prediction, CoV, and AoA into single multi-band raster
- Handles partial updates: if final raster exists, can add/update individual bands
- If a band is missing (e.g., AoA), only that band needs to be recalculated
- Adds metadata and creates Cloud-Optimized GeoTIFF (COG)
- Returns path to final product

---

## Phase 3: Create Unified Entrypoint Module

### 3.1 Create `stages/inference.py`

#### Main Architecture

```python
def main():
    args = cli()
    cfg = get_config(params_path=args.params)

    # Get trait list (from --traits or cfg.traits.names)
    traits = args.traits if args.traits else cfg.traits.names
    trait_set = args.trait_set  # e.g., "splot_gbif"

    # Determine execution mode
    use_local = determine_execution_mode(args.local)

    if use_local:
        run_local(traits, trait_set, args, cfg)
    else:
        run_slurm(traits, trait_set, args, cfg)
```

#### Command-line Arguments

```python
def cli():
    parser = argparse.ArgumentParser()

    # General args
    add_common_args(parser)  # --params, --local, --overwrite
    parser.add_argument("--traits", nargs="+", help="Specific traits to process")
    parser.add_argument("--trait-set", default="splot_gbif")

    # Task selection flags
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--cov-only", action="store_true")
    parser.add_argument("--aoa-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    # If none specified, run all tasks

    # Predict resources
    parser.add_argument("--predict-partition", default="genoa")
    parser.add_argument("--predict-cpus", type=int, default=48)
    parser.add_argument("--predict-mem", default="64GB")
    parser.add_argument("--predict-time", default="02:00:00")
    parser.add_argument("--predict-gpus", default="0")

    # CoV resources (often same as predict)
    parser.add_argument("--cov-partition", default="genoa")
    parser.add_argument("--cov-cpus", type=int, default=48)
    parser.add_argument("--cov-mem", default="64GB")
    parser.add_argument("--cov-time", default="02:00:00")
    parser.add_argument("--cov-gpus", default="0")

    # AoA resources
    parser.add_argument("--aoa-partition", default="l40s")
    parser.add_argument("--aoa-cpus", type=int, default=8)
    parser.add_argument("--aoa-mem", default="128GB")
    parser.add_argument("--aoa-time", default="04:00:00")
    parser.add_argument("--aoa-gpus", default="2")

    # Build final product resources
    parser.add_argument("--build-partition", default="genoa")
    parser.add_argument("--build-cpus", type=int, default=4)
    parser.add_argument("--build-mem", default="16GB")
    parser.add_argument("--build-time", default="00:30:00")

    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=4)

    return parser.parse_args()
```

### 3.2 Local Execution Logic

```python
def run_local(traits, trait_set, args, cfg):
    """Process traits locally with parallel execution."""

    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        # Phase 1: Predict, CoV, AoA (parallel)
        futures = {}

        for trait in traits:
            if should_run_task("predict", args):
                future = executor.submit(run_predict_task, trait, trait_set, args, cfg)
                futures[future] = (trait, "predict")

            if should_run_task("cov", args):
                future = executor.submit(run_cov_task, trait, trait_set, args, cfg)
                futures[future] = (trait, "cov")

            if should_run_task("aoa", args):
                future = executor.submit(run_aoa_task, trait, trait_set, args, cfg)
                futures[future] = (trait, "aoa")

        # Wait for Phase 1 to complete
        phase1_results = {}
        for future in as_completed(futures):
            trait, task = futures[future]
            success, output_path = future.result()
            phase1_results[(trait, task)] = (success, output_path)

        # Phase 2: Build final products (depends on Phase 1)
        if should_run_task("build", args):
            build_futures = {}
            for trait in traits:
                # Check if all required inputs are available
                predict_success = phase1_results.get((trait, "predict"), (True, None))[0]
                aoa_success = phase1_results.get((trait, "aoa"), (True, None))[0]
                cov_success = phase1_results.get((trait, "cov"), (True, None))[0]

                if predict_success:  # At minimum need prediction
                    future = executor.submit(run_build_task, trait, trait_set, args, cfg)
                    build_futures[future] = trait

            # Wait for Phase 2
            for future in as_completed(build_futures):
                trait = build_futures[future]
                success, output_path = future.result()

    # Report results
    report_results(phase1_results, build_results)
```

### 3.3 Slurm Execution Logic

```python
def run_slurm(traits, trait_set, args, cfg):
    """Submit Slurm jobs with dependency tracking."""

    log_dir = setup_log_directory("inference")

    # Phase 1: Submit predict, cov, aoa jobs (parallel per trait)
    job_tracker = {}  # {trait: {task: job_id}}

    for trait in traits:
        job_tracker[trait] = {}

        # Submit predict job
        if should_run_task("predict", args):
            predict_job_id = submit_task_job(
                task="predict",
                trait=trait,
                trait_set=trait_set,
                args=args,
                cfg=cfg,
                log_dir=log_dir,
                dependencies=None,
            )
            job_tracker[trait]["predict"] = predict_job_id

        # Submit CoV job
        if should_run_task("cov", args):
            cov_job_id = submit_task_job(
                task="cov",
                trait=trait,
                trait_set=trait_set,
                args=args,
                cfg=cfg,
                log_dir=log_dir,
                dependencies=None,
            )
            job_tracker[trait]["cov"] = cov_job_id

        # Submit AoA job
        if should_run_task("aoa", args):
            aoa_job_id = submit_task_job(
                task="aoa",
                trait=trait,
                trait_set=trait_set,
                args=args,
                cfg=cfg,
                log_dir=log_dir,
                dependencies=None,
            )
            job_tracker[trait]["aoa"] = aoa_job_id

    # Phase 2: Submit build jobs (depend on Phase 1 for each trait)
    if should_run_task("build", args):
        for trait in traits:
            # Collect dependency job IDs for this trait
            dep_jobs = [
                job_id for task, job_id in job_tracker[trait].items()
            ]

            # Build --dependency string: "afterok:job1:job2:job3"
            dependency_str = "afterok:" + ":".join(map(str, dep_jobs))

            build_job_id = submit_task_job(
                task="build",
                trait=trait,
                trait_set=trait_set,
                args=args,
                cfg=cfg,
                log_dir=log_dir,
                dependencies=dependency_str,
            )
            job_tracker[trait]["build"] = build_job_id

    # Wait for completion if requested
    if not args.no_wait:
        wait_for_all_jobs(job_tracker)
        report_results(job_tracker, log_dir)
```

### 3.4 Task Submission Helper

```python
def submit_task_job(task, trait, trait_set, args, cfg, log_dir, dependencies):
    """Submit a single task job to Slurm."""

    # Map task to module
    task_modules = {
        "predict": "src.models.predict_single_trait",
        "cov": "src.models.predict_single_trait",  # Same module, different flag
        "aoa": "src.analysis.aoa_single_trait",
        "build": "src.data.build_final_product_single_trait",
    }

    # Get resource requirements for this task
    resources = get_task_resources(task, args)

    # Build command
    extra_args = {
        "--trait": trait,
        "--trait-set": trait_set,
    }

    if task == "cov":
        extra_args["--cov"] = None  # Flag for CoV calculation

    if args.overwrite:
        extra_args["--overwrite"] = None

    cmd_parts = build_base_command(
        task_modules[task],
        params_path=args.params,
        overwrite=args.overwrite,
        extra_args=extra_args,
    )

    # Create Slurm job
    slurm = Slurm(
        job_name=f"inference_{task}_{trait}",
        output=str(log_dir / f"%j_{task}_{trait}.log"),
        error=str(log_dir / f"%j_{task}_{trait}.err"),
        partition=resources["partition"],
        time=resources["time"],
        cpus_per_task=resources["cpus"],
        mem=resources["mem"],
        gres=f"gpu:{resources['gpus']}" if resources.get("gpus", "0") != "0" else None,
        dependency=dependencies,  # None or "afterok:123:456"
    )

    job_id = slurm.sbatch(" ".join(cmd_parts))
    print(f"Submitted {task} job {job_id} for trait {trait}")

    return job_id
```

### 3.5 Smart Task Detection

```python
def check_missing_bands(trait, trait_set, cfg):
    """Check which bands are missing from final product."""
    final_product_path = get_final_product_path(trait, trait_set, cfg)

    if not final_product_path.exists():
        return {"predict", "cov", "aoa"}  # All tasks needed

    missing_bands = set()

    with rasterio.open(final_product_path) as ds:
        band_count = ds.count
        band_descriptions = [ds.get_band_description(i) for i in range(1, band_count + 1)]

        # Check for each expected band
        if not any("predict" in desc.lower() or trait in desc for desc in band_descriptions):
            missing_bands.add("predict")

        if not any("coefficient of variation" in desc.lower() for desc in band_descriptions):
            missing_bands.add("cov")

        if not any("area of applicability" in desc.lower() for desc in band_descriptions):
            missing_bands.add("aoa")

    return missing_bands

def should_run_task(task, args):
    """Determine if a task should be run based on CLI flags."""

    # If specific --xxx-only flag is set, only run that task
    if args.predict_only:
        return task == "predict"
    if args.cov_only:
        return task == "cov"
    if args.aoa_only:
        return task == "aoa"
    if args.build_only:
        return task == "build"

    # Otherwise, run all tasks
    return True
```

---

## Phase 4: Update Individual Task Modules

### 4.1 Update `src/models/predict_single_trait.py`

```python
def main(args, cfg):
    trait = args.trait
    trait_set = args.trait_set
    calculate_cov = args.cov
    overwrite = args.overwrite

    # Check if output already exists
    output_path = get_predict_output_path(trait, trait_set, cfg)
    if output_path.exists() and not overwrite:
        log.info(f"Prediction exists for {trait}, skipping...")
        return

    # Load model and predict
    # ... existing logic from predict_traits.py ...

    if calculate_cov:
        cov_output_path = get_cov_output_path(trait, trait_set, cfg)
        if cov_output_path.exists() and not overwrite:
            log.info(f"CoV exists for {trait}, skipping...")
        else:
            # Calculate CoV
            # ... existing CoV logic ...
```

### 4.2 Update `src/data/build_final_product_single_trait.py`

```python
def build_final_product_trait(trait, trait_set, cfg, overwrite=False):
    """Build or update final product for a single trait."""

    output_path = get_final_product_path(trait, trait_set, cfg)

    # Check what bands exist and what's missing
    existing_bands = {}
    if output_path.exists() and not overwrite:
        with rasterio.open(output_path) as ds:
            for i in range(1, ds.count + 1):
                desc = ds.get_band_description(i)
                existing_bands[i] = {
                    "description": desc,
                    "data": ds.read(i),
                }

    # Collect new/updated bands
    bands_to_write = {}

    # Band 1: Prediction
    predict_path = get_predict_output_path(trait, trait_set, cfg)
    if predict_path.exists():
        with rasterio.open(predict_path) as ds:
            bands_to_write[1] = {
                "data": ds.read(1),
                "description": f"{trait} prediction",
                "scale": ds.scales[0],
                "offset": ds.offsets[0],
            }
    elif 1 in existing_bands:
        bands_to_write[1] = existing_bands[1]

    # Band 2: CoV
    cov_path = get_cov_output_path(trait, trait_set, cfg)
    if cov_path.exists():
        # ... load CoV band ...
    elif 2 in existing_bands:
        bands_to_write[2] = existing_bands[2]

    # Band 3: AoA
    aoa_path = get_aoa_output_path(trait, trait_set, cfg)
    if aoa_path.exists():
        # ... load AoA band ...
    elif 3 in existing_bands:
        bands_to_write[3] = existing_bands[3]

    # Write final multi-band raster
    # ... existing COG creation logic ...
```

---

## Phase 5: Update DVC Configuration

### 5.1 Update `pipeline/products/<product>/dvc.yaml`

```yaml
inference:
  cmd: >-
    python ${project_root}/stages/inference.py
    --params params.yaml
    --trait-set splot_gbif
    --predict-partition genoa
    --predict-cpus 48
    --predict-mem 64GB
    --predict-time 02:00:00
    --cov-partition genoa
    --cov-cpus 48
    --cov-mem 64GB
    --cov-time 02:00:00
    --aoa-partition l40s
    --aoa-cpus 8
    --aoa-mem 128GB
    --aoa-time 04:00:00
    --aoa-gpus 2
    --build-partition genoa
    --build-cpus 4
    --build-mem 16GB
    --build-time 00:30:00
  deps:
    - ${project_root}/stages/inference.py
    - ${project_root}/src/models/predict_single_trait.py
    - ${project_root}/src/analysis/aoa_single_trait.py
    - ${project_root}/src/data/build_final_product_single_trait.py
    - ${project_root}/${models.dir_fp}
  params:
    - traits.names
    - train.trait_sets
  outs:
    - ${project_root}/${processed.dir}/${PFT}/${model_res}/maps:
        persist: true
```

### 5.2 Remove Old Stages

- Delete `aoa:` stage
- Delete `predict:` stage
- Delete `cov:` stage
- Delete `build_final_product:` stage

---

## Phase 6: Testing Strategy

### 6.1 Local Testing

```bash
# Test single trait, single task
python stages/inference.py --params params.yaml --traits X50 --predict-only --local

# Test single trait, all tasks
python stages/inference.py --params params.yaml --traits X50 --local

# Test multiple traits
python stages/inference.py --params params.yaml --traits X50 X55 --local --max-parallel 2
```

### 6.2 Slurm Testing

```bash
# Test with single trait
python stages/inference.py --params params.yaml --traits X50 --partition genoa

# Test with missing band scenario (--overwrite=false)
# 1. Run predict only
python stages/inference.py --params params.yaml --traits X50 --predict-only
# 2. Run AoA only (should add band to existing raster)
python stages/inference.py --params params.yaml --traits X50 --aoa-only

# Full pipeline
python stages/inference.py --params params.yaml
```

---

## Implementation Order

1. **Phase 1**: Move old code to deprecated (safe, reversible)
2. **Phase 2**: Create new per-trait modules (start with predict, then aoa, then build)
3. **Phase 3**: Create entrypoint with local execution first
4. **Phase 3**: Add Slurm execution with dependency tracking
5. **Phase 4**: Update task modules for partial updates
6. **Phase 5**: Update DVC config
7. **Phase 6**: Test incrementally

---

## Error Handling Strategy

### Per-trait Error Isolation

- If predict fails for trait X50, continue with other traits
- If AoA fails for trait X55, X55 final product will only have predict+CoV bands
- Comprehensive error reporting at end with failed traits and tasks
- Support re-running with `--overwrite=false` to only process failed/missing tasks

### Missing Dependencies

- If trained model not found for trait, skip that trait entirely
- If predict output missing when building final product, log error but continue

---

## Key Benefits

1. **Unified workflow**: Single command/stage for entire inference pipeline
2. **Parallel execution**: Traits processed in parallel, tasks within trait in parallel
3. **Resource flexibility**: Per-task resource requirements
4. **Robust**: Handles partial failures, supports incremental re-runs
5. **Smart**: Only recalculates missing bands with --overwrite=false
6. **Consistent**: Follows established patterns from calc_spatial_autocorr.py

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     stages/inference.py                         │
│                                                                 │
│  1. Parse CLI args (traits, resources, task flags)             │
│  2. Load config                                                 │
│  3. Determine execution mode (local/Slurm)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                   ┌─────────┴─────────┐
                   │                   │
            ┌──────▼──────┐    ┌──────▼──────┐
            │   Local     │    │   Slurm     │
            │  Execution  │    │  Execution  │
            └──────┬──────┘    └──────┬──────┘
                   │                   │
                   └─────────┬─────────┘
                             │
                  ┌──────────▼──────────┐
                  │  For each trait:    │
                  │                     │
                  │  Phase 1 (Parallel) │
                  │  ├─ Predict         │
                  │  ├─ CoV             │
                  │  └─ AoA             │
                  │                     │
                  │  Phase 2 (Depends)  │
                  │  └─ Build Final     │
                  └─────────────────────┘
```

---

## Task Dependencies

```
Trait X50:
  ├─ predict_X50 ─┐
  ├─ cov_X50 ─────┼─→ build_final_X50
  └─ aoa_X50 ─────┘

Trait X55:
  ├─ predict_X55 ─┐
  ├─ cov_X55 ─────┼─→ build_final_X55
  └─ aoa_X55 ─────┘

(All tasks can run in parallel across traits)
```
