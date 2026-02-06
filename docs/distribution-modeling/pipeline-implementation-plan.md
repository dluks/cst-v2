# Histogram Modeling Pipeline Implementation Plan

## Overview

This document outlines where new pipelines/stages need to be created or modified to implement the histogram-based trait distribution modeling approach within the established DVC pipeline framework.

## Existing Infrastructure (Reusable)

The following existing pipelines and data can be used directly:

| Component | Location | Status |
|-----------|----------|--------|
| EO data (22km) | `pipeline/eo_data/modis_wc2_soil_canopy_vodca_alos_22km/` | ✅ Ready |
| TRY6 traits (power-transformed) | `pipeline/traits/try6_pow-xf/` | ✅ Ready |
| GBIF filtered occurrences | `data/interim/gbif/filtered_occurrences/try6/` | ✅ Ready |
| sPlot filtered surveys | `data/interim/splot/filtered_surveys/try6/` | ✅ Ready |
| Spatial fold assignment utilities | `src/utils/training_utils.py` | ✅ Reusable |
| Slurm/local execution framework | `src/pipeline/entrypoint_utils.py` | ✅ Reusable |

## New Pipeline Structure

```
pipeline/
├── histogram_models/                    # NEW: Histogram model pipelines
│   └── try6_hist_pow-xf_22km/          # First proof-of-concept
│       ├── params.yaml
│       ├── dvc.yaml
│       └── logs/
│
└── histogram_data/                      # NEW: Histogram target construction
    └── try6_hist_pow-xf_22km/
        ├── params.yaml
        ├── dvc.yaml
        └── logs/
```

## Implementation Phases

### Phase 1: Histogram Target Construction

**New Pipeline**: `pipeline/histogram_data/try6_hist_pow-xf_22km/`

**Purpose**: Construct probability histograms from GBIF observations and sPlot surveys.

**New Files Required**:

| File | Purpose |
|------|---------|
| `src/data/build_histogram_targets.py` | Core histogram construction logic |
| `stages/build_histogram_targets.py` | Slurm/local entrypoint |
| `pipeline/histogram_data/try6_hist_pow-xf_22km/params.yaml` | Configuration |
| `pipeline/histogram_data/try6_hist_pow-xf_22km/dvc.yaml` | DVC stage definition |

**Key Parameters** (`params.yaml`):
```yaml
version: "2"
project_root: "../../.."
product_dir: "pipeline/histogram_data/try6_hist_pow-xf_22km"
product_code: "try6_hist_pow-xf_22km"

# Spatial configuration (match eo_data resolution)
target_resolution: 22000
crs: "EPSG:6933"

# Histogram configuration
histogram:
  n_bins: 20
  label_smoothing_epsilon: 0.01

# GBIF filtering
gbif:
  min_observations: 30
  min_unique_species: 3
  min_bin_coverage: 0.25

# sPlot filtering
splot:
  min_total_abundance: 0.75
  min_unique_species: 3
  min_bin_coverage: 0.25

# Trait configuration
traits:
  interim_out: data/interim/traits/try6_pow-xf/try6_pow-xf.parquet
  transformer_dir: data/interim/traits/try6_pow-xf/transformers
  names: [X4, X6, X13, ...]  # All 31 traits

# Output
output:
  dir: data/interim/histogram_targets/try6_hist_pow-xf_22km
```

**DVC Stage** (`dvc.yaml`):
```yaml
params:
  - params.yaml

stages:
  build_gbif_histograms:
    cmd: >-
      python ${project_root}/stages/build_histogram_targets.py
      --params params.yaml
      --source gbif
      --partitions milan genoa
      --cpus 16
      --mem 64GB
      --time 02:00:00
    deps:
      - ${project_root}/src/data/build_histogram_targets.py
      - ${project_root}/data/interim/gbif/filtered_occurrences/try6/gbif_filtered.parquet
      - ${project_root}/data/interim/traits/try6_pow-xf/try6_pow-xf.parquet
    params:
      - histogram
      - gbif
      - traits.names
    outs:
      - ${project_root}/${output.dir}/gbif:
          persist: true

  build_splot_histograms:
    cmd: >-
      python ${project_root}/stages/build_histogram_targets.py
      --params params.yaml
      --source splot
      --partitions milan genoa
      --cpus 16
      --mem 64GB
      --time 02:00:00
    deps:
      - ${project_root}/src/data/build_histogram_targets.py
      - ${project_root}/data/interim/splot/filtered_surveys/try6/splot_filtered.parquet
      - ${project_root}/data/interim/traits/try6_pow-xf/try6_pow-xf.parquet
    params:
      - histogram
      - splot
      - traits.names
    outs:
      - ${project_root}/${output.dir}/splot:
          persist: true
```

**Output Structure**:
```
data/interim/histogram_targets/try6_hist_pow-xf_22km/
├── gbif/
│   ├── histograms.parquet        # (n_cells, n_traits, n_bins) flattened
│   ├── masks.parquet             # (n_cells, n_traits) valid trait indicators
│   ├── coordinates.parquet       # (n_cells, 2) x, y coordinates
│   └── metadata.json             # Bin edges, trait names, etc.
├── splot/
│   ├── histograms.parquet
│   ├── masks.parquet
│   ├── coordinates.parquet
│   └── metadata.json
└── report.md                     # Summary statistics, coverage plots
```

---

### Phase 2: Training Data Preparation

**New Pipeline**: `pipeline/histogram_models/try6_hist_pow-xf_22km/`

**Purpose**: Merge histogram targets with EO features, assign folds, prepare for training.

**New Files Required**:

| File | Purpose |
|------|---------|
| `src/features/build_histogram_xy.py` | Merge histograms with EO features |
| `stages/build_histogram_xy.py` | Slurm/local entrypoint |

**Additional Stages in `dvc.yaml`**:
```yaml
  merge_histogram_sources:
    cmd: >-
      python ${project_root}/stages/build_histogram_xy.py
      --params params.yaml
      --partitions milan
      --cpus 32
      --mem 128GB
      --time 01:00:00
    deps:
      - ${project_root}/data/interim/histogram_targets/${product_code}/gbif/
      - ${project_root}/data/interim/histogram_targets/${product_code}/splot/
      - ${project_root}/data/interim/eo_data/modis_wc2_soil_canopy_vodca_alos_22km/
    outs:
      - ${project_root}/${train.dir}/${product_code}/xy_data:
          persist: true

  build_cv_splits:
    cmd: >-
      python ${project_root}/stages/build_cv_splits.py
      --params params.yaml
      --partition milan
      --cpus 8
      --mem 32GB
    deps:
      - ${project_root}/${train.dir}/${product_code}/xy_data/
    outs:
      - ${project_root}/${train.dir}/${product_code}/cv_splits/:
          persist: true
```

**Output Structure**:
```
data/features/try6_hist_pow-xf_22km/
├── xy_data/
│   ├── X.parquet                 # (n_samples, n_features) EO features
│   ├── Y.parquet                 # (n_samples, n_traits * n_bins) flattened histograms
│   ├── masks.parquet             # (n_samples, n_traits) valid trait indicators
│   ├── is_splot.parquet          # (n_samples,) boolean source indicator
│   └── coordinates.parquet       # (n_samples, 2) for spatial operations
├── cv_splits/
│   ├── fold_assignments.parquet  # (n_samples,) fold IDs
│   └── split_info.json           # Fold statistics, spatial ranges
└── metadata.json                 # Feature names, trait names, bin edges
```

---

### Phase 3: Histogram Model Training

**New Module**: `src/models/histogram_mlp/`

This is the key new implementation - a PyTorch-based histogram prediction model separate from AutoGluon.

**New Files Required**:

| File | Purpose |
|------|---------|
| `src/models/histogram_mlp/__init__.py` | Module init |
| `src/models/histogram_mlp/model.py` | HistogramMLP architecture |
| `src/models/histogram_mlp/loss.py` | Masked KL divergence loss |
| `src/models/histogram_mlp/dataset.py` | PyTorch Dataset class |
| `src/models/histogram_mlp/trainer.py` | Training loop with validation |
| `src/models/histogram_mlp/config.py` | Dataclass for model config |
| `stages/train_histogram_models.py` | Slurm/local entrypoint |

**DVC Stage**:
```yaml
  train_histogram_models:
    cmd: >-
      python ${project_root}/stages/train_histogram_models.py
      --params params.yaml
      --partition l40s
      --gpus 1
      --cpus 16
      --mem 64GB
      --time 04:00:00
    deps:
      - ${project_root}/src/models/histogram_mlp/
      - ${project_root}/${train.dir}/${product_code}/xy_data/
      - ${project_root}/${train.dir}/${product_code}/cv_splits/
    params:
      - model
      - training
    outs:
      - ${project_root}/${models.dir}/${product_code}:
          persist: true
```

**Key Parameters**:
```yaml
model:
  architecture: "simple_mlp"
  hidden_dims: [512, 256, 256]
  dropout: 0.2
  n_bins: 20
  n_traits: 31

training:
  learning_rate: 1e-3
  batch_size: 256
  n_epochs: 100
  optimizer: "adamw"
  weight_decay: 1e-4
  scheduler: "cosine"
  splot_weight: 2.0
  early_stopping_patience: 10

ensemble:
  n_models: 5
  seeds: [42, 123, 456, 789, 1011]
```

**Output Structure**:
```
models/try6_hist_pow-xf_22km/
├── fold_0/
│   ├── model_seed_42.pt          # Model checkpoint
│   ├── model_seed_123.pt
│   ├── ...
│   ├── training_log.csv          # Loss curves
│   └── validation_metrics.json   # KL div, histogram intersection, etc.
├── fold_1/
│   └── ...
├── ensemble_config.json          # Seeds, architecture, etc.
├── feature_importance.csv        # Permutation importance for AOA
└── training_report.md            # Summary with visualizations
```

---

### Phase 4: Inference & AOA

**New Files Required**:

| File | Purpose |
|------|---------|
| `src/models/histogram_mlp/predict.py` | Batch prediction with ensemble |
| `src/analysis/histogram_aoa.py` | AOA adapted for histogram models |
| `stages/predict_histograms.py` | Slurm/local entrypoint |

**DVC Stages**:
```yaml
  predict_histograms:
    cmd: >-
      python ${project_root}/stages/predict_histograms.py
      --params params.yaml
      --partition l40s
      --gpus 1
      --cpus 16
      --mem 64GB
      --time 02:00:00
    deps:
      - ${project_root}/${models.dir}/${product_code}/
      - ${project_root}/data/interim/eo_data/modis_wc2_soil_canopy_vodca_alos_22km/
    outs:
      - ${project_root}/${processed.dir}/${product_code}/predict:
          persist: true

  calculate_aoa:
    cmd: >-
      python ${project_root}/stages/calc_histogram_aoa.py
      --params params.yaml
      --partition l40s
      --gpus 1
      --time 01:00:00
    deps:
      - ${project_root}/${models.dir}/${product_code}/
      - ${project_root}/${train.dir}/${product_code}/xy_data/
    outs:
      - ${project_root}/${processed.dir}/${product_code}/aoa:
          persist: true

  build_final_products:
    cmd: >-
      python ${project_root}/stages/build_histogram_products.py
      --params params.yaml
      --cpus 16
      --mem 32GB
    deps:
      - ${project_root}/${processed.dir}/${product_code}/predict/
      - ${project_root}/${processed.dir}/${product_code}/aoa/
    outs:
      - ${project_root}/${processed.dir}/${product_code}/public:
          persist: true
```

**Output Structure**:
```
data/processed/try6_hist_pow-xf_22km/
├── predict/
│   ├── X4_histogram.tif          # (n_bins,) bands per trait
│   ├── X4_entropy.tif            # Histogram entropy (uncertainty)
│   ├── X4_ensemble_std.tif       # Ensemble std (uncertainty)
│   ├── X6_histogram.tif
│   └── ...
├── aoa/
│   ├── dissimilarity_index.tif   # DI values
│   └── aoa_mask.tif              # Binary AOA mask
├── derived/
│   ├── X4_mean.tif               # Mean derived from histogram
│   ├── X4_std.tif                # Std derived from histogram
│   ├── X4_skewness.tif           # Skewness derived from histogram
│   └── ...
└── public/
    ├── trait_histograms/         # Final packaged outputs
    └── metadata.json             # Full provenance
```

---

## Files to Create Summary

### New Source Files

| File | Lines (est.) | Priority |
|------|--------------|----------|
| `src/data/build_histogram_targets.py` | ~400 | Phase 1 |
| `src/features/build_histogram_xy.py` | ~200 | Phase 2 |
| `src/models/histogram_mlp/model.py` | ~150 | Phase 3 |
| `src/models/histogram_mlp/loss.py` | ~50 | Phase 3 |
| `src/models/histogram_mlp/dataset.py` | ~100 | Phase 3 |
| `src/models/histogram_mlp/trainer.py` | ~300 | Phase 3 |
| `src/models/histogram_mlp/predict.py` | ~150 | Phase 4 |
| `src/analysis/histogram_aoa.py` | ~200 | Phase 4 |

### New Stage Scripts

| File | Based On | Priority |
|------|----------|----------|
| `stages/build_histogram_targets.py` | `stages/build_moment_maps.py` | Phase 1 |
| `stages/build_histogram_xy.py` | `stages/prepare_xy_data.py` | Phase 2 |
| `stages/train_histogram_models.py` | New (PyTorch-specific) | Phase 3 |
| `stages/predict_histograms.py` | `stages/inference.py` | Phase 4 |
| `stages/calc_histogram_aoa.py` | `stages/inference.py` (aoa portion) | Phase 4 |
| `stages/build_histogram_products.py` | `stages/inference.py` (final portion) | Phase 4 |

### New Pipeline Configurations

| File | Priority |
|------|----------|
| `pipeline/histogram_data/try6_hist_pow-xf_22km/params.yaml` | Phase 1 |
| `pipeline/histogram_data/try6_hist_pow-xf_22km/dvc.yaml` | Phase 1 |
| `pipeline/histogram_models/try6_hist_pow-xf_22km/params.yaml` | Phase 2 |
| `pipeline/histogram_models/try6_hist_pow-xf_22km/dvc.yaml` | Phase 2 |

---

## Modifications to Existing Code

### Minor Modifications

| File | Change | Reason |
|------|--------|--------|
| `src/utils/df_utils.py` | Add histogram aggregation function | Reuse rasterization logic |
| `src/utils/training_utils.py` | Expose fold assignment utilities | Reuse spatial blocking |

### No Modifications Needed

- `src/pipeline/entrypoint_utils.py` - Reuse as-is
- `src/conf/conf.py` - Config loading works for new params
- `src/analysis/aoa.py` - Reference implementation, create adapted version

---

## Execution Order

```
1. pipeline/traits/try6_pow-xf/                    [DONE]
   └── Outputs: data/interim/traits/try6_pow-xf/

2. pipeline/eo_data/modis_wc2_soil_canopy_vodca_alos_22km/  [DONE]
   └── Outputs: data/interim/eo_data/.../

3. pipeline/histogram_data/try6_hist_pow-xf_22km/  [NEW - Phase 1]
   ├── build_gbif_histograms
   └── build_splot_histograms
   └── Outputs: data/interim/histogram_targets/

4. pipeline/histogram_models/try6_hist_pow-xf_22km/ [NEW - Phase 2-4]
   ├── merge_histogram_sources
   ├── build_cv_splits
   ├── train_histogram_models
   ├── predict_histograms
   ├── calculate_aoa
   └── build_final_products
   └── Outputs: data/processed/try6_hist_pow-xf_22km/
```

## Next Steps

1. **Phase 1**: Implement `build_histogram_targets.py` and test histogram construction
2. **Phase 2**: Implement training data preparation, verify fold assignment
3. **Phase 3**: Implement PyTorch model and training loop
4. **Phase 4**: Implement inference and AOA calculation
5. **Validation**: Compare histogram-derived moments to direct moment predictions
