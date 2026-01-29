# Statistical Moments Pipeline Implementation Plan

## Overview

Compute the 4 statistical moments (mean, variance, skewness, kurtosis) per grid cell for plant trait distributions using GBIF and sPlot data sources.

## Key Constraints

From `docs/distribution-modeling/plan.md`:
- **Pool observations within grid cells** before computing moments (NOT average pre-computed moments - Jensen's inequality)
- **Use Fisher's bias corrections** for skewness/kurtosis (`scipy bias=False`)
- **Use raw kurtosis (β₂)** where normal=3, for Pearson system compatibility
- **Normalize abundances per plot** so each plot contributes equally

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Resolution | **22km** | Match new EO data; more observations per cell for reliable moment estimation |
| n_min for moments | 30 | Higher moments need more samples for reliability |
| GBIF/sPlot maps | Separate products | Different biases; allows comparison |
| sPlot min abundance | 0.75 | Filter plots where <75% of abundance matched to traits (unreliable composition) |
| sPlot multiplier | 100 | Convert fractional abundances to pseudo-counts (0.4 → 40 obs) |
| sPlot normalization | **No normalization** | Plots contribute proportional to their trait coverage; 80% matched → ~80 pseudo-obs |
| Module location | New `src/utils/moment_utils.py` | Single-responsibility; clean separation |
| Pipeline location | `pipeline/sparse_trait_maps/try6_stg_dist_no-xf_22km/` | Follows existing sparse_trait_maps pattern; "dist" indicates distribution moments |

## Implementation

### Phase 1: Core Moment Functions

**New file: `src/utils/moment_utils.py`**

```python
def weighted_mean(values, weights) -> float
def weighted_variance(values, weights, bias_corrected=True) -> float
def weighted_skewness(values, weights, bias_corrected=True) -> float
def weighted_kurtosis(values, weights, excess=False, bias_corrected=True) -> float
def effective_sample_size(weights) -> float
def compute_moments(values, weights, n_min=30) -> dict[str, float]
```

Key implementation details:
- Variance uses reliability weights correction (Bessel's analog)
- Skewness/kurtosis use Fisher's g1/g2 corrections
- Kurtosis returns raw β₂ (normal=3) by default
- Returns NaN when n_eff < n_min

### Phase 2: Extend Grid Aggregation

**Modify: `src/utils/df_utils.py`**

Add to `agg_df()` supported functions:
- `variance`, `skewness`, `kurtosis`, `n_eff`

Update `compute_weighted_stats()` to call moment functions.

### Phase 3: Build Scripts

**New file: `src/data/build_moment_map.py`**

For GBIF:
1. Load filtered GBIF + traits
2. Join on specieskey
3. Reproject coordinates
4. Group by grid cell, compute moments via `agg_df()`
5. Rasterize and write

For sPlot:
1. Load filtered sPlot + traits
2. Join on species name (some species will be dropped due to missing trait data)
3. **Filter plots by minimum abundance** - keep only plots where cumulative abundance ≥ 0.75 (75% of original plot composition retained after trait matching)
4. **Expand abundances to pseudo-observations** (multiply by 100) - no normalization, so plots with 80% matched contribute ~80 pseudo-obs
5. Pool all pseudo-obs within each grid cell (preserving resurvey weights)
6. Compute moments from pooled data
7. Rasterize and write

**New function: `expand_splot_abundances()`**
```python
def expand_splot_abundances(df, trait_col, abundance_col, multiplier=100):
    """
    Convert fractional abundances to pseudo-observations.
    Species with Rel_Abund_Plot=0.4 → 40 pseudo-observations.
    """
```

**New file: `stages/build_moment_maps.py`**
- SLURM wrapper dispatching one job per trait

### Phase 4: Pipeline Structure

**New directory: `pipeline/sparse_trait_maps/try6_stg_dist_no-xf_22km/`**

Follows existing sparse_trait_maps pattern with "dist" suffix indicating distribution moments.

`dvc.yaml`:
```yaml
params:
  - params.yaml

stages:
  build_gbif_maps:
    cmd: >-
      python ${project_root}/stages/build_moment_maps.py
      --params params.yaml
      --source gbif
      --partitions milan genoa
      --cpus 61
      --mem 450GB
      --time 02:00:00
    deps:
      - ${project_root}/src/data/build_moment_map.py
      - ${project_root}/src/utils/moment_utils.py
      - ${project_root}/data/interim/gbif/filtered_occurrences/${trait_type}/gbif_filtered.parquet
      - ${project_root}/data/interim/traits/${trait_type}_no-xf/${trait_type}_no-xf.parquet
    outs:
      - ${project_root}/${gbif.maps.out_dir}/${product_code}:
          persist: true

  build_splot_maps:
    cmd: >-
      python ${project_root}/stages/build_moment_maps.py
      --params params.yaml
      --source splot
      --partitions milan genoa
      --cpus 61
      --mem 256GB
      --time 02:00:00
    deps:
      - ${project_root}/src/data/build_moment_map.py
      - ${project_root}/src/utils/moment_utils.py
      - ${project_root}/data/interim/splot/filtered_surveys/${trait_type}/splot_filtered.parquet
      - ${project_root}/data/interim/traits/${trait_type}_no-xf/${trait_type}_no-xf.parquet
    outs:
      - ${project_root}/${splot.maps.out_dir}/${product_code}:
          persist: true
```

`params.yaml`:
```yaml
version: "2"
project_root: "../../.."
product_dir: "pipeline/sparse_trait_maps/try6_stg_dist_no-xf_22km"
trait_type: "try6"
product_code: "try6_stg_dist_no-xf_22km"

model_res: "22km"
PFT: "Shrub_Tree_Grass"
random_seed: 42
base_resolution: 22000
target_resolution: 22000
crs: "EPSG:6933"

# Moment-specific parameters
moments:
  n_min: 30                        # Minimum observations for moment computation (GBIF)
  splot_min_abundance: 0.75        # Minimum cumulative abundance after trait matching
  splot_abundance_multiplier: 100  # Convert fractions to pseudo-counts
  kurtosis_excess: false           # Use raw kurtosis (beta_2, normal=3)

traits:
  interim_out: data/interim/traits/try6_no-xf/try6_no-xf.parquet
  names:
    - X4   # Stem specific density
    - X6   # Root rooting depth
    # ... (same traits as existing sparse_trait_maps)

gbif:
  filtered:
    out_dir: data/interim/gbif/filtered_occurrences
    fp: gbif_filtered.parquet
  maps:
    out_dir: data/interim/gbif/maps
    min_count: 10
    max_count: 500

splot:
  filtered:
    out_dir: data/interim/splot/filtered_surveys
    fp: splot_filtered.parquet
  maps:
    out_dir: data/interim/splot/maps
```

## Output Structure

```
data/interim/gbif/maps/try6_stg_dist_no-xf_22km/
  X4.tif  # Multi-band: mean, variance, skewness, kurtosis, n_eff, count
  X6.tif
  ...

data/interim/splot/maps/try6_stg_dist_no-xf_22km/
  X4.tif
  ...
```

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/utils/moment_utils.py` | CREATE - core moment functions |
| `src/utils/df_utils.py` | MODIFY - add moment support to agg_df() |
| `src/data/build_moment_map.py` | CREATE - main build script |
| `stages/build_moment_maps.py` | CREATE - SLURM wrapper |
| `pipeline/sparse_trait_maps/try6_stg_dist_no-xf_22km/dvc.yaml` | CREATE |
| `pipeline/sparse_trait_maps/try6_stg_dist_no-xf_22km/params.yaml` | CREATE |
| `tests/utils/test_moment_utils.py` | CREATE - unit tests |

## Testing Strategy

1. **Unit tests** for moment functions against scipy on unweighted data
2. **Fisher correction validation** - verify bias reduction on small samples
3. **Known distribution test** - gamma distribution has analytical moments
4. **Jensen's inequality test** - verify pooled moments ≠ averaged moments
5. **Integration test** - end-to-end pipeline on subset of traits

## Verification

1. Run unit tests: `pytest tests/utils/test_moment_utils.py -v`
2. Run on single trait: `python src/data/build_moment_map.py --trait X4 --source gbif`
3. Compare output mean band to existing sparse trait map mean
4. Validate skewness/kurtosis distributions are reasonable (skewness near 0 for many traits)

## sPlot Standards Research

Based on web search of sPlot documentation:
- Vegetation plots **record all plant species** co-occurring within delimited local areas (complete inventory)
- However, sPlot aggregates data with **disparate sampling protocols and plot sizes**
- Different plots may focus on different vegetation layers (herbaceous vs woody)
- **Recommendation**: Normalize per plot rather than weight by plot size to avoid bias from heterogeneous sampling protocols

Sources:
- [sPlot – A new tool for global vegetation analyses (2019)](https://onlinelibrary.wiley.com/doi/10.1111/jvs.12710)
- [sPlotOpen (2021)](https://onlinelibrary.wiley.com/doi/10.1111/geb.13346)
