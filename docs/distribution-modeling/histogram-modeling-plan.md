# Histogram-Based Trait Distribution Modeling

## Overview

Model plant trait distributions per grid cell as probability histograms, using a custom MLP trained with Kullback-Leibler divergence loss. This approach captures the full distributional shape rather than just summary statistics (moments).

## Design Decisions

| Decision | Choice | Rationale | Status |
|----------|--------|-----------|--------|
| Histogram bins | 20 fixed equal-width | Balance between resolution and stability | ✅ |
| Bin domain | Yeo-Johnson transformed | More uniform distributions, better bin utilization | ✅ |
| Output format | Zarr v3 store | Efficient array storage with metadata attrs | ✅ |
| Grid projection | EPSG:6933 (Equal Area Cylindrical) 22 km | Consistent cell areas globally | ✅ |
| Model architecture | Joint multi-output MLP | Single model predicts all trait histograms, can learn trait correlations | ✅ |
| Loss function | Masked KL divergence | Natural measure for comparing probability distributions; per-trait masking + source weighting | ✅ |
| Framework | PyTorch (custom) | AutoGluon doesn't support histogram outputs with KL loss | ✅ |
| CV strategy | H3 spatial folds, sPlot-only validation | Joint model needs one fold per cell; sPlot is ground truth | ✅ |
| sPlot weighting | Loss weighting (`gbif_weight = n_splot / n_total`) | Cleaner than oversampling with masked loss | ✅ |
| HPO | Optuna with JournalFileStorage | Parallel Slurm workers, MedianPruner, NFS-safe | ✅ |

## Histogram Construction ✅

> **Status**: Fully implemented in `src/data/build_histogram_targets.py` with tests in
> `tests/data/test_build_histogram_targets.py` (19 tests passing). Output is a Zarr v3 store
> containing `histograms` (N, 31, 20), `masks` (N, 31), `coords` (N, 2), `bin_edges` (31, 21),
> and `total_weight` (N,). Construction is vectorized using `np.add.at` with integer cell IDs
> (`cx * 10_000_000 + cy`) for fast groupby in EPSG:6933 at 22 km resolution.
>
> A sanity-check report is generated automatically at the end of each stage run
> (`src/data/histogram_report.py`), producing PNG figures and a markdown summary in
> `{out_dir}/report/`.

### Step 1: Define Global Bin Edges

For each trait, compute bin edges from the **full species pool** (not per grid cell):

```python
# After Yeo-Johnson transformation
trait_values = transformed_try6_traits[trait].dropna()
bin_edges = np.linspace(trait_values.min(), trait_values.max(), n_bins + 1)
```

This ensures:
- Consistent bins across all grid cells (spatial comparability)
- Bins span the full range of possible values
- Edge cases: observations outside bin range go to first/last bin

### Step 2: Construct Histograms per Grid Cell

#### For GBIF Observations

```python
def build_gbif_histogram(cell_observations, trait, bin_edges, weights=None):
    """
    Build probability histogram from GBIF observations.

    Parameters
    ----------
    cell_observations : DataFrame
        Observations in this grid cell with trait values and weights
    trait : str
        Trait column name
    bin_edges : array
        Pre-computed bin edges for this trait
    weights : str, optional
        Column name for observation weights (e.g., resurvey weights)

    Returns
    -------
    histogram : array of shape (n_bins,)
        Probability histogram (sums to 1)
    """
    values = cell_observations[trait].values
    w = cell_observations[weights].values if weights else None

    # Weighted histogram
    counts, _ = np.histogram(values, bins=bin_edges, weights=w)

    # Normalize to probability distribution
    if counts.sum() > 0:
        histogram = counts / counts.sum()
    else:
        histogram = np.full(len(counts), 1.0 / len(counts))  # Uniform if empty

    return histogram
```

#### For sPlot Surveys

Two options for incorporating abundances:

**Option A: Abundance as weight (implemented ✅)**
```python
# Combined weight = Rel_Abund_Plot × weight (resurvey weight)
combined_weights = abundances * resurvey_weights
```

> **Implementation note**: NaN weights in sPlot's `weight` column (~10.7% of rows) caused
> silent bin poisoning via `np.add.at` (one NaN poisons an entire bin sum). Fixed by extending
> the per-trait `not_null` mask to also filter NaN weights before accumulation.

**Option B: Pseudo-observation expansion (not used)**
- Expand each species to `int(abundance × multiplier)` pseudo-observations
- Build histogram from pseudo-observations
- More memory-intensive but preserves existing pipeline logic

### Step 3: Quality Filtering

Not all grid cells will have reliable histograms. Apply source-specific filters:

**GBIF filtering:**
```python
min_observations = 30        # Minimum weighted observations
min_unique_species = 3       # Minimum unique species
min_bin_coverage = 0.25      # At least 25% of bins non-empty
```

**sPlot filtering:**
```python
min_total_abundance = 0.75   # Minimum cumulative abundance after trait matching
min_unique_species = 3       # Minimum unique species per cell
min_bin_coverage = 0.25      # At least 25% of bins non-empty
```

Note: sPlot uses fractional abundances (not counts), so we require 75% total abundance coverage rather than observation counts.

Cells failing these criteria are excluded from training.

### Step 4: Handle Sparse Histograms

Even with filtering, some histograms may be sparse. Options:

1. **Label smoothing**: Add small epsilon to all bins before normalizing
   ```python
   histogram = (counts + epsilon) / (counts + epsilon).sum()
   ```

2. **Kernel density smoothing**: Smooth the histogram with a Gaussian kernel

3. **Accept sparsity**: Let the model learn that sparse histograms have higher uncertainty

**Recommendation**: Use label smoothing with small epsilon (e.g., 0.01) to avoid zero probabilities which cause issues with KL divergence.

> **Implemented** ✅ with `epsilon=0.01` (configurable via `params.yaml`).

## Model Architecture

### Input

EO features for each grid cell (same as current CWM models):
- MODIS composites
- WorldClim bioclimatic variables
- Soil properties
- Canopy height
- VODCA
- etc.

**Input dimension**: ~100-200 features (depending on configuration)

### Output

Joint histogram prediction for all traits:
- 31 traits × 20 bins = 620 output values
- Reshaped as (n_traits, n_bins) internally
- **Softmax applied per trait** to ensure each histogram sums to 1

### Architecture Options

**Option 1: Simple MLP**
```
Input (n_features)
  → Linear(n_features, 512) + ReLU + Dropout(0.2)
  → Linear(512, 256) + ReLU + Dropout(0.2)
  → Linear(256, 256) + ReLU + Dropout(0.2)
  → Linear(256, n_traits × n_bins)
  → Reshape to (n_traits, n_bins)
  → Softmax(dim=-1)  # Per-trait normalization
```

**Option 2: Trait-Aware Architecture**
```
Input (n_features)
  → Shared encoder: Linear(n_features, 256) + ReLU
  → Per-trait heads: [Linear(256, 64) + ReLU + Linear(64, n_bins) + Softmax] × n_traits
```

This allows traits to share low-level feature representations while having specialized prediction heads.

**Option 3: Attention-Based**
```
Input (n_features)
  → Linear(n_features, 256)
  → Trait embeddings (learnable, n_traits × 64)
  → Cross-attention between features and trait embeddings
  → Per-trait MLP heads → Softmax
```

**Recommendation**: Start with Option 1 (simple MLP) for baseline, then explore Option 2 if performance is lacking.

## Loss Function

### KL Divergence

PyTorch provides `torch.nn.KLDivLoss` which is numerically stable and optimized. Key considerations:

1. **Input format**: `KLDivLoss` expects **log-probabilities** as input, so use `log_softmax` instead of `softmax` on model output
2. **Reduction**: Use `reduction='batchmean'` for proper averaging
3. **Target**: The observed histogram should be regular probabilities (not log)

```python
import torch.nn.functional as F

class HistogramMLP(nn.Module):
    def forward(self, x):
        # ... hidden layers ...
        logits = self.output_layer(x)  # (batch, n_traits * n_bins)
        logits = logits.view(-1, self.n_traits, self.n_bins)

        # Return log-probabilities for KLDivLoss
        return F.log_softmax(logits, dim=-1)

# Loss function
kl_loss = nn.KLDivLoss(reduction='batchmean')

# Usage in training loop
log_predicted = model(X_batch)  # (batch, n_traits, n_bins) log-probs
loss = kl_loss(log_predicted, observed_histograms)  # observed are regular probs
```

**Note**: `KLDivLoss` computes `D_KL(target || input)`, which is what we want - it measures how well the predicted distribution approximates the observed one.

### Alternative: Cross-Entropy

Cross-entropy is equivalent to KL divergence up to a constant (the entropy of the observed distribution):

```python
def cross_entropy_loss(log_predicted, observed, epsilon=1e-8):
    """Cross-entropy: -sum(observed * log(predicted))"""
    ce = -observed * log_predicted
    return ce.sum(dim=-1).mean()
```

Both have the same gradients, so optimization behavior is identical.

### Handling Missing Traits

Not all species have all traits (NaN values in TRY). Options:

1. **Mask missing traits**: Only compute loss for traits with valid histograms
2. **Exclude incomplete cells**: Only train on cells with all traits present
3. **Impute**: Fill missing traits with global distribution (introduces bias)

**Recommendation**: Mask missing traits during loss computation.

**Important clarification**: The mask operates **per-trait, not per-cell**:
- If a grid cell has valid histograms for traits A, B, C but NaN for trait D:
  - The model still **predicts** histograms for all 4 traits (A, B, C, D)
  - But the **loss** is only computed for traits A, B, C (trait D's contribution is masked out)
- This means all traits are always predicted, but the model only learns from valid observations
- This maximizes training data usage without requiring complete trait coverage

```python
def masked_kl_loss(log_predicted, observed, mask):
    """
    KL divergence with masking for missing traits.

    Parameters
    ----------
    log_predicted : Tensor of shape (batch, n_traits, n_bins)
        Log-probabilities from model (after log_softmax)
    observed : Tensor of shape (batch, n_traits, n_bins)
        Observed probability histograms
    mask : Tensor of shape (batch, n_traits)
        1 where trait histogram is valid, 0 where missing

    Returns
    -------
    loss : Tensor
        Mean KL divergence over valid trait-cell combinations
    """
    # KL divergence per trait: sum over bins
    kl = F.kl_div(log_predicted, observed, reduction='none').sum(dim=-1)  # (batch, n_traits)

    # Apply mask and compute mean over valid entries only
    masked_kl = kl * mask
    return masked_kl.sum() / mask.sum().clamp(min=1)
```

## Training Pipeline

### Data Preparation

1. **Build histograms** for all valid grid cells (GBIF and sPlot as separate records)
2. **Store as tensors**: `(n_cells, n_traits, n_bins)` for Y, `(n_cells, n_features)` for X
3. **Record masks** for missing traits: `(n_cells, n_traits)`
4. **Record source**: Track which samples are GBIF vs sPlot
5. **Assign spatial folds**: All observations (GBIF + sPlot) get fold IDs based on location

### Validation Strategy: sPlot-Only Holdout

sPlot vegetation surveys are far more reliable than GBIF citizen science observations. Therefore, we validate **only against held-out sPlot histograms**.

**Spatial k-fold CV with sPlot-only validation:**
1. Assign spatial fold IDs to all observations (both GBIF and sPlot) based on grid cell location
2. For each fold i:
   - **Training set**: All GBIF + sPlot histograms from folds ≠ i
   - **Validation set**: Only sPlot histograms from fold i (GBIF from fold i is excluded from both training and validation)
3. Report metrics only on sPlot validation samples

```python
def get_cv_split(fold_idx, fold_ids, is_splot):
    """
    Get train/val indices for a given fold.

    Parameters
    ----------
    fold_idx : int
        Current fold index (0 to k-1)
    fold_ids : array
        Spatial fold assignment for each sample
    is_splot : array
        Boolean mask indicating sPlot samples

    Returns
    -------
    train_mask, val_mask : arrays
        Boolean masks for training and validation samples
    """
    in_fold = fold_ids == fold_idx

    # Training: all samples NOT in this fold
    train_mask = ~in_fold

    # Validation: only sPlot samples IN this fold
    val_mask = in_fold & is_splot

    return train_mask, val_mask
```

**Rationale:**
- Spatial folds ensure no leakage between train/val regions
- GBIF in held-out fold is excluded to maintain spatial separation
- Validation uses only sPlot because it's ground truth
- This tests: "Can the model accurately predict vegetation survey distributions in unseen regions?"

### Training Loop

```python
model = HistogramMLP(n_features, n_traits, n_bins)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(n_epochs):
    for X_batch, Y_batch, mask_batch in train_loader:
        optimizer.zero_grad()

        predicted = model(X_batch)  # (batch, n_traits, n_bins)
        loss = masked_cross_entropy_loss(predicted, Y_batch, mask_batch)

        loss.backward()
        optimizer.step()

    scheduler.step()

    # Validation
    val_loss = evaluate(model, val_loader)
    log(f"Epoch {epoch}: train_loss={loss:.4f}, val_loss={val_loss:.4f}")
```

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Learning rate | 1e-3 | With cosine annealing |
| Batch size | 256 | Adjust based on memory |
| Hidden dims | [512, 256, 256] | Start simple |
| Dropout | 0.2 | Regularization |
| Weight decay | 1e-4 | L2 regularization |
| Epochs | 100-200 | Early stopping on val loss |
| Label smoothing | 0.01 | Epsilon added to histograms |

## Evaluation Metrics

### Primary: KL Divergence / Cross-Entropy

Same as training loss, evaluated on held-out test set.

### Secondary Metrics

1. **Earth Mover's Distance (EMD)**: Measures "work" to transform one histogram into another
   - More intuitive than KL for comparing distributions
   - Captures that "nearby bins" are more similar

2. **Histogram Intersection**: Overlap between predicted and observed
   ```python
   intersection = np.minimum(predicted, observed).sum()
   ```

3. **Derived Moment Comparison**: Compute moments from predicted histograms and compare to observed
   - Mean: weighted average of bin centers
   - Variance: from bin centers and probabilities
   - Skewness, Kurtosis: higher moments

4. **Bin-wise R²**: For each bin, compute R² across all test cells

## Module Structure

```
src/
  data/
    build_histogram_targets.py   # ✅ Construct histograms from GBIF/sPlot → Zarr
    histogram_report.py          # ✅ Sanity-check figures + markdown report
  features/
    build_histogram_xy.py        # ✅ Merge EO features with histogram targets → train.zarr
  models/
    histogram_mlp/               # ✅ Phase 2
      __init__.py
      model.py                   # HistogramMLP: MLP → Reshape → LogSoftmax
      loss.py                    # MaskedKLDivLoss with source weighting
      dataset.py                 # Zarr loader, preprocessing, PyTorch Dataset
      cv_splits.py               # H3 spatial folds, sPlot-only validation splits
      evaluate.py                # KL, EMD, histogram intersection, moment comparison
      train.py                   # Training loop, CV orchestration, CLI
      hpo.py                     # ✅ Optuna HPO (search space, objective, study mgmt)
    run_utils.py                 # Run ID generation (run_YYYYMMDD_HHMMSS)

stages/
  build_histogram_xy.py          # ✅ Slurm/local entry point for XY merge
  train_histogram_model.py       # ✅ Slurm/local entry point for training (GPU)
  run_histogram_hpo.py           # ✅ Slurm entry point for parallel HPO workers

tests/
  data/
    test_build_histogram_targets.py  # ✅ 19 tests incl. int16 overflow regression
  models/
    conftest.py                  # ✅ Shared fixtures (rng, dims, synthetic_data)
    test_histogram_mlp.py        # ✅ Model, loss, dataset, CV, metrics, integration tests
    test_hpo.py                  # ✅ Search space, study, pruning callback tests

pipeline/
  histogram_data/
    try6_hist_pow-xf_22km/       # ✅ DVC pipeline (build_gbif_histograms, build_splot_histograms)
  products/
    try6_hist_pow-xf_22km/       # ✅ DVC pipeline (build_histogram_xy, train_histogram_model)
```

## Implementation Phases

### Phase 1: Histogram Construction ✅

- [x] Implement `build_histogram_targets.py` to create histogram targets
  - Vectorized construction via `np.add.at` with integer cell ID packing (`cx * 10_000_000 + cy`)
  - Separate GBIF and sPlot stages with source-specific filtering
- [x] Add bin edge computation with Yeo-Johnson transformation
  - Per-trait `PowerTransformer` pickles for forward/inverse transforms
- [x] Output: Zarr store with `histograms` (N, 31, 20), `masks` (N, 31), `coords` (N, 2), `bin_edges` (31, 21), `total_weight` (N,)
- [x] Quality filtering (min_observations, min_unique_species, min_bin_coverage, min_total_abundance)
- [x] Label smoothing (epsilon=0.01)
- [x] sPlot abundance weighting (Option A: combined_weight = Rel_Abund_Plot × weight)
- [x] Sanity-check report generation (spatial coverage, per-trait validity/mean maps, entropy, sample distributions, entropy-vs-observations)
- [x] DVC pipeline with Slurm execution (`pipeline/histogram_data/try6_hist_pow-xf_22km/`)
- [x] Test suite (19 tests) including int16 overflow regression test

**Bugs fixed during Phase 1:**
- **int16 overflow**: `pd.Categorical.codes` returns int16 for <32,768 categories; `code * n_bins` overflows at code 1639 with n_bins=20, causing cross-cell histogram contamination. Fix: `.astype(np.int64)`.
- **NaN weight poisoning**: `np.add.at` propagates NaN — one NaN weight poisons an entire bin. Fix: filter NaN weights in the per-trait `not_null` mask.

### Phase 2: Model Implementation & Training Pipeline ✅

- [x] `HistogramMLP` nn.Module: configurable hidden_dims, dropout, LogSoftmax per-trait output
- [x] `MaskedKLDivLoss`: per-trait masking, sPlot/GBIF source weighting (`gbif_weight_factor`)
- [x] `HistogramDataset` + `preprocess_features`: sentinel→NaN, median imputation, standardization
- [x] `assign_spatial_folds`: H3 hex-based spatial CV with fold balance optimization
- [x] `evaluate_all`: KL divergence, EMD, histogram intersection, moment comparison (mean R², MAE)
- [x] Training loop with early stopping, CosineAnnealingLR, AdamW, completion flags for resumability
- [x] CV orchestration (`run_cv`): 5-fold spatial CV + full model training
- [x] Optuna HPO integration: `hpo.py` with `JournalFileStorage`, `MedianPruner`, parallel Slurm workers
- [x] DVC stage `train_histogram_model` with GPU partition (`l40s`)
- [x] Slurm entry points: `stages/train_histogram_model.py`, `stages/run_histogram_hpo.py`
- [x] Unit tests for all modules + HPO integration tests
- [x] `optuna-dashboard` available for HPO monitoring (dev dependency)

### Phase 3: Evaluation & Comparison
- [ ] Run HPO to find optimal hyperparameters
- [ ] Full 5-fold CV with best hyperparameters
- [ ] Compare to moment-based approach (derive moments from histograms)
- [ ] Visualization of predicted vs observed histograms

### Phase 5: Production
- [ ] Prediction pipeline for new EO data
- [ ] Output format: GeoTIFF with n_bins bands per trait
- [ ] Integration with downstream applications

## Design Decisions (Resolved)

### 1. GBIF + sPlot Data Handling

**Decision**: Combine GBIF and sPlot observations but keep them as **separate training records**.

- A single grid cell can contribute two training samples: one from GBIF, one from sPlot
- This doubles training data and lets the model learn from both sources
- The model doesn't need to know which source a sample came from

**sPlot weighting**: To give sPlot observations higher influence during training:

```python
# Option A: Sample weighting in loss
sample_weights = torch.where(is_splot, splot_weight, 1.0)  # e.g., splot_weight=2.0
loss = (per_sample_loss * sample_weights).mean()

# Option B: Oversampling with WeightedRandomSampler
weights = [splot_weight if is_splot[i] else 1.0 for i in range(len(dataset))]
sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
train_loader = DataLoader(dataset, sampler=sampler, batch_size=256)
```

### 2. Out-of-Range Predictions & Extrapolation Flagging

**Decision**: Use Area of Applicability (AOA) approach from `src/analysis/aoa.py`.

AOA is more principled than simple Mahalanobis distance because it:
- Uses **feature importance weighting** from the trained model
- Computes distance to **nearest training point** (not just centroid)
- Handles **non-Gaussian** training distributions
- Captures **local gaps** in training data coverage
- Uses **spatial cross-validation** for threshold calibration

**AOA workflow:**
1. Scale and standardize features
2. Weight features by model-derived feature importance
3. Compute average pairwise distance in training data
4. For each prediction point, find minimum distance to any training point (k-NN with k=1)
5. Calculate Dissimilarity Index: `DI = min_distance / avg_train_distance`
6. Threshold via IQR: `DI_threshold = P75 + 1.5 * (P75 - P25)`
7. Points with `DI > DI_threshold` are outside AOA

```python
# Simplified from src/analysis/aoa.py
def calc_aoa(train_features, pred_features, feature_importance):
    # Scale and weight
    train_scaled = scale_features(train_features, means, stds)
    train_weighted = weight_features(train_scaled, feature_importance)

    pred_scaled = scale_features(pred_features, means, stds)
    pred_weighted = weight_features(pred_scaled, feature_importance)

    # Average training distance for normalization
    avg_train_dist = pairwise_distances(train_weighted).mean()

    # Min distance from each prediction to training
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(train_weighted)
    min_distances, _ = nn.kneighbors(pred_weighted)

    # Dissimilarity Index
    di = min_distances / avg_train_dist

    # Threshold from training CV
    di_threshold = np.percentile(di_train, 75) + 1.5 * np.subtract(*np.percentile(di_train, [75, 25]))

    aoa_mask = di <= di_threshold  # True = within AOA
    return di, aoa_mask
```

**Note**: For histogram models, feature importance comes from analyzing which input features most affect output predictions (e.g., via gradient-based attribution or permutation importance).

### 3. Uncertainty Quantification

Multiple complementary approaches:

**A. Histogram entropy** (intrinsic to output):
```python
def histogram_entropy(probs):
    """Higher entropy = more uniform = less certain about distribution shape."""
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
```

**B. Ensemble variance**:
- Train N models (e.g., 5) with different random seeds
- Prediction uncertainty = variance across ensemble predictions
- More robust than single-model approaches

**C. MC Dropout** (Bayesian approximation):
```python
model.train()  # Keep dropout active during inference
predictions = [model(X) for _ in range(N_samples)]
mean_pred = torch.stack(predictions).mean(dim=0)
uncertainty = torch.stack(predictions).std(dim=0)
```

**Recommendation**: Use ensemble variance as primary uncertainty measure, with histogram entropy as a secondary indicator.

## Remaining Open Questions

1. **Computational cost**: 620 outputs vs ~31 for moments
   - Larger model, more memory
   - But still tractable for modern hardware
   - May need gradient checkpointing for very large batches

2. **Hyperparameter tuning**: How to efficiently search the hyperparameter space?
   - Learning rate, hidden dims, dropout, sPlot weight, etc.
   - Consider Optuna or Ray Tune for automated search

3. **Ensemble size**: How many models for ensemble uncertainty?
   - 3-5 models is typical trade-off between compute and stability

## References

- Kullback-Leibler divergence: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
- Earth Mover's Distance: https://en.wikipedia.org/wiki/Earth_mover%27s_distance
- Histogram loss in deep learning: Various papers on ordinal regression and distribution learning
