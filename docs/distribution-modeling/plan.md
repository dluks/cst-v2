# Distribution Modeling Plan for CST-v2

**Version:** 0.2
**Date:** 2026-01-23

---

## Background

The current approach in cst-v2 models community-weighted means (CWM) of plant traits. To capture functional diversity, we need to characterize the **distribution** of traits within each grid cell, not just the mean.

**Previous approach (quantile modeling)** had issues:
- Non-continuity: model doesn't know 75th quantile relates to 25th
- Extreme quantiles depend on very few data points
- Results converge toward mean, providing little new information

**New approach**: Model the 4 statistical moments (mean, variance, skewness, kurtosis) and reconstruct distributions to derive quantiles.

## Goals

1. **Primary**: Compute robust trait distribution descriptors (quantiles, ranges) per grid cell
2. **Secondary**: Use these as proxies for functional diversity in ecosystem functioning analyses
3. **Future**: Full functional diversity metrics requiring trait covariances

## Technical Approach

### Data Transformation Strategy

Since raw trait distributions are highly skewed (gamma-like), we will:

1. **Power-transform** trait data (Yeo-Johnson or log) to near-normality
2. Compute moments on **transformed scale**
3. Reconstruct distributions / compute quantiles on transformed scale
4. **Back-transform** quantiles to original scale for interpretation

This simplifies distribution reconstruction—near-normal distributions can use simpler methods.

### Moment Calculation

For each grid cell, compute from **pooled weighted observations**:

| Moment | Definition | Notes |
|--------|------------|-------|
| Mean (μ) | 1st moment | Can weight-average across plots |
| Variance (σ²) | 2nd central moment | Must recompute from pooled data |
| Skewness (γ₁) | 3rd standardized moment | Must recompute; use Fisher's correction |
| Kurtosis (β₂) | 4th standardized moment | Must recompute; use Fisher's correction |

**Critical**: Do NOT average pre-computed moments across plots (Jensen's inequality).

**Convention**: Use raw kurtosis (β₂, normal = 3) for Pearson system compatibility, but document clearly.

### Distribution Reconstruction

**Hybrid approach** based on post-transformation distribution shape:

```
For each cell:
    if |skewness| < 0.5 AND |excess_kurtosis| < 1.0:
        → Use normal distribution quantiles
    else:
        → Use Gram-Charlier expansion (statsmodels.sandbox.distributions.extras.pdf_mvsk)
        → OR use Pearson system via rpy2 + PearsonDS
```

**Python implementation options**:

| Method | Package | Pros | Cons |
|--------|---------|------|------|
| Normal approx | `scipy.stats.norm` | Simple, fast | Assumes normality |
| Gram-Charlier | `statsmodels` | Pure Python, uses 4 moments | Can fail for high skew/kurt |
| Pearson system | `rpy2` + R's `PearsonDS` | Handles all shapes | Requires R |
| Fit scipy dist | `scipy.stats.gamma`, etc. | Guaranteed valid PDF | Need to select distribution |

**Recommendation**: Start with normal approximation (post-transformation), fall back to Gram-Charlier or Pearson for outlier cells.

### Weighting Observations

When combining S-Plot surveys and citizen science observations:

1. Normalize abundances **per plot** so each plot sums to 1
2. Pool all weighted observations within grid cell
3. Compute moments from pooled data

This avoids bias toward species-rich plots while respecting abundance information.

## Implementation Steps

### Phase 1: Moment Calculation Pipeline

- [ ] Add moment calculation to data processing pipeline
  - [ ] Implement Fisher-corrected skewness/kurtosis functions
  - [ ] Handle abundance weighting correctly
  - [ ] Pool observations across plots within cells (not average moments)
- [ ] Verify against known distributions (simulation tests)
- [ ] Document kurtosis convention (raw β₂ vs excess)

### Phase 2: Distribution Reconstruction

- [ ] Implement hybrid reconstruction approach:
  - [ ] Normal quantiles for near-normal cells
  - [ ] Gram-Charlier for moderate deviations
  - [ ] Pearson system fallback (optional, via rpy2)
- [ ] Validate quantile estimates against empirical quantiles (where sample size sufficient)
- [ ] Implement back-transformation to original trait scale

### Phase 3: Integration with Ecosystem Functioning Analysis

- [ ] Compute trait ranges (e.g., 95th - 5th percentile) as diversity proxies
- [ ] Extract values at flux tower locations
- [ ] Test relationships with ecosystem functional properties

### Phase 4 (Future): Full Functional Diversity

- [ ] Compute trait covariances from predicted moment surfaces
- [ ] Investigate multi-trait neural network approach
- [ ] Consider Bayesian hierarchical imputation for sparse cells

## Technical Notes: Kurtosis Conventions

There are **two separate issues** that must both be handled:

### 1. Raw vs Excess Kurtosis (Constant Shift)

| Type | Formula | Normal Distribution |
|------|---------|---------------------|
| **Raw kurtosis (β₂)** | μ₄/σ⁴ | = 3 |
| **Excess kurtosis** | μ₄/σ⁴ - 3 | = 0 |

- **scipy, pandas**: Default to excess kurtosis
- **Pearson system**: Requires raw kurtosis (β₂)

### 2. Sample Size Bias Corrections (Fisher's Corrections)

The naive kurtosis formula is a **biased estimator**—it systematically underestimates true kurtosis for finite samples. Fisher's corrections adjust for sample size with terms involving (n-1), (n-2), (n-3).

**This is what Carsten warned about**: Gross et al. (2017) uses uncorrected formulas, which is problematic when sample sizes vary across grid cells.

### Correct scipy Usage

```python
from scipy.stats import kurtosis, skew

# Step 1: Compute bias-corrected excess kurtosis (scipy default with bias=False)
excess_kurt = kurtosis(data, fisher=True, bias=False)
skewness = skew(data, bias=False)

# Step 2: Convert to raw kurtosis for Pearson system
raw_kurt = excess_kurt + 3
```

**Parameters explained**:
- `fisher=True`: Return excess kurtosis (subtract 3)
- `bias=False`: Apply Fisher's sample size correction

## Key Decisions

| Decision | Options | Recommendation |
|----------|---------|----------------|
| Kurtosis convention | Raw (β₂) vs Excess | Raw (β₂) for Pearson compatibility |
| Sample size correction | Corrected vs Uncorrected | **Corrected** (bias=False in scipy) |
| Transformation | Log vs Yeo-Johnson | Yeo-Johnson (handles zeros/negatives) |
| Near-normal threshold | Skew < 0.5, Kurt < 1? | Test empirically |
| Reconstruction method | Normal / Gram-Charlier / Pearson | Hybrid based on moments |
| Sample size minimum | n = ? | TBD based on moment stability |

## Key References

- **Karl Pearson (1901)** - Pearson distribution system
- **Jones & Gill (1998)** - Sample size corrections for moments
- **Gross et al. (2017)** - [Functional diversity & ecosystem functioning](https://www.nature.com/articles/s41559-017-0132) (note: uses uncorrected kurtosis)
- **R PearsonDS package** - [CRAN](https://cran.r-project.org/web/packages/PearsonDS/PearsonDS.pdf)
- **statsmodels Gram-Charlier** - [pdf_mvsk](https://www.statsmodels.org/stable/generated/statsmodels.sandbox.distributions.extras.pdf_mvsk.html)

## Related Files

- Meeting notes: [2026-01-22_distribution-modeling-chat_notes.md](2026-01-22_distribution-modeling-chat_notes.md)
- CST-v2 project: `projects/cst/cst-v2/`
