# Distribution Modeling Discussion with Carsten

**Date:** 2026-01-22
**Participants:** Daniel Lusk, Carsten
**Topic:** Modeling trait distributions using statistical moments instead of quantiles

---

## Summary

Discussion on improving the approach for modeling plant trait distributions at the grid cell level. The main proposal is to move away from directly modeling quantiles (which has continuity issues) to modeling the four statistical moments (mean, variance, skewness, kurtosis) and reconstructing distributions using the Pearson family of distributions.

---

## Key Discussion Points

### 1. Problems with Current Quantile Approach

- Modeling quantiles directly (e.g., 5th, 25th, 75th percentiles) has issues:
  - Model doesn't "know" that 75th quantile is related to 25th quantile (non-continuity)
  - Getting better results by approaching the mean, which doesn't provide new information
  - Extreme quantiles depend on very few data points (e.g., 28 points in a dataset of 2 million)
  - Quantile regression can produce poor confidence intervals, especially at extremes

### 2. Proposed Solution: Pearson Family of Distributions

- **Karl Pearson (1901)** developed a general approach using four moments:
  1. Mean (1st moment)
  2. Variance (2nd central moment) → standard deviation
  3. Skewness (3rd standardized moment)
  4. Kurtosis (4th standardized moment)

- The Pearson family encompasses: normal, log-normal, gamma, beta, uniform, and Student's t distributions
- **Note:** Does not include heavy-tailed distributions like Cauchy, whose moments are undefined
- Flexible for unimodal or edge-bimodal distributions
- From these four moments, you can reconstruct a PDF and derive any quantiles needed

**Advantages:**
- More robust way of computing quantiles
- Less sensitive to sample size
- Range becomes 95% quantile range instead of min-max (more robust)

### 3. Implementation Approaches

#### Option A: Simple Lookup (Recommended Starting Point)
- Compute four moments directly per cell (split second computation)
- Use these to reconstruct distribution via Pearson equations
- Derive quantiles from reconstructed distribution

#### Option B: Neural Network for Multi-Output Prediction
- Model all four moments simultaneously as function of Earth observation data
- Custom loss function considerations:

  **For accurate prediction of all moments:**
  ```
  L = (ŷ_mean - y_mean)² + (ŷ_sd - y_sd)² +
      λ₃(ŷ_skew - y_skew)² + λ₄(ŷ_kurt - y_kurt)²
  ```
  Use smaller weights (λ₃, λ₄ ~ 0.1) on higher moments since they're harder to estimate accurately.

  **For regularization toward normality:**
  ```
  L = prediction_loss + λ₃(ŷ_skew)² + λ₄(ŷ_kurt - 3)²
  ```
  Note: Normal distribution has kurtosis = 3, so penalize deviation from 3, not 0.

- **Clarification:** Higher exponents (⁴, ⁶) penalize large errors more severely but don't inherently push toward normality—that requires explicit regularization terms.

### 4. Handling Missing Data: Bayesian Hierarchical Approach

**Problem:** Many cells have missing trait values; want to inform sparse cells from similar cells with more data

**Solution concepts:**
- Bayesian hierarchical model where each cell is a variation on underlying theme
- Similar to species distribution modeling for rare species
- Environment + known traits inform predictions for missing traits
- Multiple Imputation using Chained Equations (MICE) - iterative approach with ~200 iterations

**Auto-encoder approach:**
- Encode sparse data to latent space
- Reconstruct missing values in decoder
- Train imputation alongside main model
- Loss function only evaluated on observed values, but latent space captures correlations

### 5. Functional Diversity Considerations

**Challenge:** To compute functional diversity, need covariance between traits, not just marginal distributions

- Marginal distributions per trait don't tell you about joint distribution
- Two traits with same marginals could have very different correlations
- If modeling all traits simultaneously with 4 moments each:
  - 10 traits = 40 moment values per cell
  - Need covariance structure between traits to combine distributions
  - Options: one 40×40 covariance matrix (all moment-trait combinations), or four 10×10 matrices (one per moment type, ignoring cross-moment correlations)
  - "A lot of parameters, but necessary for combining distributions"

**Simpler approach:** Model traits individually, predict globally, then compute covariances on output stacks

### 6. Sample Size Corrections

**Critical point:** Kurtosis and skewness calculations need sample size corrections

- Standard formulas give biased estimates
- Need corrections like n-4 terms for kurtosis (see Jones & Gill 1998)
- The paper by Gross et al. (2017) uses uncorrected formulas - likely wrong
- Higher moments have higher uncertainty, harder to estimate correctly

### 7. Log Transformation of Traits

- Growth is exponential process → traits affecting growth should be log-transformed
- Argument: "All traits should be log-transformed" for biological reasons
- Exception: Height is often bimodal (herbs vs trees), may not need log transform
- Consider separating woody vs non-woody plants (Sandra Díaz's work on global spectrum of plant form)

### 8. Weighting Observations from Multiple Plots

**Problem:** Two plots in one pixel - one with 2 species, one with 400 species

**Solution:** Normalize abundance per plot so each plot contributes equally:
- Scale abundances so each plot sums to same value (e.g., 1)
- Avoids biasing toward species-rich plots
- "That's what I would do" - Carsten

**Jensen's Inequality / Moment Aggregation Warning:**
- Higher moments are **not additive** across subpopulations
- Cannot average pre-computed kurtosis or skewness values from separate plots
- Must compute moments from pooled weighted observations instead
- Mean is linear (safe to weight-average), but variance/skewness/kurtosis require recomputation from raw data
- "Every time it has fins, I get careful" - use caution with non-linear aggregations

---

## Action Items

- [ ] Implement moment calculation per grid cell (mean, sd, skewness, kurtosis)
- [ ] **Decide on kurtosis convention** (raw β₂ vs excess kurtosis) and document consistently
- [ ] Research Pearson distribution libraries in Python (scipy.stats, pearson package)
- [ ] Verify kurtosis/skewness calculations include sample size corrections (Fisher's corrections)
- [ ] Test reconstructing distributions from moments
- [ ] Talk to Ayushi about neural network implementation for multi-output moment prediction
- [ ] Investigate MICE/iterative imputation for missing trait values
- [ ] Consider separating woody/non-woody for trait distributions
- [ ] **Ensure moments are computed from pooled observations**, not averaged across plots

---

## Key References

- **Pearson family of distributions** - Karl Pearson (1901)
- **Jones & Gill (1998)** - Sample size corrections for moments
- **Gross et al. (2017)** - [Relating plant functional diversity to ecosystem functioning](https://www.nature.com/articles/s41559-017-0132) (uses uncorrected kurtosis - potential issue)
- **Sandra Díaz** - Global spectrum of plant form and function
- **MICE package** - Multiple Imputation using Chained Equations (R and Python)
- **Daniel Kahneman** - "Thinking Fast and Slow" (Jensen's inequality intuition)

---

## Technical Notes

### Pearson Distribution Parameters
- Four parameters needed: typically mean (μ), sigma (σ), and two shape parameters
- Distribution type (1-7) determined by ratio of kurtosis to skewness (β₁, β₂)
- Type 2 is special case of Type 3
- R package: `PearsonDS`; Python: `scipy.stats` has individual distributions, or see `pearson` package

### Kurtosis Convention
**Important:** Clarify which convention is being used:
- **Kurtosis (β₂):** μ₄/σ⁴ — normal distribution = 3
- **Excess kurtosis:** μ₄/σ⁴ - 3 — normal distribution = 0

Most software (scipy, pandas) defaults to **excess kurtosis**. Pearson's original formulation uses raw kurtosis (β₂).

### Sample Size Corrections (Jones & Gill 1998)
- Sample skewness and kurtosis are biased estimators
- Fisher's corrections needed, especially for small n
- Corrections involve terms like (n-1), (n-2), (n-3) in denominators
- Standard formula μ₄/σ⁴ underestimates population kurtosis

### Moment Aggregation
When combining observations from multiple plots:
- **Mean:** Can weight-average plot means (linear)
- **Variance, Skewness, Kurtosis:** Must recompute from pooled raw data, not average pre-computed values
