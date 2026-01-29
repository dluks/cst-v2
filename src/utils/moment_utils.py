"""Utilities for computing statistical moments with weighted observations.

This module provides functions for computing the four statistical moments
(mean, variance, skewness, kurtosis) from weighted observations, with proper
bias corrections for finite samples.

Key conventions:
- Kurtosis returns raw β₂ (normal distribution = 3) by default, not excess kurtosis
- Fisher's bias corrections are applied by default for skewness and kurtosis
- Weights are treated as frequency weights (reliability weights)

References:
- Karl Pearson (1901) - Pearson distribution system
- Jones & Gill (1998) - Sample size corrections for moments
- scipy.stats documentation for Fisher's corrections
"""

import numpy as np
from numpy.typing import ArrayLike


def effective_sample_size(weights: ArrayLike) -> float:
    """Compute effective sample size for weighted data.

    The effective sample size accounts for the reduction in information
    when observations have varying weights. For equal weights, n_eff = n.

    Formula: n_eff = (sum(w))^2 / sum(w^2)

    This is needed for bias correction in higher moments.

    Parameters
    ----------
    weights : array-like
        Non-negative weights for each observation.

    Returns
    -------
    float
        Effective sample size.

    Examples
    --------
    >>> effective_sample_size([1, 1, 1, 1])  # Equal weights
    4.0
    >>> effective_sample_size([2, 2, 2, 2])  # Equal weights (scaled)
    4.0
    >>> effective_sample_size([1, 1, 1, 100])  # Unequal weights
    1.0594...
    """
    w = np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    w2_sum = (w**2).sum()

    if w2_sum == 0:
        return 0.0

    return (w_sum**2) / w2_sum


def weighted_mean(values: ArrayLike, weights: ArrayLike) -> float:
    """Compute weighted mean (1st moment).

    Parameters
    ----------
    values : array-like
        Data values.
    weights : array-like
        Non-negative weights for each observation.

    Returns
    -------
    float
        Weighted mean.

    Examples
    --------
    >>> weighted_mean([1, 2, 3], [1, 1, 1])
    2.0
    >>> weighted_mean([1, 2, 3], [1, 0, 0])
    1.0
    """
    return float(np.average(values, weights=weights))


def weighted_variance(
    values: ArrayLike,
    weights: ArrayLike,
    bias_corrected: bool = True,
) -> float:
    """Compute weighted variance (2nd central moment).

    Uses frequency weights interpretation with optional bias correction.
    The bias correction is analogous to Bessel's correction for unweighted
    variance (using n-1 instead of n in the denominator).

    Parameters
    ----------
    values : array-like
        Data values.
    weights : array-like
        Non-negative weights for each observation.
    bias_corrected : bool, default True
        If True, apply bias correction using effective sample size.
        This is the weighted analog of using n-1 in the denominator.

    Returns
    -------
    float
        Weighted variance.

    Notes
    -----
    For reliability (frequency) weights, the unbiased variance is:

        V1 = sum(w)
        V2 = sum(w^2)
        variance = V1 / (V1^2 - V2) * sum(w * (x - mean)^2)

    This reduces to the standard n/(n-1) correction for equal weights.

    Examples
    --------
    >>> weighted_variance([1, 2, 3], [1, 1, 1], bias_corrected=False)
    0.666...
    >>> weighted_variance([1, 2, 3], [1, 1, 1], bias_corrected=True)
    1.0
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    mean = np.average(v, weights=w)
    v1 = w.sum()  # Sum of weights
    v2 = (w**2).sum()  # Sum of squared weights

    # Weighted sum of squared deviations
    ss = np.sum(w * (v - mean) ** 2)

    if bias_corrected:
        # Reliability weights correction (Bessel's correction analog)
        denom = v1 - v2 / v1
        if denom <= 0:
            return np.nan
        return float(ss / denom)
    else:
        return float(ss / v1)


def weighted_skewness(
    values: ArrayLike,
    weights: ArrayLike,
    bias_corrected: bool = True,
) -> float:
    """Compute weighted skewness (3rd standardized moment).

    Skewness measures the asymmetry of the distribution. A normal distribution
    has skewness = 0. Positive skewness indicates a right tail, negative
    indicates a left tail.

    Parameters
    ----------
    values : array-like
        Data values.
    weights : array-like
        Non-negative weights for each observation.
    bias_corrected : bool, default True
        If True, apply Fisher's bias correction for finite samples.

    Returns
    -------
    float
        Weighted skewness. Returns NaN if effective sample size < 3.

    Notes
    -----
    Fisher's correction for skewness:

        G1 = g1 * sqrt(n * (n-1)) / (n - 2)

    where g1 is the biased sample skewness and n is the effective sample size.

    References
    ----------
    scipy.stats.skew with bias=False uses Fisher's correction.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    n_eff = effective_sample_size(w)
    if n_eff < 3:
        return np.nan

    mean = np.average(v, weights=w)

    # Use biased variance for standardization (consistent with scipy)
    var_biased = np.average((v - mean) ** 2, weights=w)
    if var_biased <= 0:
        return np.nan

    std = np.sqrt(var_biased)

    # Third standardized moment (biased/uncorrected)
    m3 = np.average(((v - mean) / std) ** 3, weights=w)

    if bias_corrected:
        # Fisher's correction: G1 = g1 * sqrt(n*(n-1)) / (n-2)
        correction = np.sqrt(n_eff * (n_eff - 1)) / (n_eff - 2)
        return float(m3 * correction)

    return float(m3)


def weighted_kurtosis(
    values: ArrayLike,
    weights: ArrayLike,
    excess: bool = False,
    bias_corrected: bool = True,
) -> float:
    """Compute weighted kurtosis (4th standardized moment).

    Kurtosis measures the "tailedness" of the distribution. By default,
    returns raw kurtosis (β₂) where a normal distribution has kurtosis = 3.

    Parameters
    ----------
    values : array-like
        Data values.
    weights : array-like
        Non-negative weights for each observation.
    excess : bool, default False
        If False (default), return raw kurtosis (β₂, normal = 3).
        If True, return excess kurtosis (normal = 0).
        Raw kurtosis is preferred for Pearson distribution system.
    bias_corrected : bool, default True
        If True, apply Fisher's bias correction for finite samples.

    Returns
    -------
    float
        Weighted kurtosis. Returns NaN if effective sample size < 4.

    Notes
    -----
    Fisher's correction for kurtosis:

        G2 = ((n+1) * g2 + 6) * (n-1) / ((n-2) * (n-3))

    where g2 is the biased excess kurtosis and n is the effective sample size.

    The Pearson distribution system uses raw kurtosis (β₂), which is why
    excess=False is the default.

    References
    ----------
    scipy.stats.kurtosis with fisher=True, bias=False uses Fisher's correction
    and returns excess kurtosis.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    n_eff = effective_sample_size(w)
    if n_eff < 4:
        return np.nan

    mean = np.average(v, weights=w)

    # Use biased variance for standardization (consistent with scipy)
    var_biased = np.average((v - mean) ** 2, weights=w)
    if var_biased <= 0:
        return np.nan

    std = np.sqrt(var_biased)

    # Fourth standardized moment (biased/uncorrected)
    m4 = np.average(((v - mean) / std) ** 4, weights=w)

    # m4 is raw kurtosis (β₂); excess kurtosis is m4 - 3
    excess_kurt = m4 - 3.0

    if bias_corrected:
        # Fisher's correction for excess kurtosis:
        # G2 = ((n+1) * g2 + 6) * (n-1) / ((n-2) * (n-3))
        n = n_eff
        g2_corrected = ((n + 1) * excess_kurt + 6) * (n - 1) / ((n - 2) * (n - 3))

        if excess:
            return float(g2_corrected)
        else:
            # Convert back to raw kurtosis
            return float(g2_corrected + 3.0)

    if excess:
        return float(excess_kurt)
    else:
        return float(m4)


def compute_moments(
    values: ArrayLike,
    weights: ArrayLike,
    n_min: int = 30,
) -> dict[str, float]:
    """Compute all 4 statistical moments from weighted observations.

    This is a convenience function that computes mean, variance, skewness,
    and kurtosis in a single call, along with the effective sample size.

    Parameters
    ----------
    values : array-like
        Data values.
    weights : array-like
        Non-negative weights for each observation.
    n_min : int, default 30
        Minimum effective sample size required for moment computation.
        Returns NaN for all moments if n_eff < n_min.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: 'mean', 'variance', 'skewness', 'kurtosis', 'n_eff'

    Notes
    -----
    - Kurtosis is raw β₂ (normal = 3), not excess kurtosis
    - Fisher's bias corrections are applied to variance, skewness, and kurtosis
    - The n_min threshold ensures reliable moment estimates

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = np.random.normal(0, 1, 100)
    >>> weights = np.ones(100)
    >>> moments = compute_moments(data, weights)
    >>> abs(moments['skewness']) < 0.5  # Should be near 0 for normal
    True
    >>> abs(moments['kurtosis'] - 3) < 1  # Should be near 3 for normal
    True
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    n_eff = effective_sample_size(w)

    # Check minimum sample size
    if n_eff < n_min:
        return {
            "mean": np.nan,
            "variance": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "n_eff": n_eff,
        }

    return {
        "mean": weighted_mean(v, w),
        "variance": weighted_variance(v, w, bias_corrected=True),
        "skewness": weighted_skewness(v, w, bias_corrected=True),
        "kurtosis": weighted_kurtosis(v, w, excess=False, bias_corrected=True),
        "n_eff": n_eff,
    }
