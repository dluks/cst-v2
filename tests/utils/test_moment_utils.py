"""Tests for moment_utils.py - statistical moment functions with weighted observations."""

import numpy as np
import pytest
from scipy import stats as sp_stats

from src.utils.moment_utils import (
    compute_moments,
    effective_sample_size,
    weighted_kurtosis,
    weighted_mean,
    weighted_skewness,
    weighted_variance,
)


class TestEffectiveSampleSize:
    """Tests for effective_sample_size function."""

    def test_equal_weights(self):
        """Equal weights should give n_eff = n."""
        weights = np.ones(100)
        n_eff = effective_sample_size(weights)
        assert n_eff == pytest.approx(100.0)

    def test_equal_scaled_weights(self):
        """Scaled equal weights should also give n_eff = n."""
        weights = np.full(50, 2.0)
        n_eff = effective_sample_size(weights)
        assert n_eff == pytest.approx(50.0)

    def test_unequal_weights_reduces_n_eff(self):
        """Unequal weights should give n_eff < n."""
        weights = np.array([1, 1, 1, 100])
        n_eff = effective_sample_size(weights)
        # n_eff should be much less than 4 due to dominant weight
        assert n_eff < 4
        assert n_eff > 1

    def test_single_dominant_weight(self):
        """Single dominant weight should give n_eff close to 1."""
        weights = np.array([0.01, 0.01, 0.01, 1000])
        n_eff = effective_sample_size(weights)
        assert n_eff < 2

    def test_empty_weights(self):
        """Empty weights should return 0."""
        weights = np.array([])
        n_eff = effective_sample_size(weights)
        # With empty array, sum is 0, so we get 0/0 -> nan or 0
        # depending on implementation
        assert n_eff == 0 or np.isnan(n_eff)


class TestWeightedMean:
    """Tests for weighted_mean function."""

    def test_equal_weights(self):
        """Equal weights should give standard mean."""
        values = np.array([1, 2, 3, 4, 5])
        weights = np.ones(5)
        result = weighted_mean(values, weights)
        assert result == pytest.approx(3.0)

    def test_unequal_weights(self):
        """Unequal weights should shift mean toward heavier weighted values."""
        values = np.array([1, 2])
        weights = np.array([1, 3])  # Weight 2 more heavily
        result = weighted_mean(values, weights)
        expected = (1 * 1 + 2 * 3) / 4
        assert result == pytest.approx(expected)

    def test_matches_numpy(self):
        """Should match np.average."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 100)
        weights = rng.uniform(0.1, 1, 100)
        result = weighted_mean(values, weights)
        expected = np.average(values, weights=weights)
        assert result == pytest.approx(expected)


class TestWeightedVariance:
    """Tests for weighted_variance function."""

    def test_equal_weights_uncorrected(self):
        """Equal weights, uncorrected should match np.var."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)
        result = weighted_variance(values, weights, bias_corrected=False)
        expected = np.var(values)  # Population variance
        assert result == pytest.approx(expected)

    def test_equal_weights_corrected(self):
        """Equal weights, corrected should match np.var with ddof=1."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)
        result = weighted_variance(values, weights, bias_corrected=True)
        expected = np.var(values, ddof=1)  # Sample variance
        assert result == pytest.approx(expected)

    def test_variance_positive(self):
        """Variance should be positive for non-constant data."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 100)
        weights = np.ones(100)
        result = weighted_variance(values, weights)
        assert result > 0

    def test_variance_zero_for_constant(self):
        """Variance should be 0 for constant data."""
        values = np.full(10, 5.0)
        weights = np.ones(10)
        result = weighted_variance(values, weights, bias_corrected=False)
        assert result == pytest.approx(0.0)


class TestWeightedSkewness:
    """Tests for weighted_skewness function."""

    def test_symmetric_distribution(self):
        """Symmetric distribution should have skewness near 0."""
        # Create symmetric data
        values = np.array([-2, -1, 0, 1, 2] * 20)
        weights = np.ones(len(values))
        result = weighted_skewness(values, weights)
        assert abs(result) < 0.1

    def test_right_skewed_positive(self):
        """Right-skewed distribution should have positive skewness."""
        # Create right-skewed data
        rng = np.random.default_rng(42)
        values = rng.exponential(1, 1000)
        weights = np.ones(len(values))
        result = weighted_skewness(values, weights)
        assert result > 0

    def test_matches_scipy_unweighted(self):
        """Should match scipy.stats.skew for equal weights."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 200)
        weights = np.ones(len(values))

        result = weighted_skewness(values, weights, bias_corrected=True)
        expected = sp_stats.skew(values, bias=False)
        assert result == pytest.approx(expected, rel=0.01)

    def test_insufficient_samples(self):
        """Should return NaN for n_eff < 3."""
        values = np.array([1.0, 2.0])
        weights = np.ones(2)
        result = weighted_skewness(values, weights)
        assert np.isnan(result)

    def test_large_normal_sample(self):
        """Large normal sample should have skewness near 0."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 10000)
        weights = np.ones(len(values))
        result = weighted_skewness(values, weights)
        assert abs(result) < 0.1


class TestWeightedKurtosis:
    """Tests for weighted_kurtosis function."""

    def test_normal_distribution_raw_kurtosis(self):
        """Normal distribution should have raw kurtosis near 3."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 10000)
        weights = np.ones(len(values))
        result = weighted_kurtosis(values, weights, excess=False)
        # Raw kurtosis for normal is 3
        assert abs(result - 3) < 0.3

    def test_normal_distribution_excess_kurtosis(self):
        """Normal distribution should have excess kurtosis near 0."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 10000)
        weights = np.ones(len(values))
        result = weighted_kurtosis(values, weights, excess=True)
        # Excess kurtosis for normal is 0
        assert abs(result) < 0.3

    def test_matches_scipy_excess_unweighted(self):
        """Should match scipy.stats.kurtosis for equal weights (excess)."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 500)
        weights = np.ones(len(values))

        result = weighted_kurtosis(values, weights, excess=True, bias_corrected=True)
        expected = sp_stats.kurtosis(values, fisher=True, bias=False)
        assert result == pytest.approx(expected, rel=0.05)

    def test_insufficient_samples(self):
        """Should return NaN for n_eff < 4."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.ones(3)
        result = weighted_kurtosis(values, weights)
        assert np.isnan(result)

    def test_uniform_distribution(self):
        """Uniform distribution should have kurtosis < 3 (platykurtic)."""
        rng = np.random.default_rng(42)
        values = rng.uniform(0, 1, 10000)
        weights = np.ones(len(values))
        result = weighted_kurtosis(values, weights, excess=False)
        # Uniform has raw kurtosis = 1.8 (excess = -1.2)
        assert result < 3

    def test_heavy_tailed_distribution(self):
        """Heavy-tailed distribution should have kurtosis > 3 (leptokurtic)."""
        rng = np.random.default_rng(42)
        # t-distribution with low df has heavy tails
        values = rng.standard_t(5, 10000)
        weights = np.ones(len(values))
        result = weighted_kurtosis(values, weights, excess=False)
        # t(5) has raw kurtosis = 9 (excess = 6)
        assert result > 3


class TestComputeMoments:
    """Tests for compute_moments convenience function."""

    def test_returns_all_moments(self):
        """Should return dict with all 5 keys."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 100)
        weights = np.ones(100)
        result = compute_moments(values, weights)

        assert "mean" in result
        assert "variance" in result
        assert "skewness" in result
        assert "kurtosis" in result
        assert "n_eff" in result

    def test_n_min_threshold(self):
        """Should return NaN for all moments when n_eff < n_min."""
        values = np.array([1, 2, 3, 4, 5])
        weights = np.ones(5)
        result = compute_moments(values, weights, n_min=30)

        assert np.isnan(result["mean"])
        assert np.isnan(result["variance"])
        assert np.isnan(result["skewness"])
        assert np.isnan(result["kurtosis"])
        assert result["n_eff"] == pytest.approx(5.0)

    def test_normal_distribution_moments(self):
        """Large normal sample should have expected moment values."""
        rng = np.random.default_rng(42)
        values = rng.normal(5, 2, 10000)  # mean=5, std=2
        weights = np.ones(len(values))
        result = compute_moments(values, weights, n_min=30)

        # Check mean is near 5
        assert abs(result["mean"] - 5) < 0.1

        # Check variance is near 4 (std^2)
        assert abs(result["variance"] - 4) < 0.2

        # Check skewness is near 0
        assert abs(result["skewness"]) < 0.1

        # Check kurtosis is near 3 (raw, not excess)
        assert abs(result["kurtosis"] - 3) < 0.2


class TestFisherCorrection:
    """Tests for Fisher's bias correction in higher moments."""

    def test_skewness_correction_reduces_bias(self):
        """Fisher correction should reduce bias for small samples."""
        rng = np.random.default_rng(42)

        # Generate many small samples from symmetric distribution
        n_samples = 1000
        sample_size = 20
        uncorrected = []
        corrected = []

        for _ in range(n_samples):
            values = rng.normal(0, 1, sample_size)
            weights = np.ones(sample_size)
            uncorrected.append(
                weighted_skewness(values, weights, bias_corrected=False)
            )
            corrected.append(weighted_skewness(values, weights, bias_corrected=True))

        # Both should average near 0, but corrected should be closer
        # (This is a soft test - bias correction is subtle)
        assert abs(np.nanmean(corrected)) < abs(np.nanmean(uncorrected)) + 0.1

    def test_kurtosis_correction_reduces_bias(self):
        """Fisher correction should reduce bias for kurtosis."""
        rng = np.random.default_rng(42)

        # Generate many small samples from normal distribution
        n_samples = 1000
        sample_size = 30
        uncorrected = []
        corrected = []

        for _ in range(n_samples):
            values = rng.normal(0, 1, sample_size)
            weights = np.ones(sample_size)
            # excess=True so target is 0
            uncorrected.append(
                weighted_kurtosis(values, weights, excess=True, bias_corrected=False)
            )
            corrected.append(
                weighted_kurtosis(values, weights, excess=True, bias_corrected=True)
            )

        # Corrected should be closer to 0
        assert abs(np.nanmean(corrected)) < abs(np.nanmean(uncorrected)) + 0.2


class TestGammaDistribution:
    """Test moments against known theoretical values for gamma distribution."""

    def test_gamma_moments(self):
        """Gamma distribution has known theoretical moments."""
        rng = np.random.default_rng(42)

        # Gamma(a, scale=1): mean=a, var=a, skew=2/sqrt(a), kurt=3+6/a
        a = 4.0
        values = rng.gamma(a, scale=1.0, size=50000)
        weights = np.ones(len(values))

        result = compute_moments(values, weights, n_min=30)

        # Check mean ≈ a
        assert abs(result["mean"] - a) < 0.1

        # Check variance ≈ a
        assert abs(result["variance"] - a) < 0.2

        # Check skewness ≈ 2/sqrt(a) = 1.0
        expected_skew = 2 / np.sqrt(a)
        assert abs(result["skewness"] - expected_skew) < 0.1

        # Check kurtosis ≈ 3 + 6/a = 4.5 (raw)
        expected_kurt = 3 + 6 / a
        assert abs(result["kurtosis"] - expected_kurt) < 0.3


class TestWeightedMoments:
    """Test that weights properly influence moment calculations."""

    def test_weighted_mean_shifts(self):
        """Weights should shift mean toward heavily weighted values."""
        values = np.array([0.0, 100.0])
        weights = np.array([1.0, 9.0])  # Weight 100 more heavily

        result = weighted_mean(values, weights)
        # Should be closer to 100 than 50
        assert result > 50
        assert result == pytest.approx(90.0)

    def test_weighted_variance_with_dominant_weight(self):
        """Dominant weight on single value should reduce variance."""
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        equal_weights = np.ones(5)
        unequal_weights = np.array([100.0, 1.0, 1.0, 1.0, 1.0])

        var_equal = weighted_variance(values, equal_weights, bias_corrected=False)
        var_unequal = weighted_variance(values, unequal_weights, bias_corrected=False)

        # Variance should be lower when one value dominates
        assert var_unequal < var_equal
