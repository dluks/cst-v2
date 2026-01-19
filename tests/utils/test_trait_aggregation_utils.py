import numpy as np
import pandas as pd
import pytest

from src.utils.trait_aggregation_utils import (
    _calculate_fric,
    _cw_quantile,
    _cw_std,
    _cwm,
    calculate_feve,
    calculate_mean_pairwise_dissimilarity,
    calculate_rao_quadratic_entropy,
    calculate_simpson_diversity,
    cw_stats,
    fd_metrics,
)


# Fixtures for test data
@pytest.fixture
def sample_trait_data_single_plot():
    """Create a sample DataFrame with species traits and abundances."""
    np.random.seed(42)
    species = [f"sp_{i}" for i in range(10)]
    trait1 = np.random.normal(size=10)
    trait2 = np.random.normal(size=10)
    abundances = np.random.uniform(size=10)
    norm_abundances = abundances / np.sum(abundances)

    df = pd.DataFrame(
        {
            "species": species,
            "trait1": trait1,
            "trait2": trait2,
            "abundance": norm_abundances,
        }
    )
    return df


@pytest.fixture
def sample_trait_data_multiplot():
    """Create a sample DataFrame representing data from multiple plots."""
    np.random.seed(42)
    plot_ids = np.repeat(["plot1", "plot2", "plot3"], [5, 4, 6])
    species = [f"sp_{np.random.randint(0, 10)}" for _ in range(15)]
    trait1 = np.random.normal(size=15)
    trait2 = np.random.normal(size=15)
    abundances = np.random.uniform(size=15)

    # Normalize abundances within each plot
    norm_abundances = []
    for plot in ["plot1", "plot2", "plot3"]:
        mask = plot_ids == plot
        plot_abundances = abundances[mask]
        norm_abundances.extend(plot_abundances / np.sum(plot_abundances))

    df = pd.DataFrame(
        {
            "plot_id": plot_ids,
            "species": species,
            "trait1": trait1,
            "trait2": trait2,
            "abundance": abundances,
            "Rel_Abund_Plot": norm_abundances,
        }
    )
    return df


#########################################################
# Tests for Community-weighted statistics
#########################################################


def test_cwm():
    """Test community-weighted mean calculation."""
    data = pd.Series([1.0, 2.0, 3.0, 4.0])
    weights = pd.Series([0.4, 0.3, 0.2, 0.1])
    result = _cwm(data, weights)

    # Manual calculation
    expected = 1.0 * 0.4 + 2.0 * 0.3 + 3.0 * 0.2 + 4.0 * 0.1

    assert np.isclose(result, expected)


def test_cw_std():
    """Test community-weighted standard deviation calculation."""
    data = pd.Series([1.0, 2.0, 3.0, 4.0])
    weights = pd.Series([0.4, 0.3, 0.2, 0.1])
    result = _cw_std(data, weights)

    # Manual calculation
    weighted_mean = 1.0 * 0.4 + 2.0 * 0.3 + 3.0 * 0.2 + 4.0 * 0.1
    weighted_variance = (
        (1.0 - weighted_mean) ** 2 * 0.4
        + (2.0 - weighted_mean) ** 2 * 0.3
        + (3.0 - weighted_mean) ** 2 * 0.2
        + (4.0 - weighted_mean) ** 2 * 0.1
    )
    expected = np.sqrt(weighted_variance)

    assert np.isclose(result, expected)


def test_cw_quantile():
    """Test community-weighted quantile calculation."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([0.4, 0.3, 0.2, 0.1])

    # Test median (50th percentile)
    result_median = _cw_quantile(data, weights, 0.5)
    assert result_median == 2.0

    # Test 25th percentile
    result_25 = _cw_quantile(data, weights, 0.25)
    assert result_25 == 1.0

    # Test 75th percentile
    result_75 = _cw_quantile(data, weights, 0.75)
    assert result_75 == 3.0


def test_cw_stats(sample_trait_data_single_plot):
    """Test community-weighted stats function with sample data."""
    # Calculate stats for trait1
    result = cw_stats(sample_trait_data_single_plot, "trait1", "abundance")

    # Check output structure
    assert isinstance(result, pd.Series)
    assert list(result.index) == ["cwm", "cw_std", "cw_med", "cw_q02", "cw_q05", "cw_q25", "cw_q75", "cw_q95", "cw_q98"]

    # Check types of output values
    for val in result:
        assert isinstance(val, float) or np.isnan(val)


def test_cw_stats_empty_data():
    """Test cw_stats with empty data."""
    empty_df = pd.DataFrame(columns=["trait1", "abundance"])
    result = cw_stats(empty_df, "trait1", "abundance")

    # Check that all values are NaN
    assert all(np.isnan(val) for val in result)


#########################################################
# Tests for Functional diversity metrics
#########################################################


def test_calculate_simpson_diversity():
    """Test Simpson diversity calculation."""
    # Equal abundances
    equal_abundances = np.array([0.25, 0.25, 0.25, 0.25])
    expected_equal = 1 - np.sum(equal_abundances**2)  # 0.75
    assert np.isclose(calculate_simpson_diversity(equal_abundances), expected_equal)

    # Unequal abundances
    unequal_abundances = np.array([0.6, 0.2, 0.1, 0.1])
    expected_unequal = 1 - np.sum(unequal_abundances**2)  # 0.58
    assert np.isclose(calculate_simpson_diversity(unequal_abundances), expected_unequal)

    # Single species (should be 0)
    single_species = np.array([1.0])
    expected_single = 1 - np.sum(single_species**2)  # 0
    assert np.isclose(calculate_simpson_diversity(single_species), expected_single)


def test_calculate_rao_quadratic_entropy():
    """Test Rao's quadratic entropy calculation."""
    trait_matrix = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

    # Equal abundances
    equal_abundances = np.array([0.25, 0.25, 0.25, 0.25])
    result_equal = calculate_rao_quadratic_entropy(trait_matrix, equal_abundances)

    # Manual calculation for equal abundances
    # The distance matrix is:
    # [0, 1, 1, sqrt(2)]
    # [1, 0, sqrt(2), 1]
    # [1, sqrt(2), 0, 1]
    # [sqrt(2), 1, 1, 0]
    # With equal weights, this averages to (4*0 + 12*1 + 4*sqrt(2))/2 = (12 + 4*sqrt(2))/2
    expected_equal = 0.42677669529663687
    assert np.isclose(result_equal, expected_equal)

    # Unequal abundances
    unequal_abundances = np.array([0.4, 0.3, 0.2, 0.1])
    result_unequal = calculate_rao_quadratic_entropy(trait_matrix, unequal_abundances)

    # Manual calculation for unequal abundances is more complex - we'll just check it's a float
    assert isinstance(result_unequal, float)


def test_calculate_mean_pairwise_dissimilarity():
    """Test mean pairwise dissimilarity calculation."""
    trait_matrix = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

    # Equal abundances
    equal_abundances = np.array([0.25, 0.25, 0.25, 0.25])
    result_equal = calculate_mean_pairwise_dissimilarity(trait_matrix, equal_abundances)

    # Since MPD = Rao / Simpson
    rao_equal = calculate_rao_quadratic_entropy(trait_matrix, equal_abundances)
    simpson_equal = calculate_simpson_diversity(equal_abundances)
    expected_equal = rao_equal / simpson_equal

    assert np.isclose(result_equal, expected_equal)

    # Should handle zero Simpson index
    single_species = np.array([1.0])
    single_trait = np.array([[1.0]])
    # This should return NaN
    result_single = calculate_mean_pairwise_dissimilarity(single_trait, single_species)
    assert np.isnan(result_single)


def test_calculate_feve():
    """Test functional evenness calculation."""
    # Simple case: points in a square
    even_trait_matrix = np.array(
        [
            [0, 0, 0],  # Species 1
            [0.5, 0.5, 0.5],  # Species 2
            [1, 1, 1],  # Species 3
        ]
    )

    uneven_trait_matrix = np.array(
        [
            [0, 0, 0],  # Species 1
            [0.1, 0.1, 0.1],  # Species 2 (very close to Species 1)
            [10, 10, 10],  # Species 3 (very far from Species 1 and 2)
        ]
    )

    # Equal abundances
    equal_abundances = np.array([0.25, 0.25, 0.25, 0.25])
    even_result = calculate_feve(even_trait_matrix, equal_abundances)
    uneven_result = calculate_feve(uneven_trait_matrix, equal_abundances)

    # Test that both results are between 0 and 1
    assert 0 <= even_result <= 1
    assert 0 <= uneven_result <= 1

    # Test that even result is close to 1 and uneven result is close(ish) to 0
    assert np.isclose(even_result, 1, atol=0.01)
    assert np.isclose(uneven_result, 0, atol=0.05)

    # Extreme case with only two species
    two_species_traits = np.array([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
    two_species_abundances = np.array([0.5, 0.5])
    result_edge = calculate_feve(two_species_traits, two_species_abundances)
    assert np.isnan(result_edge)


def test_calculate_fric():
    """Test functional richness calculation."""
    trait_matrix = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
    result = _calculate_fric(trait_matrix)
    expected = 1.0
    assert np.isclose(result, expected)

    # Not enough points for convex hull
    small_trait_matrix = np.array([[0, 0], [1, 1]])
    result_small = _calculate_fric(small_trait_matrix)
    assert np.isnan(result_small)

    # Not enough observations for convex hull
    more_traits_than_obs = np.array([[0, 0, 0], [1, 1, 1]])
    result_more_traits_than_obs = _calculate_fric(more_traits_than_obs)
    assert np.isnan(result_more_traits_than_obs)


def test_fd_metrics(sample_trait_data_single_plot):
    """Test fd_metrics function with different stats combinations."""
    trait_cols = ["trait1", "trait2"]

    # Test with all metrics
    all_stats = ["sp_ric", "f_ric", "f_eve", "f_div", "f_red"]
    result_all = fd_metrics(
        sample_trait_data_single_plot,
        trait_cols=trait_cols,
        stats=all_stats,
        species_col="species",
        abundance_col="abundance",
    )

    # Check output structure
    assert isinstance(result_all, pd.Series)
    assert set(result_all.index) == set(all_stats)

    # Test with specific metrics
    specific_stats = ["sp_ric", "f_div"]
    result_specific = fd_metrics(
        sample_trait_data_single_plot,
        trait_cols=trait_cols,
        stats=specific_stats,
        species_col="species",
        abundance_col="abundance",
    )
    assert set(result_specific.index) == set(specific_stats)

    # Test with empty dataframe
    empty_df = pd.DataFrame(columns=["species", "trait1", "trait2", "abundance"])
    result_empty = fd_metrics(
        empty_df,
        trait_cols=trait_cols,
        stats=all_stats,
        species_col="species",
        abundance_col="abundance",
    )
    assert all(np.isnan(val) for val in result_empty)

    # Test without abundance column
    result_no_abund = fd_metrics(
        sample_trait_data_single_plot,
        trait_cols=trait_cols,
        stats=all_stats,
        species_col="species",
        abundance_col=None,
    )
    assert isinstance(result_no_abund, pd.Series)
    assert set(result_no_abund.index) == set(all_stats)


#########################################################
# Integration tests
#########################################################


# TODO: Use actual sample data instead of overly simple dataframe for testing
def test_trait_aggregation_workflow(sample_trait_data_multiplot):
    """Test an end-to-end workflow with the trait aggregation utilities."""
    trait_cols = ["trait1", "trait2"]

    # Group by plot and apply fd_metrics to each group
    results = []
    for plot_id, group in sample_trait_data_multiplot.groupby("plot_id"):
        metrics = fd_metrics(
            group,
            trait_cols=trait_cols,
            stats=["sp_ric", "f_ric", "f_eve", "f_div", "f_red"],
            species_col="species",
            abundance_col="Rel_Abund_Plot",
        )
        results.append(pd.Series({**{"plot_id": plot_id}, **metrics}))

    # Combine results into DataFrame
    results_df = pd.DataFrame(results).set_index("plot_id")

    # Check structure
    assert results_df.shape == (3, 5)  # 3 plots, 5 metrics
    assert set(results_df.columns) == set(
        ["sp_ric", "f_ric", "f_eve", "f_div", "f_red"]
    )

    # Check data types
    assert results_df["sp_ric"].dtype == np.float64
    assert results_df["f_ric"].dtype == np.float64
