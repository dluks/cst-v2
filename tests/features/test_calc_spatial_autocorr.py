"""Tests for spatial autocorrelation functions."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from box import ConfigBox

from src.features.calc_spatial_autocorr_gpu import (
    _assign_zones,
    _calculate_haversine_distance,
    _compute_experimental_variogram,
    _estimate_max_distance,
    _fit_variogram_model,
    _single_trait_ranges,
    _transform_coords_to_wgs84,
    add_utm,
    calculate_variogram_gpu,
    fit_variogram_gpu,
    get_next_gpu,
    set_gpu_devices,
    spherical_model,
)


def test_set_and_get_gpu_devices():
    """Test setting GPU devices and getting the next GPU in round-robin fashion."""
    set_gpu_devices([0, 1, 2])
    assert get_next_gpu() == 0
    assert get_next_gpu() == 1
    assert get_next_gpu() == 2
    assert get_next_gpu() == 0  # Should cycle back to the first GPU


def test_spherical_model():
    """Test the spherical semivariogram model calculation."""
    # Create a distance tensor
    h = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    nugget = 0.1
    sill = 0.9
    range_param = 1.5

    result = spherical_model(h, nugget, sill, range_param)

    # For h <= range_param: nugget + sill * (1.5 * h/range - 0.5 * (h/range)^3)
    # For h > range_param: nugget + sill
    expected = torch.tensor(
        [
            0.1,  # h=0: nugget
            0.1 + 0.9 * (1.5 * 0.5 / 1.5 - 0.5 * (0.5 / 1.5) ** 3),  # h=0.5
            0.1 + 0.9 * (1.5 * 1.0 / 1.5 - 0.5 * (1.0 / 1.5) ** 3),  # h=1.0
            0.1 + 0.9 * (1.5 * 1.5 / 1.5 - 0.5 * (1.5 / 1.5) ** 3),  # h=1.5
            1.0,  # h=2.0: nugget + sill
        ]
    )

    assert torch.allclose(result, expected, rtol=1e-5)


def test_transform_coords_to_wgs84():
    """Test coordinate transformation to WGS84."""
    device = torch.device("cpu")

    # Test with coordinates already in WGS84
    coords = torch.tensor(
        [[10.0, 50.0], [11.0, 51.0]], dtype=torch.float32, device=device
    )
    result = _transform_coords_to_wgs84(coords, "EPSG:4326", device)
    assert torch.allclose(result, coords)

    # Test with coordinates in a different CRS (EPSG:3857 - Web Mercator)
    # Create some sample points in EPSG:3857
    coords_3857 = torch.tensor(
        [
            [1113194.9079327357, 6446275.841017158],  # Approx. 10°E, 50°N
            [1224639.7111912148, 6621293.7227225975],  # Approx. 11°E, 51°N
        ],
        dtype=torch.float32,
        device=device,
    )

    result = _transform_coords_to_wgs84(coords_3857, "EPSG:3857", device)

    # Expected values (approximate)
    expected = torch.tensor(
        [[10.0, 50.0], [11.0, 51.0]], dtype=torch.float32, device=device
    )

    # Use a larger tolerance for the transformation comparison
    assert torch.allclose(result, expected, rtol=1e-2)


def test_calculate_haversine_distance():
    """Test haversine distance calculation."""
    # Test with known coordinates and distances
    # New York: 40.7128° N, 74.0060° W
    # London: 51.5074° N, 0.1278° W
    lons1 = torch.tensor([[-74.0060]], dtype=torch.float32)
    lats1 = torch.tensor([[40.7128]], dtype=torch.float32)
    lons2 = torch.tensor([[-0.1278]], dtype=torch.float32)
    lats2 = torch.tensor([[51.5074]], dtype=torch.float32)

    # Calculate distance
    distance = _calculate_haversine_distance(lons1, lats1, lons2, lats2)

    # Expected distance between New York and London is approximately 5570 km
    expected_distance = torch.tensor([[5570000.0]], dtype=torch.float32)  # in meters

    assert torch.isclose(distance, expected_distance, rtol=0.01)


def test_estimate_max_distance_smoke_test():
    """Smoke test for maximum distance estimation function."""
    # Create a simple grid of coordinates (longitude/latitude format)
    lons = np.linspace(-5, 5, 10)
    lats = np.linspace(45, 55, 10)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    coords = np.column_stack((lon_grid.flatten(), lat_grid.flatten()))

    # Convert to tensor
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    n_samples = coords_tensor.shape[0]

    # Estimate max distance
    max_dist = _estimate_max_distance(coords_tensor, n_samples)

    # Basic sanity checks
    assert isinstance(max_dist, float)
    assert max_dist > 0

    # Check that the distance is reasonable for Earth-scale coordinates
    # Should be between 100km and 10000km for this coordinate range
    assert 100_000 < max_dist < 10_000_000, (
        f"Distance {max_dist}m is outside reasonable range"
    )

    # Ensure function handles small datasets
    small_coords = coords_tensor[:5]
    small_dist = _estimate_max_distance(small_coords, 5)
    assert small_dist > 0


# @pytest.mark.parametrize(
#     "crs_from,crs_to",
#     [
#         ("EPSG:4326", "EPSG:6933"),
#         ("EPSG:4326", "EPSG:3857"),
#     ],
# )
# def test_transform_coords_to_equidistant(crs_from, crs_to):
#     """Test transformation of coordinates to equidistant projection."""
#     # Create sample coordinates in WGS84
#     coords = np.array([[10.0, 50.0], [11.0, 51.0]])

#     # Transform coordinates
#     result = transform_coords_to_equidistant(coords, crs_from, crs_to)

#     # Verify the result has the right shape
#     assert result.shape == (2, 2)

#     # Verify the transformation by transforming back
#     transformer = Transformer.from_crs(crs_to, crs_from, always_xy=True)
#     lon, lat = transformer.transform(result[0], result[1])
#     back_transformed = np.column_stack((lon, lat))

#     # Check if the back-transformed coordinates are close to the original
#     assert np.allclose(back_transformed, coords, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_compute_experimental_variogram():
    """Test computation of experimental variogram."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create a simple grid of coordinates
    n = 20
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack((xx.flatten(), yy.flatten()))

    # Create values with spatial correlation (distance-based)
    values = np.sin(xx.flatten() / 3) + np.cos(yy.flatten() / 2)

    # Convert to tensors
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)

    # Set up bins
    nlags = 10
    max_dist = 15.0
    bin_edges = torch.linspace(0, max_dist, nlags + 1, device=device)

    # Compute experimental variogram
    gamma_sum, gamma_counts = _compute_experimental_variogram(
        coords_tensor, values_tensor, bin_edges, nlags, chunk_size=100, device=device
    )

    # Check shapes
    assert gamma_sum.shape == (nlags,)
    assert gamma_counts.shape == (nlags,)

    # Check that counts are positive for at least some bins
    assert torch.any(gamma_counts > 0)

    # Check that gamma values are non-negative where counts are positive
    valid_lags = gamma_counts > 0
    assert torch.all(gamma_sum[valid_lags] >= 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fit_variogram_model():
    """Test fitting of variogram model."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create synthetic experimental variogram data
    nlags = 15
    bin_centers = torch.linspace(1, 15, nlags, device=device)

    # Generate synthetic gamma values following a spherical model
    nugget_true = 0.2
    sill_true = 0.8
    range_true = 10.0
    gamma = spherical_model(bin_centers, nugget_true, sill_true, range_true)

    # Add some noise
    gamma += torch.randn_like(gamma) * 0.05

    # All lags are valid
    valid_lags = torch.ones(nlags, dtype=torch.bool, device=device)

    # Fit the model
    range_est, nugget_est, sill_est, mse = _fit_variogram_model(
        bin_centers, gamma, valid_lags, device
    )

    # Check that estimated parameters are close to true values
    assert np.isclose(range_est, range_true, rtol=0.2)
    assert np.isclose(nugget_est, nugget_true, rtol=0.2)
    assert np.isclose(sill_est, sill_true, rtol=0.2)
    assert mse < 0.01  # MSE should be small


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fit_variogram_gpu():
    """Test the full variogram fitting pipeline."""
    # Create synthetic data with spatial correlation
    n = 30
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack((xx.flatten(), yy.flatten()))

    # Generate values with a known spatial pattern
    values = np.sin(xx.flatten() / 3) + np.cos(yy.flatten() / 2)
    values += np.random.normal(0, 0.1, size=values.shape)  # Add some noise

    # Fit variogram
    range_param, nugget, sill, mse = fit_variogram_gpu(
        coords=coords,
        values=values,
        nlags=15,
        gpu_id=0,
        crs="EPSG:4326",  # Use WGS84 for simplicity
    )

    # Check that parameters are reasonable
    assert 0 < range_param < 2_000_000, (
        f"Range parameter {range_param}m outside expected bounds"
    )
    assert 0 <= nugget < 1.0, f"Nugget {nugget} outside expected bounds"
    assert 0 < sill < 2.0, f"Sill {sill} outside expected bounds"
    assert 0 <= mse < 0.1, f"MSE {mse} too high"

    # Check relationship between parameters
    assert nugget < sill, "Nugget should be less than sill"
    assert range_param > 1000, "Range should be meaningful (>1000m) for geographic data"


def test_add_utm():
    """Test the add_utm function for converting coordinates to UTM."""
    # Create a small DataFrame with WGS84 coordinates
    df = pd.DataFrame(
        {
            "x": [10.0, 11.0, 12.0],  # Longitudes
            "y": [50.0, 51.0, 52.0],  # Latitudes
        }
    )

    # Apply the add_utm function
    result_df = add_utm(df, chunksize=2)

    # Check that the result has the expected columns
    assert "zone" in result_df.columns

    # Check that all rows have values
    assert not result_df["zone"].isna().any()

    # Check that the zones are in the expected format (number + letter)
    for zone in result_df["zone"]:
        assert len(zone) >= 2
        assert zone[:-1].isdigit()
        assert zone[-1].isalpha()

    # Verify a specific coordinate conversion
    # For longitude 10.0, latitude 50.0, the UTM zone should be 32U
    first_row = result_df.iloc[0]
    assert first_row["zone"].startswith("32")


def test_assign_zones():
    """Test the _assign_zones function for dividing data into spatial zones."""
    # Create a test DataFrame with x and y coordinates
    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [0.0, 1.0, 0.5, 2.0, 1.5, 2.5],
        }
    )

    # Test with 4 zones (2x2 grid)
    result_df = _assign_zones(df, n_zones=4)

    # Check that the result has the expected columns
    assert "zone" in result_df.columns
    assert "easting" not in result_df.columns  # Should be dropped
    assert "northing" not in result_df.columns  # Should be dropped

    # Check that all rows have zone values
    assert not result_df["zone"].isna().any()

    # Check that we have the expected number of unique zones
    # With n_zones=4, we should have a 4x2 grid, so up to 8 unique zones
    unique_zones = result_df["zone"].unique()
    assert len(unique_zones) <= 8

    # Check zone format (should be x_y format)
    for zone in unique_zones:
        x, y = zone.split("_")
        assert x.isdigit()
        assert y.isdigit()

    # Test with different number of zones
    result_df_large = _assign_zones(df, n_zones=8)
    unique_zones_large = result_df_large["zone"].unique()
    assert len(unique_zones_large) <= 16  # 8x4 grid


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_calculate_variogram_gpu():
    """Test the calculate_variogram_gpu function."""
    # Create test data
    n = 300
    x = np.linspace(-10, 10, n)
    y = np.linspace(40, 60, n)
    xx, yy = np.meshgrid(x[:20], y[:20])

    # Create a DataFrame with spatial pattern
    data = pd.DataFrame(
        {
            "x": xx.flatten(),
            "y": yy.flatten(),
            "value": np.sin(xx.flatten() / 3) + np.cos(yy.flatten() / 2),
        }
    )

    # Run the function
    result = calculate_variogram_gpu(
        data, "value", nlags=10, n_max=300, crs="EPSG:4326"
    )

    # Compute the result (since it's a delayed function)
    range_param, n_samples = result.compute()

    # Verify results
    assert isinstance(range_param, float)
    assert isinstance(n_samples, int)
    assert range_param > 0
    assert n_samples > 0
    assert n_samples <= 300  # Should respect n_max


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fit_variogram_gpu_with_log_binning():
    """Test variogram fitting with logarithmic binning."""
    # Create synthetic data with spatial correlation
    n = 30
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack((xx.flatten(), yy.flatten()))

    # Generate values with a known spatial pattern
    values = np.sin(xx.flatten() / 3) + np.cos(yy.flatten() / 2)
    values += np.random.normal(0, 0.1, size=values.shape)  # Add some noise

    # Fit variogram with log binning
    range_param, nugget, sill, mse = fit_variogram_gpu(
        coords=coords,
        values=values,
        nlags=15,
        gpu_id=0,
        crs="EPSG:4326",
        log_binning=True,
    )

    # Check that parameters are reasonable
    assert 0 < range_param < 2_000_000
    assert 0 <= nugget < 1.0
    assert 0 < sill < 2.0
    assert 0 <= mse < 0.1

    # Compare with linear binning to ensure different results
    range_linear, _, _, _ = fit_variogram_gpu(
        coords=coords,
        values=values,
        nlags=15,
        gpu_id=0,
        crs="EPSG:4326",
        log_binning=False,
    )

    # The results should be different but within same order of magnitude
    assert range_param != range_linear
    assert 0.1 * range_linear < range_param < 10 * range_linear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@patch("dask.dataframe.read_parquet")
@patch("src.features.calc_spatial_autocorr_gpu.init_dask")
@patch("src.features.calc_spatial_autocorr_gpu.close_dask")
@patch("src.features.calc_spatial_autocorr_gpu.get_autocorr_ranges_fn")
@patch("src.features.calc_spatial_autocorr_gpu.get_active_traits")
@patch("src.features.calc_spatial_autocorr_gpu.get_y_fn")
def test_single_trait_ranges(
    mock_get_y_fn,
    mock_get_traits,
    mock_get_ranges_fn,
    mock_close,
    mock_init,
    mock_read_parquet,
    tmp_path,
):
    """Test the _single_trait_ranges function."""
    # Setup mocks
    mock_init.return_value = (MagicMock(), MagicMock())
    mock_get_traits.return_value = ["trait1"]
    mock_get_y_fn.return_value = "test.parquet"
    mock_get_ranges_fn.return_value = tmp_path / "ranges.parquet"

    # Create test data
    n = 10_000
    x = np.linspace(-10, 10, n)
    y = np.linspace(40, 60, n)
    xx, yy = np.meshgrid(x[:100], y[:100])

    df = pd.DataFrame(
        {
            "x": xx.flatten(),
            "y": yy.flatten(),
            "trait1": np.sin(xx.flatten() / 3) + np.cos(yy.flatten() / 2),
        }
    )

    # Create config
    cfg = ConfigBox(
        {"crs": "EPSG:4326", "calc_spatial_autocorr": {"use_existing": False}}
    )

    syscfg = ConfigBox({"n_chunks": 2})

    vgram_kwargs = {"n_max": 50, "nlags": 10}

    # Run the function
    result = _single_trait_ranges(df, "trait1", cfg, syscfg, vgram_kwargs)

    # Compute the result (since it's a delayed function)
    result_df = result.compute()

    # Verify results
    assert isinstance(result_df, pd.DataFrame)
    assert "trait" in result_df.columns
    assert "mean" in result_df.columns
    assert "std" in result_df.columns
    assert "median" in result_df.columns
    assert "stability" in result_df.columns
    assert result_df["trait"].iloc[0] == "trait1"
    assert result_df["mean"].iloc[0] > 0
