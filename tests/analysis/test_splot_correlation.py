"""Tests for splot_correlation module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.splot_correlation import main, process_pair


@pytest.fixture
def test_data_dir():
    """Get the test data directory containing the pre-copied trait map."""
    return Path("tests/test_data/interim/splot/trait_maps/Shrub_Tree_Grass/222km")


@pytest.fixture
def mock_config():
    """Create a mock config object."""
    config = MagicMock()
    config.model_res = "222km"
    config.datasets.Y.correlation_fn = "test_correlation.csv"
    return config


def test_process_pair(test_data_dir):
    """Test the process_pair function with real trait map data."""
    # Use the same file for both splot and gbif to test correlation
    splot_fn = test_data_dir / "X50.tif"
    gbif_fn = test_data_dir / "X50.tif"

    # Test the function
    result = process_pair(splot_fn, gbif_fn, "222km")

    # Verify the result
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] == "X50"  # trait_id
    assert result[1] == "222km"  # resolution
    assert np.isclose(result[2], 1.0)  # correlation (should be 1.0 for identical data)


def test_process_pair_mismatched_names(test_data_dir):
    """Test process_pair with mismatched file names."""
    splot_fn = test_data_dir / "X50.tif"
    gbif_fn = test_data_dir / "X51.tif"

    with pytest.raises(AssertionError):
        process_pair(splot_fn, gbif_fn, "222km")


@patch("src.analysis.splot_correlation.get_config")
@patch("src.analysis.splot_correlation.get_trait_map_fns")
def test_main_workflow(
    mock_get_trait_map_fns,
    mock_get_config,
    mock_config,
    test_data_dir,
    tmp_path,
):
    """Test the main workflow with real trait map data."""
    # Setup mocks
    mock_get_config.return_value = mock_config

    # Create mock file paths using the real trait map
    splot_fns = [test_data_dir / "X50.tif"]
    gbif_fns = [test_data_dir / "X50.tif"]
    mock_get_trait_map_fns.side_effect = [splot_fns, gbif_fns]

    # Create results directory
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Run main function
    with patch("src.analysis.splot_correlation.Path") as mock_path:
        mock_path.return_value = results_dir / "test_correlation.csv"
        main()

    # Verify results
    expected_df = pd.DataFrame(
        {
            "trait_id": ["X50"],
            "resolution": ["222km"],
            "pearsonr": [1.0],
        }
    )

    # Check if the CSV file was created with correct content
    result_file = results_dir / "test_correlation.csv"
    assert result_file.exists()
    result_df = pd.read_csv(result_file)
    pd.testing.assert_frame_equal(result_df, expected_df)
