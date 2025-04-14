"""Test the parallel processing of GBIF data with trait-specific options."""

import argparse
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import xarray as xr

from src.conf.conf import get_config
from src.data.build_gbif_maps import main


class TestGBIFParallel(unittest.TestCase):
    """Test the build_gbif_maps.py script with trait-specific options."""

    def setUp(self):
        """Set up the test environment."""
        self.config = get_config()
        self.traits_to_test = [
            str(self.config.datasets.Y.traits[0])
        ]  # Test with first trait

        # Create a mock arguments object as a Namespace to match function signature
        class Args(argparse.Namespace):
            def __init__(self):
                super().__init__()
                self.overwrite = False
                self.trait = self.traits_to_test[0]

        self.args = Args()

    @patch("src.data.build_gbif_maps.dd.read_parquet")
    @patch("src.data.build_gbif_maps.rasterize_points")
    @patch("src.data.build_gbif_maps.xr_to_raster")
    @patch("src.data.build_gbif_maps.Client")
    def test_single_trait_processing(
        self, mock_client, mock_xr_to_raster, mock_rasterize, mock_read_parquet
    ):
        """Test that only the specified trait is processed."""
        # Setup mock dataframes and client
        mock_client.return_value.__enter__.return_value = MagicMock()

        # Mock GBIF data
        mock_gbif = MagicMock()
        mock_gbif.pipe.return_value.pipe.return_value.set_index.return_value = mock_gbif

        # Mock trait data with only the specified trait
        mock_traits = MagicMock()
        mock_traits.set_index.return_value = mock_traits

        # Setup mock join result
        mock_merged = MagicMock()
        mock_gbif.join.return_value.reset_index.return_value.drop.return_value = (
            mock_merged
        )

        # Mock dataframe with coordinates
        mock_merged.map_partitions.return_value.drop.return_value = mock_merged

        # Mock the rasterize function
        mock_raster = xr.DataArray(np.random.rand(10, 10))
        mock_rasterize.return_value = mock_raster

        # Mock read_parquet to return our mocks
        mock_read_parquet.side_effect = [mock_gbif, mock_traits]

        # Run the main function with our mocked args
        main(self.args)

        # Check that read_parquet was called with correct columns for the specified trait
        _, trait_call_kwargs = mock_read_parquet.call_args_list[1]
        self.assertIn("columns", trait_call_kwargs)

        expected_columns = ["speciesname", f"X{self.traits_to_test[0]}"]
        self.assertEqual(set(trait_call_kwargs["columns"]), set(expected_columns))

        # Check that rasterize_points was called only once and with the correct column
        mock_rasterize.assert_called_once()
        rasterize_args, _ = mock_rasterize.call_args
        self.assertEqual(rasterize_args[1], f"X{self.traits_to_test[0]}")

        # Check that xr_to_raster was called only once
        mock_xr_to_raster.assert_called_once()


if __name__ == "__main__":
    unittest.main()
