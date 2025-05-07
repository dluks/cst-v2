"""Tests for the build_try_traits.py module."""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from box import ConfigBox
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.data.build_try_traits import (
    _apply_pca,
    _filter_if_specified,
    _log_transform_long_tails,
    _power_transform,
    _standard_normalize,
    _transform,
    main,
    standardize_trait_ids,
)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return ConfigBox(
        {
            "try_version": 5,
            "raw_dir": "data/raw",
            "interim_dir": "data/interim",
            "PFT": "Shrub",
            "trydb": {
                "raw": {
                    "try5": {
                        "dir": "TRY_5_GapFilledData_2020",
                        "zip": "TRY_5_GapFilledData_2020.zip",
                        "zipfile_csv": "TRY_50_2020_01/gapfilled_data/mean_gap_filled_back_transformed_incl_species_names.csv",
                    },
                    "pfts": "try_pft_v2.parquet",
                },
                "interim": {
                    "dir": "try",
                    "transform": "norm",
                    "quantile_range": (0.005, 0.995),
                    "filtered": "traits.parquet",
                    "transformer_fn": "norm_scaler.pkl",
                    "perform_pca": True,
                    "pca_fn": "trait_pca.pkl",
                    "pca_n_components": 0.95,
                },
            },
            "datasets": {"Y": {"traits": [1, 4, 39]}},
        }
    )


@pytest.fixture
def sample_trait_data():
    """Create a small sample of trait data for testing."""
    return pd.DataFrame(
        {
            "Species": [
                "Acacia longifolia",
                "Acer rubrum",
                "Betula pendula",
                "Quercus rubra",
                "Pinus sylvestris",
            ],
            "X1": [0.1, 0.2, 0.3, 0.4, 0.5],  # SLA - specific leaf area
            "X4": [1.0, 2.0, 3.0, 4.0, 5.0],  # Leaf N content
            "X39": [10, 20, 30, 40, 50],  # Plant height
        }
    )


@pytest.fixture
def sample_pft_data():
    """Create sample PFT data for testing."""
    return pd.DataFrame(
        {
            "AccSpeciesName": [
                "Acacia longifolia",
                "Acer rubrum",
                "Betula pendula",
                "Quercus rubra",
                "Pinus sylvestris",
            ],
            "pft": ["Shrub", "Tree", "Tree", "Tree", "Shrub"],
        }
    )


def test_standardize_trait_ids_try5():
    """Test standardizing trait IDs from TRY5 format."""
    # TRY5 format just has Xs
    df = pd.DataFrame(
        {
            "Species": ["Acacia longifolia"],
            "X1": [0.1],
            "X4": [1.0],
        }
    )

    result = standardize_trait_ids(df)
    # Should remain unchanged for TRY5
    assert list(result.columns) == ["Species", "X1", "X4"]


def test_standardize_trait_ids_try6():
    """Test standardizing trait IDs from TRY6 format."""
    # TRY6 format has X.X format
    df = pd.DataFrame(
        {
            "Species": ["Acacia longifolia"],
            "X.X1.": [0.1],
            "X.X4.": [1.0],
        }
    )

    result = standardize_trait_ids(df)
    # Should convert to X1, X4
    assert list(result.columns) == ["Species", "X1", "X4"]


def test_filter_if_specified_with_range(sample_trait_data):
    """Test filtering outliers with a quantile range."""
    # Add an outlier
    sample_with_outlier = sample_trait_data.copy()
    sample_with_outlier.loc[5] = ["Outlier Species", 100.0, 200.0, 1000.0]

    # Filter with standard quantile range
    filtered = _filter_if_specified(
        sample_with_outlier, trait_cols=["X1", "X4", "X39"], quantile_range=(0.1, 0.9)
    )

    # Should remove the outlier
    assert len(filtered) < len(sample_with_outlier)
    assert "Outlier Species" not in filtered["Species"].values


def test_filter_if_specified_with_string_column(sample_trait_data):
    """Test filtering handles string columns correctly."""
    result = _filter_if_specified(
        sample_trait_data,
        trait_cols=["Species", "X1", "X4", "X39"],  # Includes string column
        quantile_range=(0.005, 0.995),
    )

    # Should filter only numeric columns, ignoring "Species"
    assert "Species" in result.columns


def test_filter_if_specified_no_filtering(sample_trait_data):
    """Test when no filtering is requested."""
    result = _filter_if_specified(
        sample_trait_data, trait_cols=["X1", "X4", "X39"], quantile_range=None
    )

    # Should return unchanged DataFrame
    pd.testing.assert_frame_equal(result, sample_trait_data)


def test_log_transform_long_tails(sample_trait_data):
    """Test log transformation of traits with long-tailed distributions."""
    # Drop Species column for transformation
    numeric_data = sample_trait_data.drop(columns=["Species"]).copy()

    # Transform all columns
    result = _log_transform_long_tails(numeric_data, keep=[])

    # Should add _ln suffix and apply log1p transformation
    assert "X1_ln" in result.columns
    assert "X4_ln" in result.columns
    assert "X39_ln" in result.columns

    # Original columns should be dropped
    assert "X1" not in result.columns
    assert "X4" not in result.columns
    assert "X39" not in result.columns

    # Check transformation is correct (log1p)
    assert result["X1_ln"].iloc[0] == np.log1p(numeric_data["X1"].iloc[0])


def test_log_transform_with_keeps(sample_trait_data):
    """Test log transformation with some traits kept unchanged."""
    # Drop Species column for transformation
    numeric_data = sample_trait_data.drop(columns=["Species"]).copy()

    # Keep trait 4 untransformed
    result = _log_transform_long_tails(numeric_data, keep=["4"])

    # X4 should remain unchanged
    assert "X4" in result.columns
    assert "X4_ln" not in result.columns

    # X1 and X39 should be transformed
    assert "X1_ln" in result.columns
    assert "X39_ln" in result.columns


def test_power_transform():
    """Test power transformation (Yeo-Johnson)."""
    # Create sample data
    df = pd.DataFrame(
        {
            "X1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "X4": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    # Use a temporary file for the transformer
    with tempfile.NamedTemporaryFile(suffix=".pkl") as temp:
        transformer_path = Path(temp.name)
        result = _power_transform(df, transformer_path)

        # Verify the file was created
        assert transformer_path.exists()

        # Check transformer type
        with open(transformer_path, "rb") as f:
            transformer = pickle.load(f)
            assert isinstance(transformer, PowerTransformer)

        # Check output has same shape and columns
        assert result.shape == df.shape
        assert list(result.columns) == list(df.columns)


def test_standard_normalize():
    """Test standard normalization."""
    # Create sample data
    df = pd.DataFrame(
        {
            "X1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "X4": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    # Use a temporary file for the scaler
    with tempfile.NamedTemporaryFile(suffix=".pkl") as temp:
        transformer_path = Path(temp.name)
        result = _standard_normalize(df, transformer_path)

        # Verify the file was created
        assert transformer_path.exists()

        # Check scaler type
        with open(transformer_path, "rb") as f:
            scaler = pickle.load(f)
            assert isinstance(scaler, StandardScaler)

        # Check output has same shape and columns
        assert result.shape == df.shape
        assert list(result.columns) == list(df.columns)

        # Verify values are scaled with mean ~= 0 and std ~= 1
        assert np.isclose(result.mean().mean(), 0.0, atol=0.01)
        print("mean", result.mean())
        print("std", result.std())
        assert np.isclose(result.std(ddof=0).mean(), 1.0, atol=0.01)


def test_apply_pca():
    """Test PCA application to trait data."""
    # Create sample data
    df = pd.DataFrame(
        {
            "X1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "X4": [1.0, 2.0, 3.0, 4.0, 5.0],
            "X39": [10, 20, 30, 40, 50],
        }
    )

    # Use a temporary file for the PCA
    with tempfile.NamedTemporaryFile(suffix=".pkl") as temp:
        pca_path = Path(temp.name)
        result = _apply_pca(df, pca_path, n_components=0.95)

        # Verify the file was created
        assert pca_path.exists()

        # Check PCA type
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
            assert isinstance(pca, PCA)

        # Check output has reduced dimensions
        assert result.shape[0] == df.shape[0]
        assert result.shape[1] <= df.shape[1]

        # Check column names are PC1, PC2, etc.
        assert all(col.startswith("PC") for col in result.columns)


def test_transform_with_none():
    """Test transform function with no transformation requested."""
    df = pd.DataFrame({"X1": [0.1, 0.2, 0.3]})

    with tempfile.NamedTemporaryFile(suffix=".pkl") as temp:
        transformer_path = Path(temp.name)
        result = _transform(df, transform=None, transformer_fn=transformer_path)

        # Should return unchanged DataFrame
        pd.testing.assert_frame_equal(result, df)

        # Should create a placeholder transformer file
        assert transformer_path.exists()
        with open(transformer_path, "rb") as f:
            transformer = pickle.load(f)
            assert transformer is None


def test_transform_log():
    """Test transform function with log transformation."""
    df = pd.DataFrame({"X1": [0.1, 0.2, 0.3]})

    with tempfile.NamedTemporaryFile(suffix=".pkl") as temp:
        transformer_path = Path(temp.name)

        with patch("src.data.build_try_traits._log_transform_long_tails") as mock_log:
            mock_log.return_value = df  # Return unchanged for simplicity
            _ = _transform(df, transform="log", transformer_fn=transformer_path)

            # Should call _log_transform_long_tails
            mock_log.assert_called_once()


def test_transform_power():
    """Test transform function with power transformation."""
    df = pd.DataFrame({"X1": [0.1, 0.2, 0.3]})

    with tempfile.NamedTemporaryFile(suffix=".pkl") as temp:
        transformer_path = Path(temp.name)

        with patch("src.data.build_try_traits._power_transform") as mock_power:
            mock_power.return_value = df  # Return unchanged for simplicity
            _ = _transform(df, transform="power", transformer_fn=transformer_path)

            # Should call _power_transform
            mock_power.assert_called_once()


def test_transform_norm():
    """Test transform function with normalization."""
    df = pd.DataFrame({"X1": [0.1, 0.2, 0.3]})

    with tempfile.NamedTemporaryFile(suffix=".pkl") as temp:
        transformer_path = Path(temp.name)

        with patch("src.data.build_try_traits._standard_normalize") as mock_norm:
            mock_norm.return_value = df  # Return unchanged for simplicity
            _ = _transform(df, transform="norm", transformer_fn=transformer_path)

            # Should call _standard_normalize
            mock_norm.assert_called_once()


def test_transform_unknown():
    """Test transform function with unknown transformation."""
    df = pd.DataFrame({"X1": [0.1, 0.2, 0.3]})

    with tempfile.NamedTemporaryFile(suffix=".pkl") as temp:
        transformer_path = Path(temp.name)

        with pytest.raises(ValueError):
            _transform(df, transform="unknown", transformer_fn=transformer_path)
