"""Unit tests for trait_utils.py."""

from unittest.mock import mock_open, patch

import pandas as pd
import pytest
from box import ConfigBox

from src.utils.splot_utils import filter_certain_plots
from src.utils.trait_utils import (
    check_for_existing_maps,
    clean_species_name,
    filter_pft,
    format_traits_to_process,
    genus_species_caps,
    get_active_traits,
    get_trait_name_from_id,
    get_trait_number_from_id,
    get_traits_to_process,
    load_trait_mapping,
    load_try_traits,
    trim_species_name,
)


class TestGenusSpeciesCaps:
    def test_basic_functionality(self):
        # Test with properly formatted input
        input_series = pd.Series(
            ["Quercus robur", "PINUS SYLVESTRIS", "acer pseudoplatanus"]
        )
        expected = pd.Series(
            ["Quercus robur", "Pinus sylvestris", "Acer pseudoplatanus"]
        )

        result = genus_species_caps(input_series)

        # We need to check values due to dtype differences with pyarrow
        assert result.tolist() == expected.tolist()

    def test_empty_series(self):
        input_series = pd.Series([], dtype="object")
        result = genus_species_caps(input_series)
        assert len(result) == 0


class TestTrimSpeciesName:
    def test_basic_functionality(self):
        # Test with genus and species names
        input_series = pd.Series(
            ["Quercus robur", "Pinus sylvestris L.", "Acer pseudoplatanus 1234"]
        )
        expected = pd.Series(
            ["Quercus robur", "Pinus sylvestris", "Acer pseudoplatanus"]
        )

        result = trim_species_name(input_series)

        # Check values without considering data type
        assert result.tolist() == expected.tolist()

    def test_with_subspecies(self):
        # Test with subspecies that should be trimmed
        input_series = pd.Series(
            ["Quercus robur subsp. pedunculata", "Pinus sylvestris var. mongolica"]
        )
        expected = pd.Series(["Quercus robur", "Pinus sylvestris"])

        result = trim_species_name(input_series)

        assert result.tolist() == expected.tolist()

    def test_invalid_names(self):
        # Test with invalid or incomplete names
        input_series = pd.Series(["Quercus", "123", ""])

        result = trim_species_name(input_series)

        # Should return NA for invalid names
        assert all(pd.isna(result))


class TestCleanSpeciesName:
    def test_basic_functionality(self):
        # Test with proper species names
        df = pd.DataFrame(
            {
                "species": [
                    "Quercus ROBUR",
                    "Pinus sylvestris L.",
                    "Acer pseudoplatanus subsp.",
                ]
            }
        )

        result = clean_species_name(df, "species")

        expected_species = ["quercus robur", "pinus sylvestris", "acer pseudoplatanus"]
        assert result["species"].tolist() == expected_species

    def test_with_new_column(self):
        # Test with creating a new column
        df = pd.DataFrame({"original": ["Quercus ROBUR", "Pinus sylvestris L."]})

        result = clean_species_name(df, "original", "cleaned")

        assert "cleaned" in result.columns
        assert "original" in result.columns
        assert result["original"].tolist() == ["Quercus ROBUR", "Pinus sylvestris L."]
        assert result["cleaned"].tolist() == ["quercus robur", "pinus sylvestris"]

    def test_missing_values(self):
        # Test with missing values
        df = pd.DataFrame({"species": ["Quercus robur", None, ""]})

        result = clean_species_name(df, "species")

        # Should drop rows with missing values
        assert len(result) == 1
        assert result["species"].tolist() == ["quercus robur"]


class TestFilterPFT:
    def test_single_pft(self):
        df = pd.DataFrame({"pft": ["Tree", "Shrub", "Grass", "Tree"]})

        result = filter_pft(df, "Tree")

        assert len(result) == 2
        assert all(result["pft"] == "Tree")

    def test_multiple_pfts(self):
        df = pd.DataFrame({"pft": ["Tree", "Shrub", "Grass", "Tree"]})

        result = filter_pft(df, "Tree_Shrub")

        assert len(result) == 3
        assert set(result["pft"].unique()) == {"Tree", "Shrub"}

    def test_custom_column(self):
        df = pd.DataFrame({"plant_type": ["Tree", "Shrub", "Grass", "Tree"]})

        result = filter_pft(df, "Tree", pft_col="plant_type")

        assert len(result) == 2
        assert all(result["plant_type"] == "Tree")

    def test_invalid_pft(self):
        df = pd.DataFrame({"pft": ["Tree", "Shrub", "Grass", "Tree"]})

        with pytest.raises(ValueError, match="Invalid PFT designation"):
            filter_pft(df, "Invalid")


class TestGetActiveTraits:
    def test_get_active_traits(self):
        # Create test configuration
        mock_config = ConfigBox(
            {
                "datasets": {
                    "Y": {
                        "traits": [1, 3, 5],
                        "trait_stats": ["mean", "median", "std"],
                        "trait_stat": 1,
                    }
                }
            }
        )

        result = get_active_traits(mock_config)

        expected = ["X1_mean", "X3_mean", "X5_mean"]
        assert result == expected


class TestGetTraitNumberFromId:
    def test_basic_functionality(self):
        assert get_trait_number_from_id("X123_mean") == "123"
        assert get_trait_number_from_id("X4") == "4"
        assert get_trait_number_from_id("PC1") == "1"

    def test_no_number(self):
        with pytest.raises(ValueError, match="Could not extract trait number"):
            get_trait_number_from_id("Xabc")


class TestLoadTraitMapping:
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"1": {"short": "LMA", "long": "Leaf Mass per Area", "unit": "g/m²"}}',
    )
    def test_load_mapping(self, mock_file):
        mock_config = ConfigBox({"trait_mapping": "dummy_path.json"})

        result = load_trait_mapping(mock_config)

        mock_file.assert_called_once_with("dummy_path.json", encoding="utf-8")
        assert isinstance(result, dict)
        assert "1" in result
        assert result["1"]["short"] == "LMA"


class TestGetTraitNameFromId:
    mapping = ConfigBox(
        {
            "3": {"short": "LMA", "long": "Leaf Mass per Area", "unit": "g/m²"},
            "14": {"short": "SLA", "long": "Specific Leaf Area", "unit": "mm²/mg"},
        }
    )

    @pytest.fixture(autouse=True)
    def patch_load_trait_mapping(self, monkeypatch):
        """Automatically patch load_trait_mapping for all tests in this class."""
        monkeypatch.setattr(
            "src.utils.trait_utils.load_trait_mapping", lambda cfg=None: self.mapping
        )
        # The fixture value isn't used, we just need the patching side-effect

    def test_get_name_short(self):
        """Test getting the short name of a trait."""
        name, unit = get_trait_name_from_id("X3_mean", "short")

        assert name == "LMA"
        assert unit == "g/m²"

    def test_get_name_long(self):
        """Test getting the long name of a trait."""
        name, unit = get_trait_name_from_id("X3_mean", "long")

        assert name == "Leaf Mass per Area"
        assert unit == "g/m²"

    def test_invalid_trait(self):
        """Test behavior with an invalid trait number."""
        with pytest.raises(ValueError, match="Trait number 999 not in mapping"):
            get_trait_name_from_id("X999_mean")

    def test_invalid_length(self):
        """Test behavior with an invalid length specification."""
        with pytest.raises(ValueError, match="Length medium not in mapping"):
            get_trait_name_from_id("X3_mean", "medium")


class TestGetTraitsToProcessPCA:
    """Tests for get_traits_to_process function using PCA data."""

    @pytest.fixture(autouse=True)
    def patch_get_try_traits_interim_fn(self, monkeypatch):
        """Patch get_try_traits_interim_fn to return a PCA file path."""
        monkeypatch.setattr(
            "src.utils.trait_utils.get_try_traits_interim_fn",
            lambda: "tests/test_data/interim/try/traits_PCA4.parquet",
        )

    def test_pca_mode_with_trait_id(self):
        with pytest.raises(
            ValueError, match="Cannot specify a trait ID when using PCA"
        ):
            get_traits_to_process([1, 2, 3], True, 1)

    def test_pca_mode_no_trait_id(self):
        result = get_traits_to_process([1, 2], True, None)
        assert result == ["PC1", "PC2", "PC3", "PC4"]


class TestGetTraitsToProcessNonPCA:
    """Tests for get_traits_to_process function using TRY6 data."""

    @pytest.fixture(autouse=True)
    def patch_get_try_traits_interim_fn(self, monkeypatch):
        """Patch get_try_traits_interim_fn to return a TRY6 file path."""
        monkeypatch.setattr(
            "src.utils.trait_utils.get_try_traits_interim_fn",
            lambda: "tests/test_data/interim/try/traits_TRY6.parquet",
        )

    def test_specific_trait_id_valid(self):
        result = get_traits_to_process([1, 2, 3], False, 2)
        assert result == ["X2"]

    def test_all_traits(self):
        result = get_traits_to_process([1, 2, 3], False, None)
        assert result == ["X1", "X2", "X3"]


class TestCheckForExistingMaps:
    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for testing."""
        return tmp_path

    def test_all_exist_traits_no_fd(self, temp_dir):
        """Test when all trait files exist."""
        # Create temporary files for all traits
        for trait in ["X1", "X2", "X3"]:
            (temp_dir / f"{trait}.tif").touch()

        result_traits, result_fd = check_for_existing_maps(
            temp_dir, ["X1", "X2", "X3"], []
        )

        # All files exist, so none should be processed
        assert result_traits == []
        assert result_fd == []

    def test_partial_exist_traits_no_fd(self, temp_dir):
        """Test when only some trait files exist."""
        # Create only one trait file
        (temp_dir / "X1.tif").touch()

        result_traits, result_fd = check_for_existing_maps(
            temp_dir, ["X1", "X2", "X3"], []
        )

        # Only X2 and X3 should be processed
        assert sorted(result_traits) == ["X2", "X3"]
        assert result_fd == []

    def test_fd_metrics(self, temp_dir):
        """Test when some FD metrics files exist."""
        # Create FD metrics files
        (temp_dir / "f_ric.tif").touch()
        (temp_dir / "f_eve.tif").touch()

        result_traits, result_fd = check_for_existing_maps(
            temp_dir, ["X1", "X2"], ["f_ric", "f_eve", "f_div"]
        )

        # With FD metrics to process, original traits are kept
        assert result_traits == ["X1", "X2"]
        assert result_fd == ["f_div"]

    def test_all_fd_exist(self, temp_dir):
        """Test when all FD metrics files exist."""
        # Create all FD metrics files
        (temp_dir / "f_ric.tif").touch()
        (temp_dir / "f_eve.tif").touch()

        result_traits, result_fd = check_for_existing_maps(
            temp_dir, ["X1", "X2"], ["f_ric", "f_eve"]
        )

        # All FD metrics exist, so none should be processed
        assert result_traits == []
        assert result_fd == []


class TestLoadTryTraitsPCA:
    @pytest.fixture(autouse=True)
    def patch_get_try_traits_interim_fn(self, monkeypatch):
        """Patch get_try_traits_interim_fn to return a PCA file path."""
        monkeypatch.setattr(
            "src.utils.trait_utils.get_try_traits_interim_fn",
            lambda: "tests/test_data/interim/try/traits_PCA4.parquet",
        )

    def test_load_try_traits_pca(self):
        # Call the function
        result = load_try_traits(1, ["PC1", "PC2"])

        # Verify correct columns were requested
        assert set(result.columns) == {"PC1", "PC2"}
        assert result.index.name == "speciesname"


class TestLoadTryTraitsNonPCA:
    @pytest.fixture(autouse=True)
    def patch_get_try_traits_interim_fn(self, monkeypatch):
        """Patch get_try_traits_interim_fn to return a TRY6 file path."""
        monkeypatch.setattr(
            "src.utils.trait_utils.get_try_traits_interim_fn",
            lambda: "tests/test_data/interim/try/traits_TRY6.parquet",
        )

    def test_load_try_traits_non_pca(self):
        # Call the function
        result = load_try_traits(1, ["X14_mean", "X50_mean"])

        # Verify correct columns were requested
        assert set(result.columns) == {"X14_mean", "X50_mean"}
        assert result.index.name == "speciesname"


class TestFilterCertainPlots:
    def test_filter_plots(self):
        """Test filtering plots by GIVD_NU."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "GIVD_NU": ["AF-00-001", "EU-00-002", "AF-00-001", "AS-00-003"],
                "value": [1, 2, 3, 4],
            }
        )

        # Filter out AF-00-001 plots
        result = filter_certain_plots(df, givd_col="GIVD_NU", givd="AF-00-001")

        # Verify only the non-AF-00-001 plots remain
        assert len(result) == 2
        assert set(result["GIVD_NU"]) == {"EU-00-002", "AS-00-003"}
        assert set(result["value"]) == {2, 4}

    def test_no_matching_plots(self):
        """Test when no plots match the filter criteria."""
        # Create test DataFrame
        df = pd.DataFrame(
            {"GIVD_NU": ["EU-00-001", "EU-00-002", "EU-00-003"], "value": [1, 2, 3]}
        )

        # Filter out AF-00-001 plots (none exist)
        result = filter_certain_plots(df, givd_col="GIVD_NU", givd="AF-00-001")

        # Verify all plots remain
        assert len(result) == 3
        assert result.equals(df)

    def test_empty_dataframe(self):
        """Test with an empty DataFrame."""
        df = pd.DataFrame({"GIVD_NU": [], "value": []})
        result = filter_certain_plots(df, givd_col="GIVD_NU", givd="AF-00-001")
        assert len(result) == 0


class TestFormatTraitsToProcess:
    def test_specific_trait(self):
        """Test with a specific valid trait ID."""
        valid_traits = [1, 3, 5, 7]
        result = format_traits_to_process(3, valid_traits)

        assert result == ["X3"]

    def test_invalid_trait(self):
        """Test with an invalid trait ID."""
        valid_traits = [1, 3, 5, 7]

        with pytest.raises(
            ValueError, match="Invalid trait ID: 2. Valid traits are: 1, 3, 5, 7"
        ):
            format_traits_to_process(2, valid_traits)

    def test_all_traits(self):
        """Test processing all traits."""
        valid_traits = [1, 3, 5]
        result = format_traits_to_process(None, valid_traits)

        assert result == ["X1", "X3", "X5"]

    def test_empty_valid_traits(self):
        """Test with an empty list of valid traits."""
        valid_traits = []
        result = format_traits_to_process(None, valid_traits)

        assert result == []
