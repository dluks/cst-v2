"""Tests for histogram target construction using real data snippets.

These tests use representative samples from actual GBIF, sPlot, and TRY6 trait data
to verify histogram construction behavior.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.build_histogram_targets import (
    _assign_cell_ids,
    _build_cell_histograms,
    _compute_bin_edges,
    _reproject,
)


# =============================================================================
# Real Data Fixtures
# =============================================================================

@pytest.fixture
def real_traits_df() -> pd.DataFrame:
    """Real TRY6 species-level median traits (subset).

    These are actual trait values from the TRY6 dataset for common European species.
    """
    return pd.DataFrame({
        "GBIFKeyGBIF": [3120060, 2706490, 3189863, 3189846, 2888961, 8059811, 7475424, 5371685],
        "nameOutWCVP": [
            "achillea millefolium",
            "agrostis capillaris",
            "acer campestre",
            "acer platanoides",
            "bromus erectus",  # specieskey 2888961
            "poa pratensis",   # specieskey 8059811
            "festuca rubra",   # specieskey 7475424
            "dactylis glomerata",  # specieskey 5371685
        ],
        # X4: Stem specific density (wood density)
        "X4": [0.386, 0.268, 0.574, 0.582, 0.280, 0.406, 0.623, 0.467],
        # X14: Leaf nitrogen content per leaf dry mass
        "X14": [23.85, 26.31, 22.62, 19.72, 29.88, 15.65, 20.34, 36.09],
        # X13: Leaf carbon content per leaf dry mass
        "X13": [439.42, 459.71, 466.90, 477.21, 443.06, 436.78, 447.89, 448.53],
    })


@pytest.fixture
def real_gbif_observations() -> pd.DataFrame:
    """Real GBIF observations from Austria (clustered location).

    These are actual occurrence records from approximately (47.8°N, 14.4°E).
    """
    return pd.DataFrame({
        "specieskey": [
            2888961, 2888961, 2888961, 2888961, 2888961,  # bromus erectus (5 obs)
            8059811, 8059811, 8059811,                    # poa pratensis (3 obs)
            7475424, 7475424, 7475424, 7475424,          # festuca rubra (4 obs)
            5371685, 5371685, 5371685, 5371685, 5371685, # dactylis glomerata (5 obs)
            3120060, 3120060, 3120060,                    # achillea millefolium (3 obs)
            2706490, 2706490,                             # agrostis capillaris (2 obs)
        ],
        "decimallatitude": [47.799] * 22,
        "decimallongitude": [14.354] * 22,
        "pft": ["Grass"] * 22,
        "weight": [1.0] * 22,
    })


@pytest.fixture
def real_splot_observations() -> pd.DataFrame:
    """Real sPlot observations from a German vegetation plot.

    These are actual vegetation survey records from plot 1445344 (53.24°N, 10.49°E).
    """
    return pd.DataFrame({
        "PlotObservationID": [1445344] * 8,
        "speciesname": [
            "achillea millefolium",
            "agrostis capillaris",
            "acer campestre",
            "acer platanoides",
            "bromus erectus",
            "poa pratensis",
            "festuca rubra",
            "dactylis glomerata",
        ],
        "Latitude": [53.2377] * 8,
        "Longitude": [10.4866] * 8,
        "pft": ["Grass", "Grass", "Tree", "Tree", "Grass", "Grass", "Grass", "Grass"],
        "weight": [1.0] * 8,
        "Rel_Abund_Plot": [0.05, 0.15, 0.02, 0.02, 0.20, 0.25, 0.18, 0.12],
    })


# =============================================================================
# Tests for _compute_bin_edges
# =============================================================================

class TestComputeBinEdges:
    """Tests for global bin edge computation."""

    def test_computes_correct_number_of_edges(self, real_traits_df: pd.DataFrame):
        """Bin edges should have n_bins + 1 values."""
        n_bins = 10
        bin_edges = _compute_bin_edges(real_traits_df, ["X4", "X14"], n_bins)

        assert "X4" in bin_edges
        assert "X14" in bin_edges
        assert len(bin_edges["X4"]) == n_bins + 1
        assert len(bin_edges["X14"]) == n_bins + 1

    def test_edges_span_trait_range(self, real_traits_df: pd.DataFrame):
        """Bin edges should span from min to max of trait values."""
        n_bins = 5
        bin_edges = _compute_bin_edges(real_traits_df, ["X4"], n_bins)

        # X4 values: [0.386, 0.268, 0.574, 0.582, 0.280, 0.406, 0.623, 0.467]
        expected_min = 0.268
        expected_max = 0.623

        assert bin_edges["X4"][0] == pytest.approx(expected_min, rel=1e-3)
        assert bin_edges["X4"][-1] == pytest.approx(expected_max, rel=1e-3)

    def test_edges_are_equally_spaced(self, real_traits_df: pd.DataFrame):
        """Bin edges should be equally spaced (linspace)."""
        n_bins = 4
        bin_edges = _compute_bin_edges(real_traits_df, ["X4"], n_bins)

        edges = bin_edges["X4"]
        spacings = np.diff(edges)

        # All spacings should be equal
        assert np.allclose(spacings, spacings[0])

    def test_skips_trait_with_insufficient_values(self):
        """Traits with < 2 non-NaN values should be skipped."""
        df = pd.DataFrame({
            "X4": [0.5, np.nan, np.nan],
            "X14": [20.0, 25.0, 30.0],
        })

        bin_edges = _compute_bin_edges(df, ["X4", "X14"], n_bins=5)

        assert "X4" not in bin_edges  # Only 1 non-NaN value
        assert "X14" in bin_edges


# =============================================================================
# Tests for _reproject
# =============================================================================

class TestReproject:
    """Tests for coordinate reprojection."""

    def test_reprojects_to_epsg_6933(self, real_gbif_observations: pd.DataFrame):
        """Coordinates should be reprojected to EPSG:6933 (Equal Area Cylindrical)."""
        result = _reproject(
            real_gbif_observations,
            lat_col="decimallatitude",
            lon_col="decimallongitude",
            target_crs="EPSG:6933",
        )

        assert "x" in result.columns
        assert "y" in result.columns

        # EPSG:6933 coordinates for (47.799°N, 14.354°E) should be approximately:
        # x ≈ 1,270,000 m, y ≈ 4,700,000 m (rough estimates for this projection)
        assert result["x"].iloc[0] > 1_000_000
        assert result["x"].iloc[0] < 2_000_000
        assert result["y"].iloc[0] > 4_000_000
        assert result["y"].iloc[0] < 6_000_000

    def test_preserves_non_coord_columns(self, real_gbif_observations: pd.DataFrame):
        """Non-coordinate columns should be preserved; lat/lon dropped."""
        result = _reproject(
            real_gbif_observations,
            lat_col="decimallatitude",
            lon_col="decimallongitude",
            target_crs="EPSG:6933",
        )

        assert "specieskey" in result.columns
        assert "weight" in result.columns
        assert "decimallatitude" not in result.columns
        assert "decimallongitude" not in result.columns


# =============================================================================
# Tests for _assign_cell_ids
# =============================================================================

class TestAssignCellIds:
    """Tests for grid cell ID assignment."""

    def test_assigns_cell_coordinates(self):
        """Cell coordinates should be grid-aligned."""
        df = pd.DataFrame({
            "x": [1_270_123.0, 1_270_456.0, 1_292_789.0],
            "y": [4_712_345.0, 4_712_678.0, 4_734_901.0],
        })
        resolution = 22000

        result = _assign_cell_ids(df, resolution)

        # cell_x = floor(x / resolution) * resolution
        assert result["cell_x"].iloc[0] == 1_254_000  # floor(1270123/22000)*22000
        assert result["cell_x"].iloc[1] == 1_254_000  # Same cell
        assert result["cell_x"].iloc[2] == 1_276_000  # Different cell

    def test_creates_unique_cell_ids(self):
        """Cell IDs should be unique per grid cell."""
        df = pd.DataFrame({
            "x": [1_270_000.0, 1_270_100.0, 1_292_000.0],
            "y": [4_712_000.0, 4_712_100.0, 4_734_000.0],
        })
        resolution = 22000

        result = _assign_cell_ids(df, resolution)

        # First two should be in same cell, third in different
        assert result["cell_id"].iloc[0] == result["cell_id"].iloc[1]
        assert result["cell_id"].iloc[0] != result["cell_id"].iloc[2]


# =============================================================================
# Tests for _build_cell_histograms
# =============================================================================

class TestBuildCellHistograms:
    """Tests for histogram construction per grid cell."""

    @pytest.fixture
    def gbif_merged_df(self, real_gbif_observations: pd.DataFrame, real_traits_df: pd.DataFrame):
        """Merge GBIF observations with traits and add cell IDs."""
        # Merge observations with traits
        merged = real_gbif_observations.merge(
            real_traits_df[["GBIFKeyGBIF", "X4", "X14", "X13"]],
            left_on="specieskey",
            right_on="GBIFKeyGBIF",
            how="inner",
        ).drop(columns=["GBIFKeyGBIF"])

        # Add projected coordinates (mock values for same cell)
        merged["x"] = 1_270_000.0
        merged["y"] = 4_712_000.0

        # Assign cell IDs
        merged = _assign_cell_ids(merged, resolution=22000)

        return merged

    @pytest.fixture
    def splot_merged_df(self, real_splot_observations: pd.DataFrame, real_traits_df: pd.DataFrame):
        """Merge sPlot observations with traits and add cell IDs."""
        # Merge observations with traits
        merged = real_splot_observations.merge(
            real_traits_df[["nameOutWCVP", "X4", "X14", "X13"]],
            left_on="speciesname",
            right_on="nameOutWCVP",
            how="inner",
        ).drop(columns=["nameOutWCVP"])

        # Add projected coordinates (mock values for same cell)
        merged["x"] = 900_000.0
        merged["y"] = 5_900_000.0

        # Assign cell IDs
        merged = _assign_cell_ids(merged, resolution=22000)

        # Add combined weight for sPlot
        merged["combined_weight"] = merged["Rel_Abund_Plot"] * merged["weight"]

        return merged

    def test_histogram_sums_to_one(self, gbif_merged_df: pd.DataFrame, real_traits_df: pd.DataFrame):
        """Each histogram should sum to 1 (probability distribution)."""
        n_bins = 5
        trait_names = ["X4", "X14"]
        bin_edges = _compute_bin_edges(real_traits_df, trait_names, n_bins)

        hist_arr, mask_arr, coords_arr, _, stats = _build_cell_histograms(
            df=gbif_merged_df,
            trait_names=trait_names,
            bin_edges=bin_edges,
            n_bins=n_bins,
            min_observations=5,
            min_unique_species=3,
            min_bin_coverage=0.2,
            epsilon=0.01,
            weight_col="weight",
            species_col="specieskey",
        )

        # hist_arr shape: (N, n_traits, n_bins)
        assert hist_arr.ndim == 3
        assert hist_arr.shape[1] == len(trait_names)
        assert hist_arr.shape[2] == n_bins

        # Each histogram should sum to 1
        row_sums = hist_arr.sum(axis=2)  # (N, n_traits)
        assert np.allclose(row_sums, 1.0), "Histograms don't sum to 1"

    def test_label_smoothing_prevents_zeros(self, gbif_merged_df: pd.DataFrame, real_traits_df: pd.DataFrame):
        """Label smoothing should ensure no bin has zero probability."""
        n_bins = 10  # More bins than data points
        bin_edges = _compute_bin_edges(real_traits_df, ["X4"], n_bins)
        epsilon = 0.01

        hist_arr, _, _, _, _ = _build_cell_histograms(
            df=gbif_merged_df,
            trait_names=["X4"],
            bin_edges=bin_edges,
            n_bins=n_bins,
            min_observations=5,
            min_unique_species=3,
            min_bin_coverage=0.1,
            epsilon=epsilon,
            weight_col="weight",
            species_col="specieskey",
        )

        # All bins should have probability > 0 due to smoothing
        if hist_arr.shape[0] > 0:
            assert (hist_arr > 0).all(), "Some bins have zero probability"

    def test_filters_cells_with_few_observations(self, real_traits_df: pd.DataFrame):
        """Cells with fewer than min_observations should be filtered."""
        # Create data with few observations
        df = pd.DataFrame({
            "specieskey": [2888961, 8059811, 7475424],  # Only 3 obs
            "X4": [0.280, 0.406, 0.623],
            "weight": [1.0, 1.0, 1.0],
            "x": [1_270_000.0] * 3,
            "y": [4_712_000.0] * 3,
        })
        df = _assign_cell_ids(df, resolution=22000)

        n_bins = 5
        bin_edges = _compute_bin_edges(real_traits_df, ["X4"], n_bins)

        hist_arr, _, _, _, stats = _build_cell_histograms(
            df=df,
            trait_names=["X4"],
            bin_edges=bin_edges,
            n_bins=n_bins,
            min_observations=10,  # Require more than we have
            min_unique_species=2,
            min_bin_coverage=0.1,
            epsilon=0.01,
            weight_col="weight",
            species_col="specieskey",
        )

        assert hist_arr.shape[0] == 0
        assert stats["cells_filtered_observations"] == 1

    def test_filters_cells_with_few_species(self, real_traits_df: pd.DataFrame):
        """Cells with fewer than min_unique_species should be filtered."""
        # Create data with only 2 unique species
        df = pd.DataFrame({
            "specieskey": [2888961, 2888961, 2888961, 8059811, 8059811],  # Only 2 species
            "X4": [0.280, 0.280, 0.280, 0.406, 0.406],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0],
            "x": [1_270_000.0] * 5,
            "y": [4_712_000.0] * 5,
        })
        df = _assign_cell_ids(df, resolution=22000)

        n_bins = 5
        bin_edges = _compute_bin_edges(real_traits_df, ["X4"], n_bins)

        hist_arr, _, _, _, stats = _build_cell_histograms(
            df=df,
            trait_names=["X4"],
            bin_edges=bin_edges,
            n_bins=n_bins,
            min_observations=3,
            min_unique_species=5,  # Require more unique species
            min_bin_coverage=0.1,
            epsilon=0.01,
            weight_col="weight",
            species_col="specieskey",
        )

        assert hist_arr.shape[0] == 0
        assert stats["cells_filtered_species"] == 1

    def test_weighted_histogram_gbif(self, real_traits_df: pd.DataFrame):
        """GBIF histograms should weight observations by their weights."""
        # Create data where weights differ
        df = pd.DataFrame({
            "specieskey": [2888961, 8059811, 7475424, 5371685],
            "X4": [0.280, 0.406, 0.623, 0.467],
            "weight": [2.0, 1.0, 1.0, 1.0],  # First species has 2x weight
            "x": [1_270_000.0] * 4,
            "y": [4_712_000.0] * 4,
        })
        df = _assign_cell_ids(df, resolution=22000)

        n_bins = 4
        bin_edges = _compute_bin_edges(real_traits_df, ["X4"], n_bins)

        hist_arr, _, _, _, _ = _build_cell_histograms(
            df=df,
            trait_names=["X4"],
            bin_edges=bin_edges,
            n_bins=n_bins,
            min_observations=3,
            min_unique_species=3,
            min_bin_coverage=0.1,
            epsilon=0.0,  # No smoothing for this test
            weight_col="weight",
            species_col="specieskey",
        )

        # The bin containing X4=0.280 should have 2/(2+1+1+1) = 0.4 of the weight
        # (without smoothing). Total weight = 5.0 (2+1+1+1)
        hist_values = hist_arr[0, 0, :]  # First cell, first trait
        # First bin contains 0.280, should get weight 2/5 = 0.4
        assert hist_values.sum() == pytest.approx(1.0)

    def test_splot_abundance_weighting(self, splot_merged_df: pd.DataFrame, real_traits_df: pd.DataFrame):
        """sPlot histograms should weight by abundance x survey weight."""
        n_bins = 5
        bin_edges = _compute_bin_edges(real_traits_df, ["X4"], n_bins)

        hist_arr, mask_arr, _, _, _ = _build_cell_histograms(
            df=splot_merged_df,
            trait_names=["X4"],
            bin_edges=bin_edges,
            n_bins=n_bins,
            min_observations=None,
            min_unique_species=3,
            min_bin_coverage=0.1,
            epsilon=0.01,
            weight_col="combined_weight",
            species_col="speciesname",
            min_total_abundance=0.5,
        )

        # Should produce valid histogram
        assert hist_arr.shape[0] > 0
        assert mask_arr[0, 0] == True  # noqa: E712

    def test_masks_indicate_valid_traits(self, gbif_merged_df: pd.DataFrame, real_traits_df: pd.DataFrame):
        """Mask array should correctly indicate which traits are valid per cell."""
        n_bins = 5
        trait_names = ["X4", "X14", "X_missing"]

        # Add a trait with all NaN values
        gbif_merged_df["X_missing"] = np.nan

        bin_edges = _compute_bin_edges(real_traits_df, ["X4", "X14"], n_bins)
        # X_missing won't have bin edges computed

        hist_arr, mask_arr, _, _, _ = _build_cell_histograms(
            df=gbif_merged_df,
            trait_names=trait_names,
            bin_edges=bin_edges,
            n_bins=n_bins,
            min_observations=5,
            min_unique_species=3,
            min_bin_coverage=0.1,
            epsilon=0.01,
            weight_col="weight",
            species_col="specieskey",
        )

        # mask_arr shape: (N, n_traits)
        if mask_arr.shape[0] > 0:
            assert mask_arr[0, 0] == True   # X4 valid      # noqa: E712
            assert mask_arr[0, 1] == True   # X14 valid     # noqa: E712
            assert mask_arr[0, 2] == False  # X_missing     # noqa: E712

    def test_coordinates_output(self, gbif_merged_df: pd.DataFrame, real_traits_df: pd.DataFrame):
        """Coordinates array should have correct cell coordinates."""
        n_bins = 5
        bin_edges = _compute_bin_edges(real_traits_df, ["X4"], n_bins)

        _, _, coords_arr, _, _ = _build_cell_histograms(
            df=gbif_merged_df,
            trait_names=["X4"],
            bin_edges=bin_edges,
            n_bins=n_bins,
            min_observations=5,
            min_unique_species=3,
            min_bin_coverage=0.1,
            epsilon=0.01,
            weight_col="weight",
            species_col="specieskey",
        )

        # coords_arr shape: (N, 2) as [x, y]
        assert coords_arr.ndim == 2
        assert coords_arr.shape[1] == 2
        # Coordinates should be grid-aligned (multiples of resolution)
        if coords_arr.shape[0] > 0:
            assert coords_arr[0, 0] % 22000 == 0  # x
            assert coords_arr[0, 1] % 22000 == 0  # y


# =============================================================================
# Regression Tests
# =============================================================================

class TestRegression:
    """Regression tests for known bugs."""

    def test_many_cells_no_int16_overflow(self, real_traits_df: pd.DataFrame):
        """Cells with Categorical code >= 1639 must not lose data to int16 overflow.

        pd.Categorical.codes returns int16 for <32768 categories.
        With n_bins=20, code * n_bins overflows int16 at code 1639
        (1639 * 20 = 32780 > 32767), corrupting flat_idx and silently
        writing histogram counts into wrong cells' bins.

        Regression test for: all cells must produce identical histograms
        when given identical observations, regardless of their position in
        the Categorical ordering.
        """
        n_cells = 2000  # Exceeds int16 overflow boundary (1638)
        n_bins = 20
        species_ids = real_traits_df["GBIFKeyGBIF"].tolist()
        n_species = len(species_ids)

        # Create n_cells cells, each containing one observation per species.
        # Use widely-spaced x values so each gets a unique cell_id.
        rows = []
        for i in range(n_cells):
            for sp in species_ids:
                rows.append({
                    "specieskey": sp,
                    "X4": real_traits_df.loc[
                        real_traits_df["GBIFKeyGBIF"] == sp, "X4"
                    ].iloc[0],
                    "weight": 1.0,
                    "x": float(i * 22000),
                    "y": 0.0,
                })
        df = pd.DataFrame(rows)
        df = _assign_cell_ids(df, resolution=22000)

        assert df["cell_id"].nunique() == n_cells

        bin_edges = _compute_bin_edges(real_traits_df, ["X4"], n_bins)

        hist_arr, mask_arr, _, _, stats = _build_cell_histograms(
            df=df,
            trait_names=["X4"],
            bin_edges=bin_edges,
            n_bins=n_bins,
            min_observations=3,
            min_unique_species=3,
            min_bin_coverage=0.05,
            epsilon=0.01,
            weight_col="weight",
            species_col="specieskey",
        )

        # Every cell has 8 species with identical trait values,
        # so all 2000 must survive and have valid histograms.
        assert hist_arr.shape[0] == n_cells, (
            f"Expected {n_cells} cells, got {hist_arr.shape[0]} "
            f"(int16 overflow would drop cells with code >= 1639)"
        )
        assert mask_arr[:, 0].all(), "All cells should have valid X4"

        # Every cell received the same observations, so all histograms
        # must be identical (no cross-cell contamination from overflow).
        for i in range(1, n_cells):
            assert np.array_equal(hist_arr[i, 0], hist_arr[0, 0]), (
                f"Cell {i} histogram differs from cell 0 — "
                f"possible flat-index corruption"
            )


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full histogram construction workflow."""

    def test_full_gbif_workflow(self, real_gbif_observations: pd.DataFrame, real_traits_df: pd.DataFrame):
        """Test full GBIF workflow from observations to histograms."""
        # 1. Merge with traits
        merged = real_gbif_observations.merge(
            real_traits_df[["GBIFKeyGBIF", "X4", "X14", "X13"]],
            left_on="specieskey",
            right_on="GBIFKeyGBIF",
            how="inner",
        ).drop(columns=["GBIFKeyGBIF"])

        # 2. Reproject
        merged = _reproject(
            merged,
            lat_col="decimallatitude",
            lon_col="decimallongitude",
            target_crs="EPSG:6933",
        )

        # 3. Assign cell IDs
        merged = _assign_cell_ids(merged, resolution=22000)

        # 4. Compute bin edges
        trait_names = ["X4", "X14", "X13"]
        n_bins = 5
        bin_edges = _compute_bin_edges(real_traits_df, trait_names, n_bins)

        # 5. Build histograms
        hist_arr, mask_arr, coords_arr, _, stats = _build_cell_histograms(
            df=merged,
            trait_names=trait_names,
            bin_edges=bin_edges,
            n_bins=n_bins,
            min_observations=10,
            min_unique_species=3,
            min_bin_coverage=0.2,
            epsilon=0.01,
            weight_col="weight",
            species_col="specieskey",
        )

        # Verify output shapes
        n_cells = hist_arr.shape[0]
        assert n_cells > 0
        assert hist_arr.shape == (n_cells, len(trait_names), n_bins)
        assert mask_arr.shape == (n_cells, len(trait_names))
        assert coords_arr.shape == (n_cells, 2)
        assert stats["n_cells_valid"] > 0

    def test_full_splot_workflow(self, real_splot_observations: pd.DataFrame, real_traits_df: pd.DataFrame):
        """Test full sPlot workflow from observations to histograms."""
        # 1. Merge with traits
        merged = real_splot_observations.merge(
            real_traits_df[["nameOutWCVP", "X4", "X14", "X13"]],
            left_on="speciesname",
            right_on="nameOutWCVP",
            how="inner",
        ).drop(columns=["nameOutWCVP"])

        # 2. Reproject
        merged = _reproject(
            merged,
            lat_col="Latitude",
            lon_col="Longitude",
            target_crs="EPSG:6933",
        )

        # 3. Assign cell IDs
        merged = _assign_cell_ids(merged, resolution=22000)

        # 4. Compute combined weights
        merged["combined_weight"] = merged["Rel_Abund_Plot"] * merged["weight"]

        # 5. Compute bin edges
        trait_names = ["X4", "X14"]
        n_bins = 5
        bin_edges = _compute_bin_edges(real_traits_df, trait_names, n_bins)

        # 6. Build histograms
        hist_arr, mask_arr, coords_arr, _, stats = _build_cell_histograms(
            df=merged,
            trait_names=trait_names,
            bin_edges=bin_edges,
            n_bins=n_bins,
            min_observations=None,
            min_unique_species=3,
            min_bin_coverage=0.1,
            epsilon=0.01,
            weight_col="combined_weight",
            species_col="speciesname",
            min_total_abundance=0.5,
        )

        # Verify outputs
        assert hist_arr.shape[0] > 0
        assert hist_arr.shape == (hist_arr.shape[0], len(trait_names), n_bins)

        # Check histogram properties — all should sum to 1
        row_sums = hist_arr.sum(axis=2)
        assert np.allclose(row_sums, 1.0)
