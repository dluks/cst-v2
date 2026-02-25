"""Unit tests for the histogram MLP training pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip entire module if torch is not available
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")

from src.models.histogram_mlp.model import HistogramMLP
from src.models.histogram_mlp.loss import MaskedKLDivLoss
from src.models.histogram_mlp.dataset import (
    HistogramDataset,
    preprocess_features,
)
from src.models.histogram_mlp.cv_splits import (
    assign_spatial_folds,
    get_train_val_indices,
)
from src.models.histogram_mlp.evaluate import (
    compute_kl_divergence,
    compute_emd,
    compute_histogram_intersection,
    compute_moment_comparison,
    evaluate_all,
)

# Fixtures (rng, dims, synthetic_data) are in conftest.py


# ===========================================================================
# Model tests
# ===========================================================================


class TestHistogramMLP:
    """Tests for HistogramMLP architecture."""

    def test_output_shape(self, dims):
        N, F, T, B = dims["N"], dims["F"], dims["T"], dims["B"]
        model = HistogramMLP(n_features=F, n_traits=T, n_bins=B)
        x = torch.randn(N, F)
        out = model(x)
        assert out.shape == (N, T, B)

    def test_output_is_log_probabilities(self, dims):
        N, F, T, B = dims["N"], dims["F"], dims["T"], dims["B"]
        model = HistogramMLP(n_features=F, n_traits=T, n_bins=B)
        x = torch.randn(N, F)
        log_probs = model(x)

        # exp(log_probs) should sum to 1 along bin axis
        probs = torch.exp(log_probs)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_all_log_probs_nonpositive(self, dims):
        model = HistogramMLP(
            n_features=dims["F"], n_traits=dims["T"], n_bins=dims["B"]
        )
        x = torch.randn(dims["N"], dims["F"])
        log_probs = model(x)
        assert (log_probs <= 0).all()

    def test_custom_hidden_dims(self):
        model = HistogramMLP(
            n_features=10, n_traits=3, n_bins=5, hidden_dims=[64, 32]
        )
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 3, 5)

    def test_single_sample(self):
        model = HistogramMLP(n_features=10, n_traits=3, n_bins=5)
        x = torch.randn(1, 10)
        out = model(x)
        assert out.shape == (1, 3, 5)


# ===========================================================================
# Loss tests
# ===========================================================================


class TestMaskedKLDivLoss:
    """Tests for MaskedKLDivLoss."""

    def _make_inputs(self, B=8, T=4, bins=20):
        """Create simple inputs for loss testing."""
        # Uniform target
        target = torch.ones(B, T, bins) / bins
        # Slightly perturbed prediction (log-space)
        log_pred = torch.log(target + 0.001 * torch.randn_like(target).abs())
        log_pred = log_pred - torch.logsumexp(log_pred, dim=-1, keepdim=True)
        mask = torch.ones(B, T)
        source = torch.zeros(B, dtype=torch.int8)
        return log_pred, target, mask, source

    def test_basic_forward(self):
        log_pred, target, mask, source = self._make_inputs()
        loss_fn = MaskedKLDivLoss()
        loss = loss_fn(log_pred, target, mask, source)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0  # KL is non-negative

    def test_perfect_prediction_zero_loss(self):
        B, T, bins = 8, 4, 20
        target = torch.ones(B, T, bins) / bins
        log_pred = torch.log(target)
        mask = torch.ones(B, T)
        source = torch.zeros(B, dtype=torch.int8)

        loss_fn = MaskedKLDivLoss()
        loss = loss_fn(log_pred, target, mask, source)
        assert loss.item() < 1e-6

    def test_mask_zeros_out_invalid(self):
        log_pred, target, _, source = self._make_inputs(B=4, T=2)

        # All masked out → loss should be 0
        mask_none = torch.zeros(4, 2)
        loss_fn = MaskedKLDivLoss()
        loss = loss_fn(log_pred, target, mask_none, source)
        assert loss.item() == 0.0

    def test_source_weighting_changes_loss(self):
        log_pred, target, mask, _ = self._make_inputs(B=8, T=4)
        # All sPlot
        source_splot = torch.ones(8, dtype=torch.int8)

        loss_fn_equal = MaskedKLDivLoss(gbif_weight=None)
        loss_fn_weighted = MaskedKLDivLoss(splot_weight=2.0, gbif_weight=0.5)

        loss_equal = loss_fn_equal(log_pred, target, mask, source_splot)
        loss_weighted = loss_fn_weighted(log_pred, target, mask, source_splot)

        # With splot_weight=2.0, loss on all-sPlot data should differ
        assert not torch.isclose(loss_equal, loss_weighted)

    def test_no_source_weighting_when_gbif_weight_none(self):
        log_pred, target, mask, _ = self._make_inputs(B=4, T=2)

        source_gbif = torch.zeros(4, dtype=torch.int8)
        source_splot = torch.ones(4, dtype=torch.int8)

        loss_fn = MaskedKLDivLoss(gbif_weight=None)
        loss_gbif = loss_fn(log_pred, target, mask, source_gbif)
        loss_splot = loss_fn(log_pred, target, mask, source_splot)

        # Without source weighting, both should be equal
        assert torch.isclose(loss_gbif, loss_splot)


# ===========================================================================
# Dataset tests
# ===========================================================================


class TestHistogramDataset:
    """Tests for HistogramDataset."""

    def test_length_all(self, synthetic_data):
        ds = HistogramDataset(
            synthetic_data["X"],
            synthetic_data["Y_hist"],
            synthetic_data["Y_mask"],
            synthetic_data["source"],
        )
        assert len(ds) == len(synthetic_data["X"])

    def test_length_subset(self, synthetic_data):
        indices = np.array([0, 5, 10, 20])
        ds = HistogramDataset(
            synthetic_data["X"],
            synthetic_data["Y_hist"],
            synthetic_data["Y_mask"],
            synthetic_data["source"],
            indices=indices,
        )
        assert len(ds) == 4

    def test_getitem_types(self, synthetic_data):
        ds = HistogramDataset(
            synthetic_data["X"],
            synthetic_data["Y_hist"],
            synthetic_data["Y_mask"],
            synthetic_data["source"],
        )
        x, y, m, s = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert isinstance(m, torch.Tensor)
        assert isinstance(s, torch.Tensor)

    def test_getitem_shapes(self, synthetic_data, dims):
        ds = HistogramDataset(
            synthetic_data["X"],
            synthetic_data["Y_hist"],
            synthetic_data["Y_mask"],
            synthetic_data["source"],
        )
        x, y, m, s = ds[0]
        assert x.shape == (dims["F"],)
        assert y.shape == (dims["T"], dims["B"])
        assert m.shape == (dims["T"],)
        assert s.shape == ()


class TestPreprocessFeatures:
    """Tests for feature preprocessing."""

    def test_no_nan_after_preprocessing(self, rng):
        X = rng.randn(100, 5).astype(np.float32)
        X[10, 2] = np.nan
        X[20, 4] = 32767.0
        X[30, 1] = -32767.0

        train_mask = np.ones(100, dtype=bool)
        X_clean, stats = preprocess_features(X, train_mask)

        assert not np.isnan(X_clean).any()
        assert "median" in stats

    def test_standardization(self, rng):
        X = rng.randn(200, 5).astype(np.float32) * 10 + 50
        train_mask = np.ones(200, dtype=bool)

        X_clean, stats = preprocess_features(X, train_mask, standardize=True)

        assert "mean" in stats
        assert "std" in stats
        # Standardized training data should have ~0 mean, ~1 std
        assert np.abs(X_clean.mean(axis=0)).max() < 0.5
        assert np.abs(X_clean.std(axis=0) - 1.0).max() < 0.5

    def test_no_standardization(self, rng):
        X = rng.randn(100, 5).astype(np.float32) * 10 + 50
        train_mask = np.ones(100, dtype=bool)

        X_clean, stats = preprocess_features(X, train_mask, standardize=False)

        assert "mean" not in stats
        assert "std" not in stats

    def test_train_mask_used_for_stats(self, rng):
        X = rng.randn(100, 3).astype(np.float32)
        X[50:, :] = 100.0  # Val data has large values

        train_mask = np.zeros(100, dtype=bool)
        train_mask[:50] = True

        X_clean, stats = preprocess_features(X, train_mask, standardize=True)

        # Stats should be based on train split only (rows 0-49)
        assert stats["mean"].max() < 5.0  # Not influenced by the 100s


# ===========================================================================
# CV splits tests
# ===========================================================================


class TestCVSplits:
    """Tests for spatial fold assignment."""

    def test_get_train_val_indices_basic(self):
        N = 100
        folds = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20)
        source = np.array([0, 1] * 50, dtype=np.int8)  # alternating GBIF/sPlot

        train_idx, val_idx = get_train_val_indices(folds, source, fold_id=0)

        # Training: all cells NOT in fold 0
        assert len(train_idx) == 80
        assert not np.isin(train_idx, np.where(folds == 0)[0]).any()

        # Validation: only sPlot cells IN fold 0
        for i in val_idx:
            assert folds[i] == 0
            assert source[i] == 1

    def test_get_train_val_no_overlap(self):
        folds = np.array([0] * 10 + [1] * 10)
        source = np.ones(20, dtype=np.int8)  # all sPlot

        train_idx, val_idx = get_train_val_indices(folds, source, fold_id=0)

        # No overlap between train and val
        assert len(np.intersect1d(train_idx, val_idx)) == 0

    def test_gbif_excluded_from_val(self):
        folds = np.array([0] * 20)
        source = np.zeros(20, dtype=np.int8)  # all GBIF

        _, val_idx = get_train_val_indices(folds, source, fold_id=0)

        # No GBIF cells in validation
        assert len(val_idx) == 0

    def test_assign_spatial_folds_shape(self):
        """Test with mock H3/pyproj to avoid coordinate dependency."""
        N = 50
        coords = np.column_stack([
            np.random.randn(N) * 1e6,
            np.random.randn(N) * 1e6,
        ])

        with patch("src.models.histogram_mlp.cv_splits.pyproj") as mock_pyproj, \
             patch("src.models.histogram_mlp.cv_splits.h3") as mock_h3:

            # Mock coordinate transformation
            mock_transformer = mock_pyproj.Transformer.from_crs.return_value
            mock_transformer.transform.return_value = (
                np.random.uniform(-180, 180, N),
                np.random.uniform(-90, 90, N),
            )

            # Mock H3 to return hex IDs (10 unique hexagons)
            hex_ids = [f"hex_{i % 10}" for i in range(N)]
            mock_h3.latlng_to_cell.side_effect = hex_ids

            folds = assign_spatial_folds(coords, n_folds=5, n_iterations=10)

        assert folds.shape == (N,)
        assert set(folds).issubset(set(range(5)))
        # All cells assigned
        assert len(folds) == N


# ===========================================================================
# Evaluation metric tests
# ===========================================================================


class TestEvaluationMetrics:
    """Tests for evaluation metrics."""

    @pytest.fixture()
    def eval_data(self, rng, dims):
        """Evaluation test data with known properties."""
        N, T, B = dims["N"], dims["T"], dims["B"]

        pred = rng.dirichlet(np.ones(B), size=(N, T)).astype(np.float64)
        target = rng.dirichlet(np.ones(B), size=(N, T)).astype(np.float64)
        mask = np.ones((N, T), dtype=np.float32)
        bin_edges = np.tile(
            np.linspace(0, 1, B + 1), (T, 1)
        )
        return pred, target, mask, bin_edges

    def test_kl_divergence_nonnegative(self, eval_data):
        pred, target, mask, _ = eval_data
        result = compute_kl_divergence(pred, target, mask)
        assert result["overall"] >= 0
        for v in result["per_trait"]:
            assert v >= 0

    def test_kl_divergence_zero_for_identical(self, dims):
        N, T, B = dims["N"], dims["T"], dims["B"]
        probs = np.ones((N, T, B)) / B
        mask = np.ones((N, T))

        result = compute_kl_divergence(probs, probs, mask)
        assert result["overall"] < 1e-6

    def test_emd_nonnegative(self, eval_data):
        pred, target, mask, bin_edges = eval_data
        result = compute_emd(pred, target, mask, bin_edges)
        assert result["overall"] >= 0
        for v in result["per_trait"]:
            assert v >= 0

    def test_emd_zero_for_identical(self, dims):
        N, T, B = dims["N"], dims["T"], dims["B"]
        probs = np.ones((N, T, B)) / B
        mask = np.ones((N, T))
        bin_edges = np.tile(np.linspace(0, 1, B + 1), (T, 1))

        result = compute_emd(probs, probs, mask, bin_edges)
        assert result["overall"] < 1e-10

    def test_histogram_intersection_range(self, eval_data):
        pred, target, mask, _ = eval_data
        result = compute_histogram_intersection(pred, target, mask)
        assert 0.0 <= result["overall"] <= 1.0
        for v in result["per_trait"]:
            assert 0.0 <= v <= 1.0

    def test_histogram_intersection_identical(self, dims):
        N, T, B = dims["N"], dims["T"], dims["B"]
        probs = np.ones((N, T, B)) / B
        mask = np.ones((N, T))

        result = compute_histogram_intersection(probs, probs, mask)
        assert abs(result["overall"] - 1.0) < 1e-6

    def test_moment_comparison_keys(self, eval_data):
        pred, target, mask, bin_edges = eval_data
        result = compute_moment_comparison(pred, target, mask, bin_edges)
        assert "mean_r2" in result
        assert "mean_mae" in result
        assert "overall" in result["mean_r2"]
        assert "per_trait" in result["mean_r2"]

    def test_moment_comparison_identical_r2(self, dims):
        N, T, B = dims["N"], dims["T"], dims["B"]
        rng = np.random.RandomState(0)
        probs = rng.dirichlet(np.ones(B), size=(N, T))
        mask = np.ones((N, T))
        bin_edges = np.tile(np.linspace(0, 1, B + 1), (T, 1))

        result = compute_moment_comparison(probs, probs, mask, bin_edges)
        # Perfect prediction → R² ≈ 1, MAE ≈ 0
        assert result["mean_r2"]["overall"] > 0.99
        assert result["mean_mae"]["overall"] < 1e-10

    def test_evaluate_all_keys(self, eval_data):
        pred, target, mask, bin_edges = eval_data
        result = evaluate_all(pred, target, mask, bin_edges)
        assert "kl_divergence" in result
        assert "emd" in result
        assert "histogram_intersection" in result
        assert "moment_comparison" in result

    def test_mask_respected(self, dims):
        """Metrics should handle fully masked-out traits gracefully."""
        N, T, B = dims["N"], dims["T"], dims["B"]
        rng = np.random.RandomState(0)

        pred = rng.dirichlet(np.ones(B), size=(N, T))
        target = rng.dirichlet(np.ones(B), size=(N, T))

        # Mask out trait 0 entirely
        mask = np.ones((N, T))
        mask[:, 0] = 0

        result_kl = compute_kl_divergence(pred, target, mask)
        # Trait 0 should be NaN
        assert np.isnan(result_kl["per_trait"][0])
        # Other traits should be valid
        for v in result_kl["per_trait"][1:]:
            assert not np.isnan(v)


# ===========================================================================
# Integration test: model training step
# ===========================================================================


class TestTrainingIntegration:
    """Integration test for a single training step."""

    def test_single_training_step(self, synthetic_data, dims):
        """Verify one forward + backward pass works end-to-end."""
        F, T, B = dims["F"], dims["T"], dims["B"]

        model = HistogramMLP(n_features=F, n_traits=T, n_bins=B, hidden_dims=[32, 16])
        criterion = MaskedKLDivLoss(gbif_weight=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Batch from dataset
        ds = HistogramDataset(
            synthetic_data["X"],
            synthetic_data["Y_hist"],
            synthetic_data["Y_mask"],
            synthetic_data["source"],
        )
        x, y, m, s = ds[0]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        m = m.unsqueeze(0)
        s = s.unsqueeze(0)

        model.train()
        optimizer.zero_grad()
        log_pred = model(x)
        loss = criterion(log_pred, y, m, s)
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_loss_decreases_over_steps(self, synthetic_data, dims):
        """Verify loss trends downward over a few training steps."""
        F, T, B = dims["F"], dims["T"], dims["B"]

        model = HistogramMLP(
            n_features=F, n_traits=T, n_bins=B,
            hidden_dims=[32, 16], dropout=0.0,
        )
        criterion = MaskedKLDivLoss(gbif_weight=None)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        ds = HistogramDataset(
            synthetic_data["X"],
            synthetic_data["Y_hist"],
            synthetic_data["Y_mask"],
            synthetic_data["source"],
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

        losses = []
        for _ in range(20):
            epoch_loss = 0.0
            for x, y, m, s in loader:
                optimizer.zero_grad()
                log_pred = model(x)
                loss = criterion(log_pred, y, m, s)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)

        # Loss at end should be lower than start (allowing some noise)
        assert losses[-1] < losses[0]
