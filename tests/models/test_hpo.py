"""Tests for Optuna HPO integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Skip entire module if torch or optuna are not available
torch = pytest.importorskip("torch")
optuna = pytest.importorskip("optuna")

from src.models.histogram_mlp.hpo import (
    create_objective,
    create_study,
    define_search_space,
    save_best_params,
)

# Fixtures (rng, dims, synthetic_data) are in conftest.py


# ===========================================================================
# Search space tests
# ===========================================================================


class TestSearchSpace:
    """Tests for HPO search space definition."""

    def test_returns_all_keys(self):
        study = optuna.create_study()
        trial = study.ask()
        hp = define_search_space(trial)

        expected_keys = {
            "hidden_dims", "dropout", "lr",
            "weight_decay", "batch_size", "gbif_weight_factor",
        }
        assert expected_keys == set(hp.keys())

    def test_hidden_dims_length_matches_n_layers(self):
        study = optuna.create_study()
        trial = study.ask()
        hp = define_search_space(trial)
        assert len(hp["hidden_dims"]) == trial.params["n_layers"]

    def test_parameter_ranges(self):
        study = optuna.create_study()
        for _ in range(20):
            trial = study.ask()
            hp = define_search_space(trial)

            assert 2 <= len(hp["hidden_dims"]) <= 4
            for dim in hp["hidden_dims"]:
                assert dim in [128, 256, 512, 768, 1024]
            assert 0.0 <= hp["dropout"] <= 0.5
            assert 1e-4 <= hp["lr"] <= 1e-2
            assert 1e-5 <= hp["weight_decay"] <= 1e-1
            assert hp["batch_size"] in [256, 512, 1024, 2048]
            assert 0.5 <= hp["gbif_weight_factor"] <= 2.0

            study.tell(trial, 0.5)


# ===========================================================================
# Study management tests
# ===========================================================================


class TestCreateStudy:
    """Tests for study creation and persistence."""

    def test_creates_storage_file(self, tmp_path):
        storage_path = tmp_path / "test_study" / "journal.log"
        study = create_study("test_study", storage_path)

        assert study.study_name == "test_study"
        assert storage_path.exists()

    def test_load_if_exists(self, tmp_path):
        storage_path = tmp_path / "journal.log"
        study1 = create_study("test", storage_path)

        # Add a trial
        study1.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)

        # Reopen — should see the existing trial
        study2 = create_study("test", storage_path)
        assert len(study2.trials) == 1


# ===========================================================================
# Save best params tests
# ===========================================================================


class TestSaveBestParams:
    """Tests for saving best parameters."""

    def test_save_creates_valid_json(self, tmp_path):
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: (
                define_search_space(trial) and trial.suggest_float("_dummy", 0, 1)
            ) or np.random.rand(),
            n_trials=5,
        )

        output_path = tmp_path / "best_params.json"
        result = save_best_params(study, output_path)

        assert output_path.exists()
        loaded = json.loads(output_path.read_text())
        assert "hidden_dims" in loaded
        assert "best_val_loss" in loaded
        assert "best_trial_number" in loaded
        assert isinstance(loaded["hidden_dims"], list)

    def test_hidden_dims_reconstructed_correctly(self, tmp_path):
        study = optuna.create_study(direction="minimize")

        def obj(trial):
            define_search_space(trial)
            return np.random.rand()

        study.optimize(obj, n_trials=3)

        output_path = tmp_path / "best.json"
        result = save_best_params(study, output_path)

        n_layers = study.best_trial.params["n_layers"]
        assert len(result["hidden_dims"]) == n_layers


# ===========================================================================
# Epoch callback integration tests
# ===========================================================================


class TestEpochCallback:
    """Tests for pruning integration via epoch_callback in train_fold."""

    def test_train_fold_accepts_callback(self, synthetic_data, dims, tmp_path):
        """Verify train_fold calls the callback each epoch."""
        from box import ConfigBox

        from src.models.histogram_mlp.train import train_fold

        callback_calls = []

        def tracking_callback(epoch, val_loss):
            callback_calls.append((epoch, val_loss))

        cfg = ConfigBox({
            "train": {
                "hidden_dims": [32, 16],
                "dropout": 0.0,
                "batch_size": 32,
                "lr": 0.01,
                "weight_decay": 0.01,
                "max_epochs": 3,
                "patience": 100,
            }
        })

        folds = np.zeros(dims["N"], dtype=np.int32)
        folds[: dims["N"] // 2] = 1

        output_dir = tmp_path / "test_callback"
        device = torch.device("cpu")

        train_fold(
            fold_id=0,
            data=synthetic_data,
            folds=folds,
            output_dir=output_dir,
            cfg=cfg,
            device=device,
            epoch_callback=tracking_callback,
        )

        assert len(callback_calls) == 3
        for epoch, val_loss in callback_calls:
            assert isinstance(epoch, int)
            assert isinstance(val_loss, float)

    def test_pruning_stops_training_early(self, synthetic_data, dims, tmp_path):
        """Verify TrialPruned exception propagates from callback."""
        from box import ConfigBox

        from src.models.histogram_mlp.train import train_fold

        def pruning_callback(epoch, val_loss):
            if epoch >= 1:
                raise optuna.TrialPruned()

        cfg = ConfigBox({
            "train": {
                "hidden_dims": [32, 16],
                "dropout": 0.0,
                "batch_size": 32,
                "lr": 0.01,
                "weight_decay": 0.01,
                "max_epochs": 100,
                "patience": 100,
            }
        })

        folds = np.zeros(dims["N"], dtype=np.int32)
        folds[: dims["N"] // 2] = 1

        output_dir = tmp_path / "test_pruning"
        device = torch.device("cpu")

        with pytest.raises(optuna.TrialPruned):
            train_fold(
                fold_id=0,
                data=synthetic_data,
                folds=folds,
                output_dir=output_dir,
                cfg=cfg,
                device=device,
                epoch_callback=pruning_callback,
            )

    def test_no_callback_backward_compatible(self, synthetic_data, dims, tmp_path):
        """train_fold still works without a callback."""
        from box import ConfigBox

        from src.models.histogram_mlp.train import train_fold

        cfg = ConfigBox({
            "train": {
                "hidden_dims": [32, 16],
                "dropout": 0.0,
                "batch_size": 32,
                "lr": 0.01,
                "weight_decay": 0.01,
                "max_epochs": 2,
                "patience": 100,
            }
        })

        folds = np.zeros(dims["N"], dtype=np.int32)
        folds[: dims["N"] // 2] = 1

        output_dir = tmp_path / "test_no_callback"
        device = torch.device("cpu")

        metrics = train_fold(
            fold_id=0,
            data=synthetic_data,
            folds=folds,
            output_dir=output_dir,
            cfg=cfg,
            device=device,
        )

        assert "best_val_loss" in metrics
