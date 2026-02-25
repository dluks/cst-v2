"""Optuna hyperparameter optimization for the histogram MLP."""

from __future__ import annotations

import argparse
import json
import logging
import os
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.conf.conf import get_config
from src.models.histogram_mlp.cv_splits import assign_spatial_folds
from src.models.histogram_mlp.dataset import load_zarr_arrays, preprocess_features
from src.models.histogram_mlp.train import train_fold

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


def define_search_space(trial) -> dict:
    """Define the Optuna search space.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial for suggesting hyperparameters.

    Returns
    -------
    dict
        Hyperparameter overrides for ``cfg.train``.
    """
    # Architecture
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_dims = [
        trial.suggest_categorical(f"hidden_dim_{i}", [128, 256, 512, 768, 1024])
        for i in range(n_layers)
    ]

    # Regularization
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)

    # Optimization
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)

    # Data
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])

    # Loss weighting
    gbif_weight_factor = trial.suggest_float("gbif_weight_factor", 0.5, 2.0)

    return {
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "gbif_weight_factor": gbif_weight_factor,
    }


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def create_objective(
    data: dict,
    folds: np.ndarray,
    cfg,
    device,
    hpo_dir: Path,
    hpo_fold: int = 0,
):
    """Create an Optuna objective function wrapping ``train_fold``.

    Parameters
    ----------
    data : dict
        Preprocessed data arrays.
    folds : np.ndarray
        Fold assignments.
    cfg : ConfigBox
        Base configuration (deep-copied per trial).
    device : torch.device
        CUDA or CPU device.
    hpo_dir : Path
        Base directory for HPO outputs.
    hpo_fold : int
        Fold to use for validation (default: 0).

    Returns
    -------
    Callable
        Objective function for ``study.optimize()``.
    """
    import optuna

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        hp = define_search_space(trial)

        # Override config for this trial
        trial_cfg = deepcopy(cfg)
        trial_cfg.train.hidden_dims = hp["hidden_dims"]
        trial_cfg.train.dropout = hp["dropout"]
        trial_cfg.train.lr = hp["lr"]
        trial_cfg.train.weight_decay = hp["weight_decay"]
        trial_cfg.train.batch_size = hp["batch_size"]
        trial_cfg.train.gbif_weight_factor = hp["gbif_weight_factor"]

        # Reduce patience for HPO (faster iteration)
        trial_cfg.train.patience = min(cfg.train.patience, 15)

        trial_dir = hpo_dir / "trials" / f"trial_{trial.number:04d}"

        # Pruning callback
        def epoch_callback(epoch: int, val_loss: float) -> None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            metrics = train_fold(
                fold_id=hpo_fold,
                data=data,
                folds=folds,
                output_dir=trial_dir,
                cfg=trial_cfg,
                device=device,
                epoch_callback=epoch_callback,
            )
            val_loss = metrics["best_val_loss"]
        except optuna.TrialPruned:
            log.info("Trial %d pruned", trial.number)
            raise
        except Exception as e:
            log.error("Trial %d failed: %s", trial.number, e)
            raise optuna.TrialPruned() from e

        log.info("Trial %d: val_loss=%.6f", trial.number, val_loss)
        return val_loss

    return objective


# ---------------------------------------------------------------------------
# Study management
# ---------------------------------------------------------------------------


def create_study(study_name: str, storage_path: Path):
    """Create or load an Optuna study with journal file storage.

    Parameters
    ----------
    study_name : str
        Name of the study.
    storage_path : Path
        Path to the journal log file.

    Returns
    -------
    optuna.Study
        Optuna study object.
    """
    import optuna
    from optuna.storages import JournalFileStorage, JournalStorage

    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileStorage(str(storage_path)))

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1,
        ),
        load_if_exists=True,
    )
    return study


def save_best_params(study, output_path: Path) -> dict:
    """Save best trial parameters to JSON.

    Parameters
    ----------
    study : optuna.Study
        Completed study.
    output_path : Path
        Path to save the JSON file.

    Returns
    -------
    dict
        Best parameters dictionary.
    """
    best = study.best_trial
    params = best.params

    # Reconstruct hidden_dims from per-layer params
    n_layers = params["n_layers"]
    hidden_dims = [params[f"hidden_dim_{i}"] for i in range(n_layers)]

    best_config = {
        "hidden_dims": hidden_dims,
        "dropout": params["dropout"],
        "lr": params["lr"],
        "weight_decay": params["weight_decay"],
        "batch_size": params["batch_size"],
        "gbif_weight_factor": params["gbif_weight_factor"],
        "best_val_loss": best.value,
        "best_trial_number": best.number,
        "n_trials_completed": len(
            [t for t in study.trials if t.state.name == "COMPLETE"]
        ),
        "n_trials_total": len(study.trials),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_config, f, indent=2)

    log.info("Best params saved to %s", output_path)
    log.info("  val_loss=%.6f, trial=%d", best.value, best.number)
    for k, v in best_config.items():
        if k not in ("best_val_loss", "best_trial_number"):
            log.info("  %s: %s", k, v)

    return best_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli() -> argparse.Namespace:
    """Parse HPO command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optuna HPO for histogram MLP.",
    )
    parser.add_argument(
        "--params", type=str, required=True, help="Path to params.yaml",
    )
    parser.add_argument(
        "--study-name", type=str, default="histogram_mlp_hpo",
        help="Optuna study name (default: histogram_mlp_hpo).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20,
        help="Number of trials for this worker (default: 20).",
    )
    parser.add_argument(
        "--hpo-fold", type=int, default=0,
        help="Fold ID to use for HPO validation (default: 0).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: 5 epochs, reduced patience.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Ignored (present for compatibility with build_base_command).",
    )
    return parser.parse_args()


def main() -> None:
    """Main HPO entry point."""
    import torch

    args = cli()
    cfg = get_config(params_path=args.params)

    if args.debug:
        cfg.train.max_epochs = 5
        cfg.train.patience = 100

    # Resolve paths
    proj_root = os.environ.get("PROJECT_ROOT")
    if proj_root is None:
        raise ValueError("PROJECT_ROOT environment variable is not set")
    proj_root = Path(proj_root)

    zarr_path = proj_root / cfg.output.xy_dir / "train.zarr"
    models_base = proj_root / cfg.models.dir_fp / "histogram_mlp"
    hpo_dir = models_base / "hpo" / args.study_name
    hpo_dir.mkdir(parents=True, exist_ok=True)

    storage_path = hpo_dir / "journal.log"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Seed
    seed = cfg.get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load and preprocess data
    log.info("Loading data from %s", zarr_path)
    data = load_zarr_arrays(zarr_path)

    dummy_mask = np.ones(len(data["X"]), dtype=bool)
    data["X"], feature_stats = preprocess_features(
        data["X"],
        train_mask=dummy_mask,
        standardize=cfg.train.get("standardize_features", True),
        vodca_sentinel=cfg.train.get("vodca_sentinel", 32767.0),
    )

    # Compute folds (cached)
    folds_path = hpo_dir / "fold_assignments.npy"
    if folds_path.exists():
        log.info("Loading cached fold assignments from %s", folds_path)
        folds = np.load(folds_path)
    else:
        log.info("Assigning spatial folds...")
        folds = assign_spatial_folds(
            data["coords"],
            n_folds=cfg.train.n_folds,
            h3_resolution=cfg.train.get("h3_resolution", 2),
            n_iterations=cfg.train.get("n_fold_iterations", 100),
            random_seed=seed,
        )
        np.save(folds_path, folds)

    # Create study and objective
    study = create_study(args.study_name, storage_path)
    objective = create_objective(
        data, folds, cfg, device, hpo_dir, hpo_fold=args.hpo_fold,
    )

    log.info("Starting %d HPO trials (study: %s)", args.n_trials, args.study_name)
    study.optimize(objective, n_trials=args.n_trials)

    # Save best params (all workers do this; last writer wins, which is fine)
    save_best_params(study, hpo_dir / "best_params.json")

    log.info("HPO complete. %d trials finished.", len(study.trials))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
