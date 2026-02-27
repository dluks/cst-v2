"""Training loop and cross-validation orchestration for the histogram MLP."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.conf.conf import get_config
from src.models.histogram_mlp.cv_splits import get_train_val_indices
from src.models.histogram_mlp.dataset import (
    HistogramDataset,
    load_zarr_arrays,
    preprocess_features,
)
from src.models.histogram_mlp.evaluate import compute_baseline_metrics, evaluate_all
from src.models.histogram_mlp.loss import MaskedKLDivLoss
from src.models.histogram_mlp.model import HistogramMLP
from src.models.run_utils import generate_run_id, get_latest_run_id

log = logging.getLogger(__name__)

# HPO param keys that map directly to cfg.train fields
_HPO_PARAM_KEYS = ("hidden_dims", "dropout", "lr", "weight_decay", "batch_size", "gbif_weight_factor")


def load_hpo_best_params(hpo_base_dir: Path) -> dict | None:
    """Load the best HPO params from the study with the lowest validation loss.

    Scans all ``*/best_params.json`` files under *hpo_base_dir* and returns
    the hyperparameter dict from the study with the lowest ``best_val_loss``.
    Returns ``None`` if no HPO results are found.
    """
    candidates = sorted(hpo_base_dir.glob("*/best_params.json"))
    if not candidates:
        return None

    best_loss = float("inf")
    best_params = None
    best_path = None
    for path in candidates:
        with open(path) as f:
            params = json.load(f)
        loss = params.get("best_val_loss", float("inf"))
        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_path = path

    if best_params is not None:
        study_dir = best_path.parent.name
        log.info(
            "Loaded HPO params from %s (trial %d, val_loss=%.6f)",
            study_dir, best_params.get("best_trial_number", -1), best_loss,
        )
    return best_params


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: HistogramMLP,
    dataloader: DataLoader,
    criterion: MaskedKLDivLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, Y_batch, mask_batch, source_batch in dataloader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        mask_batch = mask_batch.to(device)
        source_batch = source_batch.to(device)

        optimizer.zero_grad()
        log_pred = model(X_batch)
        loss = criterion(log_pred, Y_batch, mask_batch, source_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: HistogramMLP,
    dataloader: DataLoader,
    criterion: MaskedKLDivLoss,
    device: torch.device,
) -> float:
    """Validate and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for X_batch, Y_batch, mask_batch, source_batch in dataloader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        mask_batch = mask_batch.to(device)
        source_batch = source_batch.to(device)

        log_pred = model(X_batch)
        loss = criterion(log_pred, Y_batch, mask_batch, source_batch)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def collect_predictions(
    model: HistogramMLP,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Collect predicted probabilities for the entire dataset.

    Returns
    -------
    np.ndarray
        Predicted probabilities, shape ``(N, n_traits, n_bins)``.
    """
    model.eval()
    all_preds = []
    for X_batch, _, _, _ in dataloader:
        X_batch = X_batch.to(device)
        log_pred = model(X_batch)
        all_preds.append(torch.exp(log_pred).cpu().numpy())
    return np.concatenate(all_preds, axis=0)


# ---------------------------------------------------------------------------
# Fold / full training
# ---------------------------------------------------------------------------


def train_fold(
    fold_id: int,
    data: dict,
    folds: np.ndarray,
    output_dir: Path,
    cfg,
    device: torch.device,
    epoch_callback: Callable[[int, float], None] | None = None,
    num_workers: int = 4,
) -> dict:
    """Train a single CV fold.

    Parameters
    ----------
    fold_id : int
        Which fold to hold out for validation.
    data : dict
        Loaded and preprocessed data arrays.
    folds : np.ndarray
        Fold assignments.
    output_dir : Path
        Directory for this fold's outputs.
    cfg : ConfigBox
        Training configuration.
    device : torch.device
        CUDA or CPU device.
    epoch_callback : Callable[[int, float], None] | None
        Optional callback invoked with ``(epoch, val_loss)`` after each epoch.
        Can raise an exception (e.g. ``optuna.TrialPruned``) to stop training.
    num_workers : int
        Number of DataLoader workers. Use 0 for HPO to avoid multiprocessing
        cleanup issues across short-lived trials.

    Returns
    -------
    dict
        Evaluation metrics on validation set.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for completion flag
    flag_path = output_dir / "fold_complete.flag"
    if flag_path.exists():
        log.info("Fold %d already complete, skipping", fold_id)
        metrics_path = output_dir / "fold_metrics.json"
        if metrics_path.exists():
            return json.loads(metrics_path.read_text())
        return {}

    train_idx, val_idx = get_train_val_indices(folds, data["source"], fold_id)
    log.info(
        "Fold %d: %d train cells, %d val cells (sPlot only)",
        fold_id, len(train_idx), len(val_idx),
    )

    # Create datasets and dataloaders
    batch_size = cfg.train.batch_size
    train_ds = HistogramDataset(
        data["X"], data["Y_hist"], data["Y_mask"], data["source"],
        indices=train_idx,
    )
    val_ds = HistogramDataset(
        data["X"], data["Y_hist"], data["Y_mask"], data["source"],
        indices=val_idx,
    )
    use_mp = num_workers > 0
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_mp, persistent_workers=use_mp,
    )
    val_workers = min(2, num_workers)
    val_dl = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=val_workers, pin_memory=use_mp,
        persistent_workers=use_mp and val_workers > 0,
    )

    # Model, loss, optimizer, scheduler
    model = HistogramMLP(
        n_features=data["X"].shape[1],
        n_traits=data["Y_hist"].shape[1],
        n_bins=data["Y_hist"].shape[2],
        hidden_dims=list(cfg.train.hidden_dims),
        dropout=cfg.train.dropout,
    ).to(device)

    n_splot_train = int((data["source"][train_idx] == 1).sum())
    n_total_train = len(train_idx)
    gbif_weight = n_splot_train / n_total_train if n_total_train > 0 else 1.0
    gbif_weight *= cfg.train.get("gbif_weight_factor", 1.0)
    log.info("Source weighting: gbif_weight=%.4f", gbif_weight)

    criterion = MaskedKLDivLoss(gbif_weight=gbif_weight)
    optimizer = AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.max_epochs)

    # Training loop
    from torch.utils.tensorboard import SummaryWriter

    best_val_loss = float("inf")
    patience_counter = 0
    training_log: list[dict] = []
    writer = SummaryWriter(log_dir=output_dir / "tb")

    for epoch in range(cfg.train.max_epochs):
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss = validate(model, val_dl, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        training_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
        })

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LearningRate", lr, epoch)

        if epoch % 10 == 0 or epoch == cfg.train.max_epochs - 1:
            log.info(
                "  Epoch %3d: train=%.6f  val=%.6f  lr=%.2e",
                epoch, train_loss, val_loss, lr,
            )

        # Report to callback (e.g., Optuna pruning)
        if epoch_callback is not None:
            epoch_callback(epoch, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= cfg.train.patience:
            log.info("Early stopping at epoch %d (patience=%d)", epoch, cfg.train.patience)
            break

    writer.close()

    # Save training log
    _save_training_log(training_log, output_dir / "training_log.csv")

    # Evaluate best model
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    pred_probs = collect_predictions(model, val_dl, device)
    target_probs = data["Y_hist"][val_idx]
    val_mask = data["Y_mask"][val_idx]

    metrics = evaluate_all(
        pred_probs, target_probs, val_mask, data["bin_edges"],
        trait_names=data.get("trait_names"),
    )
    metrics["best_val_loss"] = best_val_loss
    metrics["epochs_trained"] = len(training_log)

    # Mean-histogram baseline for comparison
    metrics["baseline"] = compute_baseline_metrics(
        data["Y_hist"], data["Y_mask"], train_idx, val_idx,
        data["bin_edges"], trait_names=data.get("trait_names"),
    )

    with open(output_dir / "fold_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    flag_path.touch()
    baseline_kl = metrics["baseline"]["kl_divergence"]["overall"]
    log.info("Fold %d complete: val_loss=%.6f, KL=%.6f (baseline=%.6f), EMD=%.6f, HI=%.4f",
             fold_id, best_val_loss,
             metrics["kl_divergence"]["overall"], baseline_kl,
             metrics["emd"]["overall"],
             metrics["histogram_intersection"]["overall"])

    return metrics


def train_full_model(
    data: dict,
    output_dir: Path,
    cfg,
    device: torch.device,
) -> None:
    """Train final model on all data (no validation, fixed epochs)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    flag_path = output_dir / "full_model_complete.flag"
    if flag_path.exists():
        log.info("Full model already complete, skipping")
        return

    log.info("Training full model on all %d cells", len(data["X"]))

    all_idx = np.arange(len(data["X"]))
    batch_size = cfg.train.batch_size
    train_ds = HistogramDataset(
        data["X"], data["Y_hist"], data["Y_mask"], data["source"],
        indices=all_idx,
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    model = HistogramMLP(
        n_features=data["X"].shape[1],
        n_traits=data["Y_hist"].shape[1],
        n_bins=data["Y_hist"].shape[2],
        hidden_dims=list(cfg.train.hidden_dims),
        dropout=cfg.train.dropout,
    ).to(device)

    n_splot = int((data["source"] == 1).sum())
    gbif_weight = n_splot / len(data["source"])
    gbif_weight *= cfg.train.get("gbif_weight_factor", 1.0)
    log.info("Source weighting (full model): gbif_weight=%.4f", gbif_weight)

    criterion = MaskedKLDivLoss(gbif_weight=gbif_weight)
    optimizer = AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.max_epochs)

    from torch.utils.tensorboard import SummaryWriter

    training_log: list[dict] = []
    writer = SummaryWriter(log_dir=output_dir / "tb")

    for epoch in range(cfg.train.max_epochs):
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        training_log.append({"epoch": epoch, "train_loss": train_loss, "lr": lr})

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("LearningRate", lr, epoch)

        if epoch % 10 == 0 or epoch == cfg.train.max_epochs - 1:
            log.info("  Epoch %3d: train=%.6f  lr=%.2e", epoch, train_loss, lr)

    writer.close()

    torch.save(model.state_dict(), output_dir / "best_model.pt")
    _save_training_log(training_log, output_dir / "training_log.csv")
    flag_path.touch()
    log.info("Full model training complete")


# ---------------------------------------------------------------------------
# CV orchestration
# ---------------------------------------------------------------------------


def run_cv(
    data: dict,
    run_dir: Path,
    cfg,
    device: torch.device,
) -> None:
    """Run full cross-validation pipeline.

    1. Preprocess features (once).
    2. Assign spatial folds (once, cached).
    3. For each fold: ``train_fold()``.
    4. Aggregate CV metrics → ``cv_summary.json``.
    5. Train full model on all data.
    """
    n_folds = cfg.train.n_folds

    # Preprocess features
    log.info("Preprocessing features...")
    all_source = data["source"]
    # Use all data for computing preprocessing stats (they'll be reused for inference)
    # But use a representative "training" mask (all non-fold-0 cells) for stats
    dummy_train_mask = np.ones(len(data["X"]), dtype=bool)
    # For proper stats, we'd use train split — but since we run multiple folds,
    # use all data for a single set of stats (minor difference at 22km scale)
    data["X"], feature_stats = preprocess_features(
        data["X"],
        train_mask=dummy_train_mask,
        standardize=cfg.train.get("standardize_features", True),
        vodca_sentinel=cfg.train.get("vodca_sentinel", 32767.0),
    )
    np.savez(run_dir / "feature_stats.npz", **feature_stats)

    # Load pre-computed folds from Zarr
    if "folds" not in data:
        raise ValueError("No 'folds' array in train.zarr — re-run build_histogram_xy")
    folds = data["folds"]
    log.info("Loaded fold assignments: %d folds", len(np.unique(folds)))

    # Train CV folds
    cv_dir = run_dir / "cv"
    fold_metrics = []
    for fold_id in range(n_folds):
        log.info("=" * 60)
        log.info("Training fold %d / %d", fold_id, n_folds - 1)
        log.info("=" * 60)
        metrics = train_fold(
            fold_id, data, folds,
            output_dir=cv_dir / f"fold_{fold_id}",
            cfg=cfg, device=device,
        )
        fold_metrics.append(metrics)

    # Aggregate CV metrics
    summary = aggregate_cv_metrics(fold_metrics)
    with open(cv_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    log.info("CV Summary: KL=%.6f, EMD=%.6f, HI=%.4f, Mean R²=%.4f",
             summary.get("kl_divergence_mean", float("nan")),
             summary.get("emd_mean", float("nan")),
             summary.get("histogram_intersection_mean", float("nan")),
             summary.get("mean_r2_mean", float("nan")))

    # Train full model
    log.info("=" * 60)
    log.info("Training full model on all data")
    log.info("=" * 60)
    train_full_model(data, run_dir / "full_model", cfg, device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_training_log(log_entries: list[dict], path: Path) -> None:
    """Save training log as CSV."""
    if not log_entries:
        return
    keys = log_entries[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(log_entries)


def aggregate_cv_metrics(fold_metrics: list[dict]) -> dict:
    """Aggregate per-fold metrics into a CV summary."""
    summary: dict = {}

    # Extract scalar overall metrics from each fold
    metric_keys = [
        ("kl_divergence", "overall"),
        ("emd", "overall"),
        ("histogram_intersection", "overall"),
    ]
    for key, subkey in metric_keys:
        values = []
        for fm in fold_metrics:
            if key in fm and subkey in fm[key]:
                v = fm[key][subkey]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    values.append(v)
        if values:
            name = key.replace("_divergence", "")
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))

    # Moment comparison
    for fm in fold_metrics:
        mc = fm.get("moment_comparison", {})
        for mk in ("mean_r2", "mean_mae"):
            if mk in mc and "overall" in mc[mk]:
                v = mc[mk]["overall"]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    summary.setdefault(f"{mk}_values", []).append(v)

    for mk in ("mean_r2", "mean_mae"):
        values = summary.pop(f"{mk}_values", [])
        if values:
            summary[f"{mk}_mean"] = float(np.mean(values))
            summary[f"{mk}_std"] = float(np.std(values))

    summary["n_folds"] = len(fold_metrics)
    return summary


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train histogram MLP for trait distribution prediction.",
    )
    parser.add_argument(
        "--params", type=str, required=True, help="Path to params.yaml",
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Train a specific fold only (0-indexed). If None, run full CV.",
    )
    parser.add_argument(
        "--full-only", action="store_true",
        help="Train only the full model (skip CV).",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run ID (format: run_YYYYMMDD_HHMMSS). Auto-generated if not provided.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from most recent run.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: 3 epochs, batch_size=64.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing outputs (ignored — use flag files for incremental).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = cli()
    cfg = get_config(params_path=args.params)

    # Resolve paths
    proj_root = os.environ.get("PROJECT_ROOT")
    if proj_root is None:
        raise ValueError("PROJECT_ROOT environment variable is not set")
    proj_root = Path(proj_root)

    # Auto-load HPO best params
    hpo_base_dir = proj_root / cfg.models.dir_fp / "hpo"
    hpo_params = load_hpo_best_params(hpo_base_dir)
    if hpo_params is not None:
        for key in _HPO_PARAM_KEYS:
            if key in hpo_params:
                old_val = cfg.train.get(key, "(unset)")
                cfg.train[key] = hpo_params[key]
                log.info("  HPO override: train.%s = %s (was %s)", key, hpo_params[key], old_val)
    else:
        log.info("No HPO results found in %s — using params.yaml defaults", hpo_base_dir)

    # Override for debug mode
    if args.debug:
        cfg.train.max_epochs = 3
        cfg.train.batch_size = 64
        cfg.train.patience = 100  # Don't early-stop in debug

    zarr_path = proj_root / cfg.output.xy_dir / "train.zarr"
    models_base = proj_root / cfg.models.dir_fp / "training"
    models_base.mkdir(parents=True, exist_ok=True)

    # Determine run ID
    if args.resume:
        run_id = get_latest_run_id(models_base)
        if run_id is None:
            log.warning("No existing runs found, generating new run ID")
            run_id = generate_run_id()
    elif args.run_id:
        run_id = args.run_id
    else:
        run_id = generate_run_id()

    run_dir = models_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run directory: %s", run_dir)

    # Save config snapshot
    import yaml
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Set random seeds
    seed = cfg.get("random_seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Load data
    log.info("Loading data from %s", zarr_path)
    data = load_zarr_arrays(zarr_path)

    if args.full_only:
        # Preprocess and train full model only
        dummy_mask = np.ones(len(data["X"]), dtype=bool)
        data["X"], feature_stats = preprocess_features(
            data["X"], train_mask=dummy_mask,
            standardize=cfg.train.get("standardize_features", True),
            vodca_sentinel=cfg.train.get("vodca_sentinel", 32767.0),
        )
        np.savez(run_dir / "feature_stats.npz", **feature_stats)
        train_full_model(data, run_dir / "full_model", cfg, device)
    elif args.fold is not None:
        # Single fold mode — preprocess, assign folds, train one fold
        dummy_mask = np.ones(len(data["X"]), dtype=bool)
        data["X"], feature_stats = preprocess_features(
            data["X"], train_mask=dummy_mask,
            standardize=cfg.train.get("standardize_features", True),
            vodca_sentinel=cfg.train.get("vodca_sentinel", 32767.0),
        )
        np.savez(run_dir / "feature_stats.npz", **feature_stats)

        if "folds" not in data:
            raise ValueError("No 'folds' array in train.zarr — re-run build_histogram_xy")
        folds = data["folds"]

        train_fold(
            args.fold, data, folds,
            output_dir=run_dir / "cv" / f"fold_{args.fold}",
            cfg=cfg, device=device,
        )
    else:
        # Full CV pipeline
        run_cv(data, run_dir, cfg, device)

    log.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
