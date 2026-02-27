"""Evaluation metrics for histogram predictions."""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def compute_kl_divergence(
    pred_probs: np.ndarray,
    target_probs: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-10,
) -> dict[str, float | list[float]]:
    """Compute per-trait and overall KL divergence.

    Parameters
    ----------
    pred_probs : np.ndarray
        Predicted probability histograms, shape ``(N, n_traits, n_bins)``.
    target_probs : np.ndarray
        Target probability histograms, shape ``(N, n_traits, n_bins)``.
    mask : np.ndarray
        Validity mask, shape ``(N, n_traits)``.
    eps : float
        Small constant for numerical stability in log.

    Returns
    -------
    dict
        ``{'overall': float, 'per_trait': list[float]}``
    """
    n_traits = pred_probs.shape[1]
    pred_safe = np.clip(pred_probs, eps, None)
    target_safe = np.clip(target_probs, eps, None)

    # KL(target || pred) per cell per trait
    kl = (target_safe * np.log(target_safe / pred_safe)).sum(axis=-1)  # (N, n_traits)

    per_trait = []
    for j in range(n_traits):
        valid = mask[:, j].astype(bool)
        if valid.any():
            per_trait.append(float(kl[valid, j].mean()))
        else:
            per_trait.append(float("nan"))

    overall = float(kl[mask.astype(bool)].mean()) if mask.any() else float("nan")
    return {"overall": overall, "per_trait": per_trait}


def compute_emd(
    pred_probs: np.ndarray,
    target_probs: np.ndarray,
    mask: np.ndarray,
    bin_edges: np.ndarray,
) -> dict[str, float | list[float]]:
    """Compute per-trait and overall Earth Mover's Distance.

    Uses ``scipy.stats.wasserstein_distance`` with bin centers.

    Parameters
    ----------
    pred_probs : np.ndarray
        Shape ``(N, n_traits, n_bins)``.
    target_probs : np.ndarray
        Shape ``(N, n_traits, n_bins)``.
    mask : np.ndarray
        Shape ``(N, n_traits)``.
    bin_edges : np.ndarray
        Shape ``(n_traits, n_bins + 1)``.

    Returns
    -------
    dict
        ``{'overall': float, 'per_trait': list[float]}``
    """
    n_traits = pred_probs.shape[1]
    bin_centers = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2  # (n_traits, n_bins)

    per_trait = []
    all_emds = []
    for j in range(n_traits):
        valid = mask[:, j].astype(bool)
        if not valid.any():
            per_trait.append(float("nan"))
            continue
        trait_emds = []
        centers_j = bin_centers[j]
        for i in np.where(valid)[0]:
            emd = wasserstein_distance(
                centers_j, centers_j,
                u_weights=pred_probs[i, j],
                v_weights=target_probs[i, j],
            )
            trait_emds.append(emd)
        per_trait.append(float(np.mean(trait_emds)))
        all_emds.extend(trait_emds)

    overall = float(np.mean(all_emds)) if all_emds else float("nan")
    return {"overall": overall, "per_trait": per_trait}


def compute_histogram_intersection(
    pred_probs: np.ndarray,
    target_probs: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | list[float]]:
    """Compute histogram intersection (overlap) metric.

    Intersection = ``sum(min(pred, target))`` per (cell, trait).
    Perfect match = 1.0, no overlap = 0.0.

    Parameters
    ----------
    pred_probs : np.ndarray
        Shape ``(N, n_traits, n_bins)``.
    target_probs : np.ndarray
        Shape ``(N, n_traits, n_bins)``.
    mask : np.ndarray
        Shape ``(N, n_traits)``.

    Returns
    -------
    dict
        ``{'overall': float, 'per_trait': list[float]}``
    """
    n_traits = pred_probs.shape[1]
    intersection = np.minimum(pred_probs, target_probs).sum(axis=-1)  # (N, n_traits)

    per_trait = []
    for j in range(n_traits):
        valid = mask[:, j].astype(bool)
        if valid.any():
            per_trait.append(float(intersection[valid, j].mean()))
        else:
            per_trait.append(float("nan"))

    overall = (
        float(intersection[mask.astype(bool)].mean()) if mask.any() else float("nan")
    )
    return {"overall": overall, "per_trait": per_trait}


def compute_moment_comparison(
    pred_probs: np.ndarray,
    target_probs: np.ndarray,
    mask: np.ndarray,
    bin_edges: np.ndarray,
) -> dict[str, dict[str, float | list[float]]]:
    """Compare derived moments (mean, variance) between predicted and target.

    Computes bin centers, then derives mean and variance for each distribution.

    Parameters
    ----------
    pred_probs : np.ndarray
        Shape ``(N, n_traits, n_bins)``.
    target_probs : np.ndarray
        Shape ``(N, n_traits, n_bins)``.
    mask : np.ndarray
        Shape ``(N, n_traits)``.
    bin_edges : np.ndarray
        Shape ``(n_traits, n_bins + 1)``.

    Returns
    -------
    dict
        ``{'mean_r2': {'overall': float, 'per_trait': list}, ...}``
    """
    n_traits = pred_probs.shape[1]
    bin_centers = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2  # (n_traits, n_bins)

    # Broadcast bin_centers for vectorized computation: (1, n_traits, n_bins)
    centers = bin_centers[np.newaxis, :, :]

    # Means: sum(prob * center) → (N, n_traits)
    pred_means = (pred_probs * centers).sum(axis=-1)
    target_means = (target_probs * centers).sum(axis=-1)

    per_trait_mean_r2 = []
    per_trait_mean_mae = []
    for j in range(n_traits):
        valid = mask[:, j].astype(bool)
        if valid.sum() < 2:
            per_trait_mean_r2.append(float("nan"))
            per_trait_mean_mae.append(float("nan"))
            continue

        p = pred_means[valid, j]
        t = target_means[valid, j]
        mae = float(np.abs(p - t).mean())
        ss_res = float(((t - p) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        per_trait_mean_r2.append(r2)
        per_trait_mean_mae.append(mae)

    all_valid = mask.astype(bool)
    p_all = pred_means[all_valid]
    t_all = target_means[all_valid]
    overall_mae = float(np.abs(p_all - t_all).mean()) if len(p_all) > 0 else float("nan")
    ss_res = float(((t_all - p_all) ** 2).sum())
    ss_tot = float(((t_all - t_all.mean()) ** 2).sum())
    overall_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "mean_r2": {"overall": overall_r2, "per_trait": per_trait_mean_r2},
        "mean_mae": {"overall": overall_mae, "per_trait": per_trait_mean_mae},
    }


def compute_baseline_metrics(
    Y_hist: np.ndarray,
    Y_mask: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    bin_edges: np.ndarray,
    trait_names: list[str] | None = None,
) -> dict:
    """Evaluate a mean-histogram baseline: predict the training mean for every cell.

    For each trait, the mean histogram is computed from training cells where
    that trait is valid.  This constant prediction is then evaluated against
    validation targets using :func:`evaluate_all`.

    Parameters
    ----------
    Y_hist : np.ndarray
        Full histogram array, shape ``(N_total, n_traits, n_bins)``.
    Y_mask : np.ndarray
        Full validity mask, shape ``(N_total, n_traits)``.
    train_idx, val_idx : np.ndarray
        Integer index arrays into the first axis of *Y_hist* / *Y_mask*.
    bin_edges : np.ndarray
        Shape ``(n_traits, n_bins + 1)``.
    trait_names : list[str] | None
        Optional trait names for labeling.

    Returns
    -------
    dict
        Same structure as :func:`evaluate_all`.
    """
    n_traits, n_bins = Y_hist.shape[1], Y_hist.shape[2]
    train_hist = Y_hist[train_idx]  # (N_train, n_traits, n_bins)
    train_mask = Y_mask[train_idx]  # (N_train, n_traits)

    # Per-trait masked mean histogram
    mean_hist = np.zeros((n_traits, n_bins), dtype=np.float64)
    for j in range(n_traits):
        valid = train_mask[:, j].astype(bool)
        if valid.any():
            mean_hist[j] = train_hist[valid, j].mean(axis=0)

    # Broadcast to all validation cells: (N_val, n_traits, n_bins)
    n_val = len(val_idx)
    baseline_pred = np.broadcast_to(mean_hist[np.newaxis], (n_val, n_traits, n_bins)).copy()

    return evaluate_all(
        baseline_pred, Y_hist[val_idx], Y_mask[val_idx], bin_edges,
        trait_names=trait_names,
    )


def evaluate_all(
    pred_probs: np.ndarray,
    target_probs: np.ndarray,
    mask: np.ndarray,
    bin_edges: np.ndarray,
    trait_names: list[str] | None = None,
) -> dict:
    """Run all evaluation metrics.

    Parameters
    ----------
    pred_probs : np.ndarray
        Predicted probability histograms, shape ``(N, n_traits, n_bins)``.
    target_probs : np.ndarray
        Target probability histograms, shape ``(N, n_traits, n_bins)``.
    mask : np.ndarray
        Validity mask, shape ``(N, n_traits)``.
    bin_edges : np.ndarray
        Bin edges, shape ``(n_traits, n_bins + 1)``.
    trait_names : list[str] | None
        Optional trait names for labeling.

    Returns
    -------
    dict
        Comprehensive evaluation results.
    """
    results = {
        "kl_divergence": compute_kl_divergence(pred_probs, target_probs, mask),
        "emd": compute_emd(pred_probs, target_probs, mask, bin_edges),
        "histogram_intersection": compute_histogram_intersection(
            pred_probs, target_probs, mask
        ),
        "moment_comparison": compute_moment_comparison(
            pred_probs, target_probs, mask, bin_edges
        ),
    }
    if trait_names is not None:
        results["trait_names"] = trait_names
    return results
