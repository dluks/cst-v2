"""PyTorch Dataset for histogram training data loaded from Zarr."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

log = logging.getLogger(__name__)

VODCA_SENTINEL = 32767.0


def load_zarr_arrays(zarr_path: Path) -> dict[str, np.ndarray | list[str]]:
    """Load all arrays and metadata from the train.zarr store into memory.

    Parameters
    ----------
    zarr_path : Path
        Path to train.zarr.

    Returns
    -------
    dict
        Keys: ``Y_hist``, ``Y_mask``, ``X``, ``coords``, ``source``,
        ``bin_edges``, ``feature_names``, ``trait_names``.
    """
    root = zarr.open_group(zarr_path, mode="r")

    data = {
        "Y_hist": np.asarray(root["Y_hist"]),
        "Y_mask": np.asarray(root["Y_mask"]),
        "X": np.asarray(root["X"]),
        "coords": np.asarray(root["coords"]),
        "source": np.asarray(root["source"]),
        "bin_edges": np.asarray(root["bin_edges"]),
        "feature_names": list(root.attrs.get("feature_names", [])),
        "trait_names": list(root.attrs.get("trait_names", [])),
    }

    log.info(
        "Loaded train.zarr: %d cells, %d traits, %d bins, %d features",
        data["Y_hist"].shape[0],
        data["Y_hist"].shape[1],
        data["Y_hist"].shape[2],
        data["X"].shape[1],
    )
    return data


def preprocess_features(
    X: np.ndarray,
    train_mask: np.ndarray,
    standardize: bool = True,
    vodca_sentinel: float = VODCA_SENTINEL,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Impute NaN/sentinels and optionally standardize features.

    Steps:

    1. Replace sentinel values (e.g. 32767) with NaN.
    2. Compute per-feature medians from training rows only.
    3. Fill NaN with training medians.
    4. Optionally standardize: ``(X - mean) / std`` using training statistics.

    Parameters
    ----------
    X : np.ndarray
        Raw features, shape ``(N, F)``.
    train_mask : np.ndarray
        Boolean mask indicating training rows, shape ``(N,)``.
    standardize : bool
        Whether to zero-mean / unit-variance standardize.
    vodca_sentinel : float
        Sentinel value to replace with NaN.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]]
        ``(preprocessed_X, stats)`` where stats contains ``'median'``,
        ``'mean'``, ``'std'`` arrays each of shape ``(F,)``.
    """
    X = X.copy()

    # Replace sentinels with NaN
    X[X == vodca_sentinel] = np.nan
    X[X == -vodca_sentinel] = np.nan

    # Compute stats from training rows only
    X_train = X[train_mask]
    medians = np.nanmedian(X_train, axis=0)

    # Impute NaN with training medians
    nan_mask = np.isnan(X)
    for j in range(X.shape[1]):
        X[nan_mask[:, j], j] = medians[j]

    stats: dict[str, np.ndarray] = {"median": medians}

    if standardize:
        # Recompute from imputed training data
        X_train = X[train_mask]
        means = X_train.mean(axis=0)
        stds = X_train.std(axis=0)
        stds[stds == 0] = 1.0  # Avoid division by zero for constant features
        X = (X - means) / stds
        stats["mean"] = means
        stats["std"] = stds

    n_imputed = nan_mask.sum()
    log.info(
        "Preprocessed features: imputed %d NaN values (%.2f%%), standardize=%s",
        n_imputed,
        100 * n_imputed / X.size,
        standardize,
    )
    return X.astype(np.float32), stats


class HistogramDataset(Dataset):
    """PyTorch Dataset providing (X, Y_hist, Y_mask, source) tuples.

    Parameters
    ----------
    X : np.ndarray
        Preprocessed features, shape ``(N, F)``.
    Y_hist : np.ndarray
        Histogram targets, shape ``(N, n_traits, n_bins)``.
    Y_mask : np.ndarray
        Validity mask, shape ``(N, n_traits)``.
    source : np.ndarray
        Source indicator, shape ``(N,)``. 0 = GBIF, 1 = sPlot.
    indices : np.ndarray | None
        Subset indices to use. If ``None``, use all rows.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y_hist: np.ndarray,
        Y_mask: np.ndarray,
        source: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> None:
        self.X = X
        self.Y_hist = Y_hist
        self.Y_mask = Y_mask
        self.source = source
        self.indices = indices

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        return len(self.X)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (features, histograms, mask, source) for a single cell."""
        i = self.indices[idx] if self.indices is not None else idx
        return (
            torch.from_numpy(self.X[i]),
            torch.from_numpy(self.Y_hist[i]),
            torch.from_numpy(self.Y_mask[i].astype(np.float32)),
            torch.tensor(self.source[i], dtype=torch.int8),
        )
