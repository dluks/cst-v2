"""Spatial fold assignment for the joint histogram model."""

from __future__ import annotations

import logging

import h3
import numpy as np
import pyproj

log = logging.getLogger(__name__)


def assign_spatial_folds(
    coords: np.ndarray,
    n_folds: int = 5,
    h3_resolution: int = 2,
    n_iterations: int = 100,
    random_seed: int = 42,
    from_crs: str = "EPSG:6933",
) -> np.ndarray:
    """Assign spatial CV folds to cells based on H3 hexagon grouping.

    Converts cell coordinates to WGS84, assigns H3 hexagon IDs, then
    distributes hexagons across folds to maximize fold size balance.

    Parameters
    ----------
    coords : np.ndarray
        Cell center coordinates, shape ``(N, 2)`` as ``[x, y]`` in *from_crs*.
    n_folds : int
        Number of CV folds.
    h3_resolution : int
        H3 resolution for hexagon grouping. Lower = larger hexagons = more
        spatial separation between folds.
    n_iterations : int
        Number of random shuffles to find the best fold assignment.
    random_seed : int
        Random seed for reproducibility.
    from_crs : str
        CRS of input coordinates.

    Returns
    -------
    np.ndarray
        Fold assignments, shape ``(N,)``, values in ``[0, n_folds)``.
    """
    rng = np.random.RandomState(random_seed)

    # Convert to WGS84 for H3
    transformer = pyproj.Transformer.from_crs(from_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(coords[:, 0], coords[:, 1])

    # Assign H3 hex IDs
    hex_ids = np.array([
        h3.latlng_to_cell(la, lo, h3_resolution)
        for la, lo in zip(lat, lon)
    ])

    unique_hexes = np.unique(hex_ids)
    log.info(
        "Assigned %d cells to %d H3 hexagons (resolution %d)",
        len(coords), len(unique_hexes), h3_resolution,
    )

    # Map hex_id → cell indices
    hex_to_indices: dict[str, list[int]] = {}
    for i, h in enumerate(hex_ids):
        hex_to_indices.setdefault(h, []).append(i)

    # Optimize fold assignment for balanced fold sizes
    best_score = -1.0
    best_folds = np.zeros(len(coords), dtype=np.int32)

    for _ in range(n_iterations):
        shuffled = unique_hexes.copy()
        rng.shuffle(shuffled)
        hex_groups = np.array_split(shuffled, n_folds)
        hex_to_fold = {
            h: fold_id
            for fold_id, group in enumerate(hex_groups)
            for h in group
        }

        # Assign folds to cells
        trial_folds = np.array([hex_to_fold[h] for h in hex_ids], dtype=np.int32)

        # Compute balance score (1 - CV of fold sizes)
        fold_counts = np.array([
            (trial_folds == f).sum() for f in range(n_folds)
        ], dtype=float)
        mean_count = fold_counts.mean()
        if mean_count == 0:
            continue
        cv = fold_counts.std() / mean_count
        score = 1.0 - cv

        if score > best_score:
            best_score = score
            best_folds = trial_folds

    fold_counts = [(best_folds == f).sum() for f in range(n_folds)]
    log.info("Fold sizes: %s (balance score: %.4f)", fold_counts, best_score)

    return best_folds


def get_train_val_indices(
    folds: np.ndarray,
    source: np.ndarray,
    fold_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get training and validation indices for a specific fold.

    - **Training**: all cells NOT in ``fold_id``.
    - **Validation**: only sPlot cells (``source == 1``) IN ``fold_id``.
    - GBIF cells in ``fold_id`` are excluded from both train and val.

    Parameters
    ----------
    folds : np.ndarray
        Fold assignments, shape ``(N,)``.
    source : np.ndarray
        Source indicator, shape ``(N,)``. 0 = GBIF, 1 = sPlot.
    fold_id : int
        Fold to use for validation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(train_indices, val_indices)`` as integer index arrays.
    """
    in_fold = folds == fold_id
    is_splot = source == 1

    train_idx = np.where(~in_fold)[0]
    val_idx = np.where(in_fold & is_splot)[0]

    return train_idx, val_idx
