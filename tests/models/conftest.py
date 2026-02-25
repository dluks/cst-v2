"""Shared fixtures for histogram MLP tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def rng():
    """Deterministic numpy RNG."""
    return np.random.RandomState(42)


@pytest.fixture()
def dims():
    """Standard test dimensions."""
    return {"N": 64, "F": 10, "T": 4, "B": 20}


@pytest.fixture()
def synthetic_data(rng, dims):
    """Synthetic histogram training data."""
    N, F, T, B = dims["N"], dims["F"], dims["T"], dims["B"]

    # Random features
    X = rng.randn(N, F).astype(np.float32)

    # Random probability histograms (each sums to 1)
    raw = rng.dirichlet(np.ones(B), size=(N, T)).astype(np.float32)
    Y_hist = raw

    # Validity mask: ~80% valid
    Y_mask = (rng.rand(N, T) > 0.2).astype(np.float32)

    # Source: 60% GBIF (0), 40% sPlot (1)
    source = (rng.rand(N) > 0.6).astype(np.int8)

    # Bin edges: uniform [0, 1] per trait
    bin_edges = np.tile(
        np.linspace(0, 1, B + 1).astype(np.float32), (T, 1)
    )

    return {
        "X": X,
        "Y_hist": Y_hist,
        "Y_mask": Y_mask,
        "source": source,
        "bin_edges": bin_edges,
        "coords": rng.randn(N, 2).astype(np.float64) * 1e6,
        "trait_names": [f"trait_{i}" for i in range(T)],
    }
