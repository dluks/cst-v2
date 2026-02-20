"""MLP model for predicting trait probability histograms."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HistogramMLP(nn.Module):
    """Multi-layer perceptron predicting (n_traits, n_bins) log-probability histograms.

    Architecture::

        Input(n_features)
        → [Linear → ReLU → Dropout] × len(hidden_dims)
        → Linear(hidden_dims[-1], n_traits * n_bins)
        → Reshape(n_traits, n_bins)
        → LogSoftmax(dim=-1)

    Parameters
    ----------
    n_features : int
        Number of input EO features.
    n_traits : int
        Number of traits.
    n_bins : int
        Number of histogram bins per trait.
    hidden_dims : list[int] | None
        Hidden layer dimensions. Default: [512, 256, 256].
    dropout : float
        Dropout probability. Default: 0.2.
    """

    def __init__(
        self,
        n_features: int = 151,
        n_traits: int = 31,
        n_bins: int = 20,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_traits = n_traits
        self.n_bins = n_bins

        if hidden_dims is None:
            hidden_dims = [512, 256, 256]

        layers: list[nn.Module] = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, n_traits * n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape ``(batch_size, n_features)``.

        Returns
        -------
        torch.Tensor
            Log-probability histograms, shape ``(batch_size, n_traits, n_bins)``.
        """
        h = self.encoder(x)
        logits = self.head(h)
        logits = logits.view(-1, self.n_traits, self.n_bins)
        return F.log_softmax(logits, dim=-1)
