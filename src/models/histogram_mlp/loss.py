"""Masked KL divergence loss with source weighting."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedKLDivLoss(nn.Module):
    """KL divergence loss masked by trait validity and weighted by data source.

    Computes ``KL(target || pred)`` for each (cell, trait) pair, then:

    1. Zeros out invalid traits using ``mask``.
    2. Optionally weights by data source (sPlot vs GBIF).
    3. Averages over valid (cell, trait) pairs.

    Parameters
    ----------
    splot_weight : float
        Weight for sPlot samples. Default: 1.0.
    gbif_weight : float | None
        Weight for GBIF samples. If ``None``, all samples get weight 1.0
        (no source weighting).
    """

    def __init__(
        self,
        splot_weight: float = 1.0,
        gbif_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.splot_weight = splot_weight
        self.gbif_weight = gbif_weight

    def forward(
        self,
        log_pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute masked, weighted KL divergence.

        Parameters
        ----------
        log_pred : torch.Tensor
            Log-probability predictions, shape ``(B, n_traits, n_bins)``.
        target : torch.Tensor
            Target probability distributions, shape ``(B, n_traits, n_bins)``.
        mask : torch.Tensor
            Validity mask, shape ``(B, n_traits)``. True where trait is valid.
        source : torch.Tensor | None
            Source indicator, shape ``(B,)``. 0 = GBIF, 1 = sPlot.
            If ``None``, no source weighting is applied.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        # KL divergence per element, then sum over bins → (B, n_traits)
        kl = F.kl_div(log_pred, target, reduction="none").sum(dim=-1)

        # Mask invalid traits
        mask_f = mask.float()
        kl = kl * mask_f

        # Apply source weighting
        if source is not None and self.gbif_weight is not None:
            weights = torch.where(
                source == 1,
                torch.tensor(self.splot_weight, device=kl.device, dtype=kl.dtype),
                torch.tensor(self.gbif_weight, device=kl.device, dtype=kl.dtype),
            )
            # Broadcast (B,) → (B, 1) to weight each cell's traits equally
            kl = kl * weights.unsqueeze(-1)

        # Average over valid entries
        n_valid = mask_f.sum().clamp(min=1)
        return kl.sum() / n_valid
