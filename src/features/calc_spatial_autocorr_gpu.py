"""
Calculates spatial autocorrelation for a single trait using GPU acceleration.

This module computes variograms using great-circle (Haversine) distances on
lat/lon coordinates, providing scientifically accurate pairwise distances for
global data. Key features:

- Great-circle distances for accurate global spatial analysis
- Self-pair exclusion and upper-triangular-only computation (no double-counting)
- H3 hierarchical spatial blocking tuned to sample density
- Adaptive log-binning based on nearest-neighbor distances
- Constrained range parameter fitting to prevent unrealistic extrapolation
- Pair-weighted combination of chunk estimates

Coordinates are expected in (lon, lat) degrees (EPSG:4326) or will be
transformed to lat/lon if in a different CRS.
"""

import argparse
import logging
from pathlib import Path

import dask.dataframe as dd
import h3
import numpy as np
import pandas as pd
import torch
import utm
from box import ConfigBox
from dask import compute
from pyproj import Transformer
from torch.optim.adam import Adam

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, init_dask


def cli() -> argparse.Namespace:
    """Command line interface for calculating spatial autocorrelation."""
    parser = argparse.ArgumentParser(
        description="Calculate spatial autocorrelation for a single trait using GPU "
        "acceleration."
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        help="Path to a params.yaml.",
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite existing files."
    )
    parser.add_argument(
        "-t",
        "--trait",
        type=str,
        required=True,
        help="Trait name to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for trait result.",
    )
    return parser.parse_args()


# Available GPUs
# Under Slurm: GPU_DEVICES=[0] (Slurm maps allocated GPU to device 0)
# Local: GPU_DEVICES from config (e.g., [0, 1, 2, 3])
GPU_DEVICES = []
CURRENT_GPU_IDX = 0  # Only used for round-robin with multiple GPUs


def main(args: argparse.Namespace) -> None:
    """Main function for calculating spatial autocorrelation for a single trait."""
    # Import locally to avoid module-level config loading
    from src.utils.dataset_utils import get_y_fn

    # Load config with params if provided
    params_path = Path(args.params).resolve() if args.params else None
    log.info("Loading config from %s", params_path or "default")
    cfg = get_config(params_path)

    syscfg = cfg[detect_system()]["calc_spatial_autocorr"]

    if args.debug:
        log.info("Running in debug mode...")
        log.setLevel(logging.DEBUG)

    # Set up output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_fp = output_dir / f"spatial_autocorr_{args.trait}.parquet"

    # Check if output already exists
    if output_fp.exists() and not args.overwrite:
        log.warning("Output file already exists: %s", output_fp)
        log.warning("Use --overwrite flag to overwrite existing files.")
        return

    # Set GPU devices: respect Slurm allocation if running under Slurm,
    # otherwise use config-specified GPU IDs for local execution
    import os
    if "SLURM_JOB_ID" in os.environ:
        # Running under Slurm - use device 0 (Slurm maps allocated GPU to device 0)
        set_gpu_devices([0])
        log.info(
            "Running under Slurm - using GPU device 0 "
            "(mapped via CUDA_VISIBLE_DEVICES)"
        )
    elif hasattr(syscfg, "gpu_ids"):
        # Local execution - use config-specified GPU IDs
        set_gpu_devices(syscfg.gpu_ids)
        log.info(f"Local execution - using GPU devices from config: {GPU_DEVICES}")
    else:
        # Fallback
        set_gpu_devices([0])
        log.warning("No GPU IDs specified in config, defaulting to [0]")

    # Check GPU availability
    using_gpu = check_gpu_availability()
    if not using_gpu:
        log.warning("Falling back to CPU calculations")

    y_fn = get_y_fn(cfg)

    log.info("Initializing Dask...")
    client, _ = init_dask(
        dashboard_address=cfg.dask_dashboard,
        n_workers=syscfg.n_workers,
        threads_per_worker=1,
    )

    # Use only sPlot data to calculate spatial autocorrelation
    log.info("Reading sPlot features from %s...", y_fn)
    valid_cols = (
        dd.read_parquet(y_fn).columns.difference(["x", "y", "source"]).to_list()
    )

    # Validate trait exists
    if args.trait not in valid_cols:
        log.error(f"Trait '{args.trait}' not found in active traits: {valid_cols}")
        close_dask(client)
        return

    log.info(f"Processing trait: {args.trait}")

    # Read only the specified trait (including x and y coordinates)
    y_df = (
        dd.read_parquet(y_fn, columns=["x", "y", "source", args.trait])
        .query("source == 's'")
        .drop(columns=["source"])
        .dropna(subset=[args.trait])
        .reset_index(drop=True)
        .astype(np.float32)
        .compute()
    )

    log.info(
        "Using coordinates in %s (will compute great-circle distances)...",
        cfg.crs,
    )
    # Keep original coordinates in lat/lon for great-circle distance calculation
    # Note: Assuming cfg.crs is in lat/lon format (e.g., EPSG:4326)
    # If not, we need to transform to lat/lon first
    if cfg.crs != "EPSG:4326":
        log.info("Transforming coordinates from %s to EPSG:4326 (lat/lon)...", cfg.crs)
        transformer = Transformer.from_crs(cfg.crs, "EPSG:4326", always_xy=True)

        # Transform to lat/lon
        x, y = transformer.transform(y_df["x"], y_df["y"])

        y_df = y_df.assign(
            x=pd.Series(x, index=y_df.index),
            y=pd.Series(y, index=y_df.index),
        )
    else:
        log.info(
            "Coordinates already in EPSG:4326 (lat/lon), no transformation needed."
        )

    vgram_kwargs = {"n_max": 18000, "nlags": 50, "log_binning": True}

    # Compute directly without Dask delayed overhead
    log.info(f"Computing range statistics for trait: {args.trait}...")

    # Call the function directly (it's not decorated with @delayed)
    ranges_df = _single_trait_ranges(y_df, args.trait, cfg, syscfg, vgram_kwargs)

    close_dask(client)

    # Save results
    log.info("Saving trait result to %s...", output_fp)
    ranges_df.to_parquet(output_fp)
    log.info(f"Successfully saved result for trait: {args.trait}")


def set_gpu_devices(gpu_ids):
    """Set the available GPU devices from configuration."""
    global GPU_DEVICES
    GPU_DEVICES = gpu_ids
    log.info(f"Using GPU devices: {GPU_DEVICES}")


def get_next_gpu() -> int:
    """Returns the next GPU ID in round-robin fashion.

    For single-GPU setups (e.g., Slurm jobs with --gpus 1), always returns 0.
    For multi-GPU setups, rotates through available devices.
    """
    global CURRENT_GPU_IDX, GPU_DEVICES
    if not GPU_DEVICES:
        log.warning("No GPU devices configured, defaulting to device 0")
        return 0

    # Single GPU - no need for round-robin
    if len(GPU_DEVICES) == 1:
        return GPU_DEVICES[0]

    # Multiple GPUs - rotate through them
    gpu_id = GPU_DEVICES[CURRENT_GPU_IDX]
    CURRENT_GPU_IDX = (CURRENT_GPU_IDX + 1) % len(GPU_DEVICES)
    return gpu_id


def get_utm_zones(x: np.ndarray, y: np.ndarray) -> tuple[list, list, list]:
    """
    Converts latitude and longitude coordinates to UTM zones.

    Args:
        x (np.ndarray): Array of longitude coordinates.
        y (np.ndarray): Array of latitude coordinates.

    Returns:
        tuple[list, list, list]: A tuple containing three lists - eastings, northings,
            and zones.
    """
    eastings, northings, zones = [], [], []

    for x_, y_ in zip(x, y):
        easting, northing, zone, letter = utm.from_latlon(y_, x_)
        eastings.append(easting)
        northings.append(northing)
        zones.append(f"{zone}{letter}")

    return eastings, northings, zones


def add_utm(df: pd.DataFrame, chunksize: int = 10000) -> pd.DataFrame:
    """
    Adds UTM coordinates to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to which UTM coordinates will be added.
        chunksize (int, optional): The size of each chunk for parallel processing.
            Defaults to 10000.

    Returns:
        pd.DataFrame: The DataFrame with UTM coordinates added.
    """
    x = df.x.to_numpy()
    y = df.y.to_numpy()

    # Split x and y into chunks
    x_chunks = [x[i : i + chunksize] for i in range(0, len(x), chunksize)]
    y_chunks = [y[i : i + chunksize] for i in range(0, len(y), chunksize)]

    # Compute the UTM zones for each chunk in parallel
    results = [
        get_utm_zones(x_chunk, y_chunk) for x_chunk, y_chunk in zip(x_chunks, y_chunks)
    ]

    results = compute(*results)

    # Assign the results to new columns in df
    df["x"] = [e for result in results for e in result[0]]
    df["y"] = [n for result in results for n in result[1]]
    df["zone"] = [z for result in results for z in result[2]]

    return df


def spherical_model(
    h: torch.Tensor,
    nugget: float | torch.Tensor,
    sill: float | torch.Tensor,
    range_param: float | torch.Tensor,
) -> torch.Tensor:
    """
    Compute the spherical semivariogram model.

    Args:
        h (torch.Tensor): Distance tensor
        nugget (float): Nugget effect parameter
        sill (float): Sill parameter
        range_param (float): Range parameter

    Returns:
        torch.Tensor: Semivariogram values
    """
    result = torch.zeros_like(h)
    mask = h <= range_param
    scaled_h = h[mask] / range_param
    result[mask] = nugget + sill * (1.5 * scaled_h - 0.5 * scaled_h**3)
    result[~mask] = nugget + sill
    return result


def _calculate_haversine_distance(
    coords1: torch.Tensor, coords2: torch.Tensor, R: float = 6371000.0
) -> torch.Tensor:
    """
    Calculate great-circle distances using Haversine formula.

    Args:
        coords1 (torch.Tensor): First set of coordinates [n, 2] in
            (lon, lat) degrees
        coords2 (torch.Tensor): Second set of coordinates [m, 2] in
            (lon, lat) degrees
        R (float): Earth radius in meters (default: 6371 km)

    Returns:
        torch.Tensor: Great-circle distances in meters [n, m]
    """
    # coords1: [n, 2], coords2: [m, 2] in (lon, lat) degrees
    # Convert to radians
    coords1_rad = coords1 * (torch.pi / 180.0)
    coords2_rad = coords2 * (torch.pi / 180.0)

    # Extract lon, lat
    lon1 = coords1_rad[:, 0].unsqueeze(1)  # [n, 1]
    lat1 = coords1_rad[:, 1].unsqueeze(1)  # [n, 1]
    lon2 = coords2_rad[:, 0].unsqueeze(0)  # [1, m]
    lat2 = coords2_rad[:, 1].unsqueeze(0)  # [1, m]

    # Haversine formula:
    # a = sin²(Δlat/2) + cos(lat1)*cos(lat2)*sin²(Δlon/2)
    # c = 2*atan2(√a, √(1-a))
    # d = R * c

    dlat = lat2 - lat1  # [n, m]
    dlon = lon2 - lon1  # [n, m]

    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return R * c  # [n, m]


def _estimate_max_distance(coords_tensor: torch.Tensor, n_samples: int) -> float:
    """
    Estimate maximum distance from a subset of data points.

    Args:
        coords_tensor (torch.Tensor): Coordinates tensor in (lon, lat) degrees
        n_samples (int): Total number of samples

    Returns:
        float: Estimated maximum distance in meters
    """
    with torch.no_grad():
        # Estimate max distance from a subset of data
        subset_size = min(1000, n_samples)
        # Don't use a fixed seed here - we want to sample the full spatial extent
        # Using the same seed would give identical distance estimates across traits
        indices = torch.randperm(n_samples, device=coords_tensor.device)[:subset_size]
        subset_coords = coords_tensor[indices]

        # Calculate great-circle distances for the subset
        subset_dists = _calculate_haversine_distance(subset_coords, subset_coords)

        max_dist = float(subset_dists.max().item() * 0.7)  # Use 70% of max distance
        log.debug(
            f"Estimated max_dist: {max_dist:.2f}m from {subset_size} points, "
            f"raw_max={subset_dists.max().item():.2f}"
        )
        return max_dist


def _estimate_min_distance(
    coords_tensor: torch.Tensor, n_samples: int, percentile: float = 0.01
) -> float:
    """
    Estimate minimum distance for log-binning using nearest-neighbor distances.

    Uses a percentile of nearest-neighbor distances to ensure short lags are
    captured for nugget estimation.

    Args:
        coords_tensor (torch.Tensor): Coordinates tensor in (lon, lat) degrees
        n_samples (int): Total number of samples
        percentile (float): Percentile of nearest-neighbor distances to use
            (default: 0.01 = 1st percentile)

    Returns:
        float: Estimated minimum distance in meters
    """
    with torch.no_grad():
        # Sample a subset for efficiency
        subset_size = min(2000, n_samples)
        # Don't use a fixed seed here - we want to sample the full spatial extent
        # Using the same seed would give identical distance estimates across traits
        indices = torch.randperm(n_samples, device=coords_tensor.device)[:subset_size]
        subset_coords = coords_tensor[indices]

        # Calculate pairwise distances
        dists = _calculate_haversine_distance(subset_coords, subset_coords)

        # For each point, find nearest neighbor (exclude self by masking diagonal)
        diagonal_mask = torch.eye(
            subset_size, device=coords_tensor.device, dtype=torch.bool
        )
        dists_masked = dists.masked_fill(diagonal_mask, float("inf"))

        # Get minimum distance for each point (nearest neighbor)
        nearest_neighbor_dists = dists_masked.min(dim=1).values

        # Use a percentile of these distances as minimum bin edge
        min_dist = float(torch.quantile(nearest_neighbor_dists, percentile).item())

        # Ensure a reasonable lower bound (500m) to avoid numerical issues
        # but don't force it higher than necessary
        min_dist = max(500.0, min_dist)

        log.debug(
            f"Estimated min_dist: {min_dist:.2f}m "
            f"(p{percentile * 100:.1f} of nearest-neighbor distances)"
        )
        return min_dist


def _compute_experimental_variogram(
    coords_tensor: torch.Tensor,
    values_tensor: torch.Tensor,
    bin_edges: torch.Tensor,
    nlags: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the experimental variogram from coordinates and values.

    Excludes self-pairs and avoids double-counting by only processing
    upper-triangular pairs (j >= i).

    Args:
        coords_tensor (torch.Tensor): Coordinates tensor in (lon, lat) degrees
        values_tensor (torch.Tensor): Values tensor
        bin_edges (torch.Tensor): Bin edges for distance binning
        nlags (int): Number of lags
        chunk_size (int): Size of chunks for processing
        device (torch.device): PyTorch device

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Gamma values and counts for each bin
    """
    n_samples = coords_tensor.shape[0]

    # Initialize arrays to accumulate semivariance values
    gamma_sum = torch.zeros(nlags, device=device)
    gamma_counts = torch.zeros(nlags, device=device)

    # Process data in chunks, only upper-triangular to avoid double-counting
    for i in range(0, n_samples, chunk_size):
        i_end = min(i + chunk_size, n_samples)
        chunk_coords = coords_tensor[i:i_end]
        chunk_values = values_tensor[i:i_end]

        # Start j from i to process only upper-triangular pairs
        for j in range(i, n_samples, chunk_size):
            j_end = min(j + chunk_size, n_samples)

            # Calculate great-circle distances
            dists = _calculate_haversine_distance(chunk_coords, coords_tensor[j:j_end])

            # Calculate pairwise semivariances
            v1 = chunk_values.unsqueeze(1)
            v2 = values_tensor[j:j_end].unsqueeze(0)
            sv = 0.5 * (v1 - v2) ** 2

            # When i == j block, exclude diagonal (self-pairs)
            if i == j:
                # Create mask to exclude diagonal elements
                chunk_size_i = i_end - i
                chunk_size_j = j_end - j
                diagonal_mask = torch.eye(
                    chunk_size_i, chunk_size_j, device=device, dtype=torch.bool
                )
                # Zero out diagonal in distances and semivariances
                dists = dists.masked_fill(diagonal_mask, float("inf"))

            # Bin the semivariances by distance
            for lag in range(nlags):
                mask = (dists >= bin_edges[lag]) & (dists < bin_edges[lag + 1])
                gamma_sum[lag] += sv[mask].sum()
                gamma_counts[lag] += mask.sum()

    return gamma_sum, gamma_counts


def _fit_variogram_model(
    bin_centers: torch.Tensor,
    gamma: torch.Tensor,
    valid_lags: torch.Tensor,
    device: torch.device,
    min_range: float | None = None,
    max_range: float | None = None,
    weights: torch.Tensor | None = None,
) -> tuple[float, float, float, float]:
    """
    Fit a spherical variogram model to the experimental variogram.

    Constrains the range parameter to be within [min_range, max_range] using
    a sigmoid transformation to prevent unrealistic range estimates.

    Args:
        bin_centers (torch.Tensor): Centers of distance bins
        gamma (torch.Tensor): Experimental variogram values
        valid_lags (torch.Tensor): Mask of valid lags
        device (torch.device): PyTorch device
        min_range (float | None): Minimum allowed range (default: first bin center)
        max_range (float | None): Maximum allowed range (default: last bin edge)
        weights (torch.Tensor | None): Optional per-bin weights (e.g., pair counts)

    Returns:
        tuple[float, float, float, float]: Range, nugget, sill, and (weighted) MSE
    """
    bin_centers_valid = bin_centers[valid_lags]
    gamma_valid = gamma[valid_lags]

    # Check if we have any valid lags
    if bin_centers_valid.shape[0] == 0:
        raise ValueError(
            "No valid distance bins found for variogram fitting. "
            "This may indicate that all distances fall outside the bin range, "
            "or that there are insufficient pairs at any distance lag."
        )

    with torch.no_grad():
        # Initial parameter estimates
        nugget_init = gamma_valid[0].item() if gamma_valid.shape[0] > 0 else 0.0
        sill_init = gamma_valid[-5:].mean().item() - nugget_init
        range_init = bin_centers_valid[-1].item() * 0.6

        # Set range bounds if not provided
        if min_range is None:
            min_range = bin_centers_valid[0].item()
        if max_range is None:
            max_range = bin_centers_valid[-1].item()

        # Transform initial range to unconstrained space
        # range = min_range + (max_range - min_range) * sigmoid(range_param)
        # Inverse: range_param = logit((range - min_range) / (max_range - min_range))
        range_normalized = (range_init - min_range) / (max_range - min_range)
        range_normalized = max(0.01, min(0.99, range_normalized))  # Clip to valid range
        range_param_init = torch.logit(torch.tensor(range_normalized)).item()

    # Fit parameters using optimization
    # nugget and sill use softplus, range uses sigmoid with bounds
    params = torch.tensor(
        [nugget_init, sill_init, range_param_init], device=device, requires_grad=True
    )

    optimizer = Adam([params], lr=0.01)

    min_range_tensor = torch.tensor(min_range, device=device)
    max_range_tensor = torch.tensor(max_range, device=device)

    weights_valid = None
    if weights is not None:
        weights_valid = weights[valid_lags].to(device)
        # Normalize to keep loss scale comparable
        weights_valid = weights_valid / (weights_valid.sum() + 1e-12)

    for _ in range(500):
        optimizer.zero_grad()
        nugget, sill, range_param = params

        # Ensure parameters are positive
        nugget_pos = torch.nn.functional.softplus(nugget)
        sill_pos = torch.nn.functional.softplus(sill)

        # Constrain range to [min_range, max_range] using sigmoid
        range_pos = min_range_tensor + (
            max_range_tensor - min_range_tensor
        ) * torch.sigmoid(range_param)

        # Calculate predicted values using spherical model
        pred = spherical_model(bin_centers_valid, nugget_pos, sill_pos, range_pos)

        # Weighted or unweighted MSE loss
        residuals = pred - gamma_valid
        if weights_valid is not None:
            mse_loss = torch.sum(weights_valid * residuals**2)
        else:
            mse_loss = torch.mean(residuals**2)

        loss = mse_loss
        loss.backward()
        optimizer.step()

    # Get final parameters
    with torch.no_grad():
        nugget, sill, range_param = params
        nugget = torch.nn.functional.softplus(nugget).item()
        sill = torch.nn.functional.softplus(sill).item()
        range_val = (
            min_range + (max_range - min_range) * torch.sigmoid(range_param)
        ).item()

        # Final (weighted) MSE
        pred = spherical_model(bin_centers_valid, nugget, sill, range_val)
        residuals = pred - gamma_valid
        if weights_valid is not None:
            mse = torch.sum(weights_valid * residuals**2).item()
        else:
            mse = torch.mean(residuals**2).item()

        return range_val, nugget, sill, mse


def fit_variogram_gpu(
    coords: np.ndarray,
    values: np.ndarray,
    nlags: int = 50,
    max_dist: float | None = None,
    gpu_id: int = 0,
    crs: str = "EPSG:6933",
    log_binning: bool = False,
) -> tuple[float, float, float, float]:
    """
    Fit spherical variogram model using GPU with great-circle distances.

    Args:
        coords (np.ndarray): Coordinates array of shape (n_samples, 2) in
            (lon, lat) degrees
        values (np.ndarray): Values array of shape (n_samples,)
        nlags (int): Number of distance lags
        max_dist (float | None): Maximum distance to consider (in meters)
        gpu_id (int): GPU device ID to use
        crs (str): Coordinate reference system of input coordinates
            (unused, for API compatibility)
        log_binning (bool): Whether to use logarithmic binning

    Returns:
        Tuple[float, float, float, float]: Estimated range, nugget,
            sill, and (weighted) MSE
    """
    # Set the GPU device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Convert to PyTorch tensors and move to GPU
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)

    # Validate coordinates
    if torch.isnan(coords_tensor).any() or torch.isinf(coords_tensor).any():
        n_nan = torch.isnan(coords_tensor).sum().item()
        n_inf = torch.isinf(coords_tensor).sum().item()
        raise ValueError(
            f"Coordinates contain {n_nan} NaN and {n_inf} inf values! "
            f"coords range: [{coords_tensor.min().item()}, "
            f"{coords_tensor.max().item()}]"
        )

    # Number of samples
    n_samples = coords_tensor.shape[0]

    # Process in chunks to handle large datasets
    chunk_size = 30000

    # Initialize bins for the experimental variogram
    if max_dist is None:
        max_dist = _estimate_max_distance(coords_tensor, n_samples)

    # Create distance bins - either linear or logarithmic
    # Start at small epsilon to ensure zero distance never falls in any bin
    epsilon = 1e-6  # effectively excludes self-pairs

    if log_binning:
        # Use logarithmic binning for adaptive bin sizes
        # Estimate minimum distance from nearest-neighbor distribution
        # to capture short lags crucial for nugget estimation
        min_dist = _estimate_min_distance(coords_tensor, n_samples, percentile=0.01)

        log.info(
            f"Log-binning from {min_dist:.0f}m to {max_dist:.0f}m with {nlags} bins"
        )

        bin_edges = torch.logspace(
            torch.log10(torch.tensor(min_dist)),
            torch.log10(torch.tensor(max_dist)),
            nlags + 1,
            device=device,
        )
    else:
        # Use linear binning starting from epsilon (not 0)
        bin_edges = torch.linspace(epsilon, max_dist, nlags + 1, device=device)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute experimental variogram
    gamma_sum, gamma_counts = _compute_experimental_variogram(
        coords_tensor, values_tensor, bin_edges, nlags, chunk_size, device
    )

    # Calculate mean semivariance for each bin
    with torch.no_grad():
        # Filter out sparsely populated bins to reduce noise
        # dynamic threshold: at least 100 pairs or 1% of max count across bins
        min_pairs_abs = 100.0
        max_count_val = float(
            gamma_counts.max().item() if gamma_counts.numel() > 0 else 0.0
        )
        min_pairs_rel = 0.01 * max_count_val
        min_pairs = max(min_pairs_abs, min_pairs_rel)

        valid_lags = gamma_counts >= min_pairs
        gamma = torch.zeros_like(gamma_counts)
        gamma[valid_lags] = gamma_sum[valid_lags] / gamma_counts[valid_lags]

        # Debug logging
        n_valid = valid_lags.sum().item()
        log.debug(
            f"Variogram: {n_valid}/{nlags} bins with data (min_pairs={min_pairs:.0f}), "
            f"max_dist={max_dist:.0f}m, "
            f"bin range=[{bin_edges[0]:.0f}, {bin_edges[-1]:.0f}]"
        )
        if n_valid == 0:
            log.warning(
                f"No valid distance bins! "
                f"coords shape={coords_tensor.shape}, "
                f"max_dist={max_dist:.2f}"
            )

    # Fit variogram model with range constraint and bin-count weights
    max_range_constraint = float(bin_edges[-1].item())
    return _fit_variogram_model(
        bin_centers,
        gamma,
        valid_lags,
        device,
        max_range=max_range_constraint,
        weights=gamma_counts,
    )


def calculate_variogram_gpu(
    group: pd.DataFrame, data_col: str, **kwargs
) -> tuple[float, int]:
    """
    Calculate the variogram for a given group of data points using GPU.

    Parameters:
        group (pd.DataFrame): The group of data points.
        data_col (str): The column name of the data points.
        **kwargs: Additional keyword arguments.

    Returns:
        float: The variogram range parameter.
        int: The number of samples used to calculate the variogram.
    """
    log.info("Calculating variogram for group with %d samples", len(group))
    if not isinstance(group, pd.DataFrame) or len(group) < 200:
        log.info("Skipping variogram calculation for group with less than 200 samples")
        return 0, 0

    n_max = 20_000
    if "n_max" in kwargs:
        n_max = kwargs.pop("n_max")

    group = group.copy()
    if len(group) > n_max:
        group = group.sample(n_max, random_state=42)

    # Set n_lags dynamically based on the number of samples
    nlags = min(50, max(10, len(group) // 20))
    if "nlags" in kwargs:
        nlags = kwargs.pop("nlags")

    # Get coordinates and values
    coords = group[["x", "y"]].values.astype(np.float32)
    values = group[data_col].values.astype(np.float32)

    # Get next available GPU
    gpu_id = get_next_gpu()

    # Get CRS from kwargs or default to EPSG:6933
    crs = kwargs.get("crs", "EPSG:6933")
    log_binning = kwargs.get("log_binning", False)

    # Use fixed max_dist if provided
    max_dist = kwargs.get("max_dist")

    try:
        # Fit variogram using GPU with great-circle distances
        range_param, nugget, sill, mse = fit_variogram_gpu(
            coords=coords,
            values=values,
            nlags=nlags,
            max_dist=max_dist,  # Pass through the fixed max_dist
            gpu_id=gpu_id,
            crs=crs,
            log_binning=log_binning,
        )

        return range_param, len(group)

    except Exception as e:
        log.error(f"GPU variogram calculation failed: {e}")
        import traceback

        log.error(f"Full traceback:\n{traceback.format_exc()}")
        log.info("Falling back to CPU calculation...")
        # Fallback to CPU calculation if GPU fails
        from pykrige.ok import OrdinaryKriging

        # Filter out GPU-specific parameters that OrdinaryKriging doesn't accept
        gpu_specific_params = {"log_binning", "max_dist", "crs"}
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in gpu_specific_params
        }

        orig_kwargs = {
            "variogram_model": "spherical",
            "nlags": nlags,
            "anisotropy_scaling": 1,
            "anisotropy_angle": 0,
        }
        orig_kwargs.update(**filtered_kwargs)

        ok_vgram = OrdinaryKriging(
            group["x"], group["y"], group[data_col], **orig_kwargs
        )
        return ok_vgram.variogram_model_parameters[1], len(group)


def _determine_h3_resolution(
    n_samples: int, target_samples_per_cell: int = 5000
) -> int:
    """
    Determine appropriate H3 resolution based on sample density.

    H3 resolutions and approximate cell counts globally:
    - res 0: ~122 cells (very coarse, ~4M km² per cell)
    - res 1: ~842 cells (~600k km² per cell)
    - res 2: ~5882 cells (~86k km² per cell)
    - res 3: ~41162 cells (~12k km² per cell)
    - res 4: ~288122 cells (~1800 km² per cell)
    - res 5: ~2M cells (~252 km² per cell)

    Args:
        n_samples (int): Total number of samples
        target_samples_per_cell (int): Target number of samples per H3 cell

    Returns:
        int: H3 resolution (0-5)
    """
    # Estimate desired number of cells
    target_n_cells = max(1, n_samples // target_samples_per_cell)

    # Map to appropriate H3 resolution
    # Use global cell counts to estimate resolution
    if target_n_cells <= 122:
        return 0
    elif target_n_cells <= 842:
        return 1
    elif target_n_cells <= 5882:
        return 2
    elif target_n_cells <= 41162:
        return 3
    elif target_n_cells <= 288122:
        return 4
    else:
        return 5  # Maximum resolution we'll use for global data


def _assign_h3_cells(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """
    Assign H3 cells to DataFrame based on lat/lon coordinates.

    Args:
        df (pd.DataFrame): DataFrame with 'x' (lon) and 'y' (lat) columns
        resolution (int): H3 resolution level

    Returns:
        pd.DataFrame: DataFrame with additional 'h3_cell' column
    """
    # H3 expects (lat, lon) order
    # Using h3.geo_to_h3 for h3-py v3.x or h3.latlng_to_cell for v4.x
    try:
        # Try v4.x API first
        h3_cells = [
            h3.latlng_to_cell(lat, lon, resolution) for lon, lat in zip(df.x, df.y)
        ]
    except AttributeError:
        # Fall back to v3.x API
        h3_cells = [h3.geo_to_h3(lat, lon, resolution) for lon, lat in zip(df.x, df.y)]
    return df.assign(h3_cell=h3_cells)


def _single_trait_ranges(
    ddf: pd.DataFrame,
    trait_col: str,
    cfg: ConfigBox,
    syscfg: ConfigBox,
    vgram_kwargs: dict,
) -> pd.DataFrame:
    trait_df = ddf

    log.info("Calculating variogram ranges for %s...", trait_col)

    # Set a fixed max_dist for consistent scale across all chunks
    # fixed_max_dist = 6_000_000  # 6000 km in meters - approximate half-Earth distance
    # vgram_kwargs["max_dist"] = fixed_max_dist

    # Calculate global variogram first (with smaller sample)
    global_kwargs = vgram_kwargs.copy()
    global_kwargs["n_max"] = min(30000, len(trait_df))
    global_sample = (
        trait_df.sample(global_kwargs["n_max"], random_state=42)
        if len(trait_df) > global_kwargs["n_max"]
        else trait_df
    )
    global_result = calculate_variogram_gpu(global_sample, trait_col, **global_kwargs)

    # Then calculate by chunks for local patterns using H3 spatial blocking
    if syscfg.n_chunks > 1:
        log.info("Using H3 spatial blocking for robust local estimates...")

        # Determine appropriate H3 resolution based on sample density
        # Target ~5000 samples per cell for good local estimates
        h3_resolution = _determine_h3_resolution(
            len(trait_df), target_samples_per_cell=5000
        )
        log.info(f"Using H3 resolution {h3_resolution} for {len(trait_df)} samples")

        # Assign H3 cells to each sample
        trait_df = _assign_h3_cells(trait_df, h3_resolution)

        # Filter to cells with sufficient samples for reliable variogram estimation
        min_samples_per_cell = 100
        cell_counts = trait_df.h3_cell.value_counts()
        valid_cells = cell_counts[cell_counts >= min_samples_per_cell].index

        log.info(
            f"Found {len(valid_cells)} H3 cells with >={min_samples_per_cell} "
            f"samples (out of {trait_df.h3_cell.nunique()} total cells)"
        )

        # Set max_samples per cell to balance representation
        samples_per_chunk = min(10000, len(trait_df) // len(valid_cells))
        vgram_kwargs["n_max"] = samples_per_chunk

        results = [
            calculate_variogram_gpu(group, trait_col, **vgram_kwargs)
            for cell, group in trait_df.groupby("h3_cell")
            if cell in valid_cells
        ]

        # Add global result to the chunk results
        results.append(global_result)
    else:
        results = [global_result]

    autocorr_ranges = list(compute(*results))
    if len(autocorr_ranges) == 1:
        global_range, _ = autocorr_ranges[0]
        # No local ranges to aggregate; produce a minimal output row
        filt_ranges = []
        sample_sizes = np.array([], dtype=float)
    else:
        global_range, _ = autocorr_ranges[-1]
        autocorr_ranges = autocorr_ranges[:-1]

        filt_ranges = [
            (r, n) for r, n in autocorr_ranges if isinstance(n, (int, float)) and n > 0
        ]
        # Weight by number of unique pairs per chunk
        sample_sizes = np.array([n for _, n in filt_ranges])

    n_pairs = (
        sample_sizes * (sample_sizes - 1) / 2 if sample_sizes.size else np.array([])
    )
    weights = (n_pairs / n_pairs.sum()) if n_pairs.size else np.array([])
    ranges = np.array([r for r, _ in filt_ranges]) if filt_ranges else np.array([])

    # Create output row
    if ranges.size and weights.size:
        mean_val = float(np.average(ranges, weights=weights))
        std_val = float(
            np.sqrt(np.average((ranges - ranges.mean()) ** 2, weights=weights))
        )
        median_val = float(np.median(ranges))
        q05_val = float(np.quantile(ranges, 0.05))
        q95_val = float(np.quantile(ranges, 0.95))
        n_chunks_val = int(len(ranges))
        n_val = int(sample_sizes.sum())
    else:
        # Fallback if only global or no valid chunks
        mean_val = float(global_range)
        std_val = 0.0
        median_val = float(global_range)
        q05_val = float(global_range)
        q95_val = float(global_range)
        n_chunks_val = 1
        n_val = int(len(trait_df))

    new_ranges = pd.DataFrame(
        [
            {
                "trait": trait_col,
                "mean": mean_val,
                "std": std_val,
                "median": median_val,
                "q05": q05_val,
                "q95": q95_val,
                "n": n_val,
                "n_chunks": n_chunks_val,
            }
        ]
    )

    # Add stability metrics to output
    new_ranges["stability"] = 1.0 - (new_ranges["std"] / new_ranges["mean"])
    new_ranges["global_range"] = global_range

    return new_ranges


def check_gpu_availability() -> bool:
    """Check if GPUs are available and print their info."""
    if not torch.cuda.is_available():
        log.warning("CUDA not available. Using CPU instead.")
        return False

    log.info(f"CUDA available: {torch.cuda.is_available()}")
    log.info(f"Total GPUs: {torch.cuda.device_count()}")

    for i in GPU_DEVICES:
        if i < torch.cuda.device_count():
            log.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            log.warning(f"GPU {i} not available")

    # Ensure we have at least one valid GPU
    valid_gpus = [i for i in GPU_DEVICES if i < torch.cuda.device_count()]
    if not valid_gpus:
        log.warning("None of the specified GPUs are available")
        return False

    return True


if __name__ == "__main__":
    args = cli()
    main(args)
