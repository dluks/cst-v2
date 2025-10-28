"""
Calculates spatial autocorrelation for each trait in a feature set using GPU
acceleration.
"""

import argparse
import logging
import shutil
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
import utm
from box import ConfigBox
from dask import compute, delayed
from pyproj import Transformer
from torch.optim.adam import Adam

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import get_autocorr_ranges_fn, get_y_fn
from src.utils.trait_utils import get_active_traits


def cli() -> argparse.Namespace:
    """Command line interface for calculating spatial autocorrelation."""
    parser = argparse.ArgumentParser(
        description="Calculate spatial autocorrelation for traits using GPU "
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
    return parser.parse_args()


# Available GPUs
GPU_DEVICES = []  # Will be populated from syscfg
CURRENT_GPU_IDX = 0


def main(args: argparse.Namespace) -> None:
    """Main function for calculating spatial autocorrelation."""
    # Load config with params if provided
    params_path = Path(args.params).absolute() if args.params else None
    log.info("Loading config from %s", params_path or "default")
    cfg = get_config(params_path)

    syscfg = cfg[detect_system()]["calc_spatial_autocorr"]

    if args.debug:
        log.info("Running in debug mode...")
        log.setLevel(logging.DEBUG)

    # Check if output already exists and handle overwrite
    ranges_fn_dvc = get_autocorr_ranges_fn(cfg)
    if ranges_fn_dvc.exists() and not args.overwrite:
        log.info("Output file already exists: %s", ranges_fn_dvc)
        log.info("Use --overwrite flag to overwrite existing files.")
        return

    if cfg.calc_spatial_autocorr.use_existing:
        copy_ref_to_dvc(cfg)
        return

    # Set GPU devices from configuration
    if hasattr(syscfg, "gpu_ids"):
        set_gpu_devices(syscfg.gpu_ids)
        log.info(f"Set GPU devices from config: {GPU_DEVICES}")
    else:
        log.warning("No GPU IDs specified in config, defaulting to [0]")
        set_gpu_devices([0])

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
    valid_cols = get_active_traits(cfg)
    y_ddf = (
        dd.read_parquet(y_fn, columns=["source", *valid_cols])
        .query("source == 's'")
        .drop(columns=["source"])
        .reset_index()  # x and y are in the index
    )
    y_cols = y_ddf.columns.difference(["x", "y"]).to_list()

    log.info(
        "Reprojecting from %s to Oblique Azimuthal Equidistant (OAE) projection...",
        cfg.crs,
    )
    proj_str = "+proj=aeqd +lon_0=0 +lat_0=0 +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs(cfg.crs, proj_str, always_xy=True)
    x, y = transformer.transform(y_ddf.x.values.compute(), y_ddf.y.values.compute())
    orig_xy = y_ddf[["x", "y"]].copy()
    y_ddf = y_ddf.assign(x=x, y=y)

    vgram_kwargs = {"n_max": 18000, "nlags": 50, "log_binning": True}

    results = [
        _single_trait_ranges(
            (
                y_ddf[["x", "y", trait_col]]
                .astype(np.float32)
                .dropna(subset=[trait_col])
                .reset_index(drop=True)
            ),
            trait_col,
            cfg,
            syscfg,
            vgram_kwargs,
        )
        for trait_col in y_cols
    ]

    log.info("Computing range statistics for all traits...")
    ranges_df = pd.concat(compute(*results), ignore_index=True).assign(  # pyright: ignore[reportArgumentType]
        x=orig_xy.x, y=orig_xy.y
    )

    close_dask(client)

    log.info("Saving range statistics to DataFrame...")
    # Path to be checked into DVC
    ranges_fn_dvc = get_autocorr_ranges_fn(cfg)

    # Path to be used as reference when computing ranges for other resolutions.
    # Tracked with git.
    trait_stat = cfg.datasets.Y.trait_stats[cfg.datasets.Y.trait_stat - 1]
    ranges_fn_ref = Path(
        "reference",
        f"{ranges_fn_dvc.stem}_{cfg.PFT}_{cfg.model_res}_{trait_stat}{ranges_fn_dvc.suffix}",
    )

    log.info("Saving spatial autocorrelation ranges to %s...", ranges_fn_dvc)
    if ranges_fn_dvc.exists():
        log.info("Overwriting existing spatial autocorrelation ranges...")
        ranges_fn_dvc.unlink()

    ranges_df.to_parquet(ranges_fn_dvc)
    shutil.copy(ranges_fn_dvc, ranges_fn_ref)


def set_gpu_devices(gpu_ids):
    """Set the available GPU devices from configuration."""
    global GPU_DEVICES
    GPU_DEVICES = gpu_ids
    log.info(f"Using GPU devices: {GPU_DEVICES}")


def get_next_gpu() -> int:
    """Returns the next GPU ID in round-robin fashion."""
    global CURRENT_GPU_IDX, GPU_DEVICES
    if not GPU_DEVICES:
        log.warning("No GPU devices configured, defaulting to device 0")
        return 0

    gpu_id = GPU_DEVICES[CURRENT_GPU_IDX]
    CURRENT_GPU_IDX = (CURRENT_GPU_IDX + 1) % len(GPU_DEVICES)
    return gpu_id


@delayed
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


def _transform_coords_to_wgs84(
    coords_tensor: torch.Tensor, crs: str, device: torch.device
) -> torch.Tensor:
    """
    Transform coordinates from the given CRS to WGS84 (lat/lon).

    Args:
        coords_tensor (torch.Tensor): Coordinates tensor
        crs (str): Source coordinate reference system
        device (torch.device): PyTorch device

    Returns:
        torch.Tensor: Transformed coordinates in WGS84
    """

    if crs == "EPSG:4326":
        return coords_tensor

    # Create transformer on CPU
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # Convert to numpy, transform, then back to tensor
    coords_np = coords_tensor.cpu().numpy()
    lon, lat = transformer.transform(coords_np[:, 0], coords_np[:, 1])

    # Replace coords_tensor with lat/lon coordinates
    return torch.tensor(np.column_stack([lon, lat]), dtype=torch.float32, device=device)


def _estimate_max_distance(coords_tensor: torch.Tensor, n_samples: int) -> float:
    """
    Estimate maximum distance from a subset of data points.

    Args:
        coords_tensor (torch.Tensor): Coordinates tensor
        n_samples (int): Total number of samples

    Returns:
        float: Estimated maximum distance in meters
    """
    with torch.no_grad():
        # Estimate max distance from a subset of data
        subset_size = min(1000, n_samples)
        indices = torch.randperm(n_samples)[:subset_size]
        subset_coords = coords_tensor[indices]

        # Calculate haversine distances for the subset
        lons1 = subset_coords[:, 0].unsqueeze(1)  # [subset_size, 1]
        lats1 = subset_coords[:, 1].unsqueeze(1)  # [subset_size, 1]
        lons2 = subset_coords[:, 0].unsqueeze(0)  # [1, subset_size]
        lats2 = subset_coords[:, 1].unsqueeze(0)  # [1, subset_size]

        # Use the existing haversine distance function
        subset_dists = _calculate_haversine_distance(lons1, lats1, lons2, lats2)

        return float(subset_dists.max().item() * 0.7)  # Use 70% of max distance


def _calculate_haversine_distance(
    lons1: torch.Tensor, lats1: torch.Tensor, lons2: torch.Tensor, lats2: torch.Tensor
) -> torch.Tensor:
    """
    Calculate haversine distances between two sets of coordinates.

    Args:
        lons1 (torch.Tensor): First set of longitudes
        lats1 (torch.Tensor): First set of latitudes
        lons2 (torch.Tensor): Second set of longitudes
        lats2 (torch.Tensor): Second set of latitudes

    Returns:
        torch.Tensor: Haversine distances in kilometers
    """
    # Convert to radians
    lons1, lats1 = torch.deg2rad(lons1), torch.deg2rad(lats1)
    lons2, lats2 = torch.deg2rad(lons2), torch.deg2rad(lats2)

    # Haversine formula
    dlon = lons2 - lons1
    dlat = lats2 - lats1
    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lats1) * torch.cos(lats2) * torch.sin(dlon / 2) ** 2
    )
    return 2 * 6371000 * torch.asin(torch.sqrt(a))  # Earth radius in m


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

    Args:
        coords_tensor (torch.Tensor): Coordinates tensor
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

    # Process data in chunks
    for i in range(0, n_samples, chunk_size):
        i_end = min(i + chunk_size, n_samples)
        chunk_coords = coords_tensor[i:i_end]
        chunk_values = values_tensor[i:i_end]

        for j in range(0, n_samples, chunk_size):
            j_end = min(j + chunk_size, n_samples)

            # Get coordinates for both chunks
            lons1 = chunk_coords[:, 0].unsqueeze(1)  # [chunk_size, 1]
            lats1 = chunk_coords[:, 1].unsqueeze(1)  # [chunk_size, 1]
            lons2 = coords_tensor[j:j_end, 0].unsqueeze(0)  # [1, chunk_size]
            lats2 = coords_tensor[j:j_end, 1].unsqueeze(0)  # [1, chunk_size]

            # Calculate haversine distances
            dists = _calculate_haversine_distance(lons1, lats1, lons2, lats2)

            # Calculate pairwise semivariances
            v1 = chunk_values.unsqueeze(1)
            v2 = values_tensor[j:j_end].unsqueeze(0)
            sv = 0.5 * (v1 - v2) ** 2

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
) -> tuple[float, float, float, float]:
    """
    Fit a spherical variogram model to the experimental variogram.

    Args:
        bin_centers (torch.Tensor): Centers of distance bins
        gamma (torch.Tensor): Experimental variogram values
        valid_lags (torch.Tensor): Mask of valid lags
        device (torch.device): PyTorch device

    Returns:
        tuple[float, float, float, float]: Range, nugget, sill, and MSE
    """
    with torch.no_grad():
        # Initial parameter estimates
        nugget_init = (
            gamma[valid_lags][0].item() if gamma[valid_lags].shape[0] > 0 else 0.0
        )
        sill_init = gamma[valid_lags][-5:].mean().item() - nugget_init
        range_init = bin_centers[valid_lags][-1].item() * 0.6

    # Fit parameters using optimization
    params = torch.tensor(
        [nugget_init, sill_init, range_init], device=device, requires_grad=True
    )

    optimizer = Adam([params], lr=0.01)

    bin_centers_valid = bin_centers[valid_lags]
    gamma_valid = gamma[valid_lags]

    for _ in range(500):
        optimizer.zero_grad()
        nugget, sill, range_param = params

        # Ensure parameters are positive
        nugget_pos = torch.nn.functional.softplus(nugget)
        sill_pos = torch.nn.functional.softplus(sill)
        range_pos = torch.nn.functional.softplus(range_param)

        # Calculate predicted values using spherical model
        pred = spherical_model(bin_centers_valid, nugget_pos, sill_pos, range_pos)

        # Calculate loss (MSE)
        loss = torch.mean((pred - gamma_valid) ** 2)
        loss.backward()
        optimizer.step()

    # Get final parameters
    with torch.no_grad():
        nugget, sill, range_param = params
        nugget = torch.nn.functional.softplus(nugget).item()
        sill = torch.nn.functional.softplus(sill).item()
        range_val = torch.nn.functional.softplus(range_param).item()

        # Calculate final MSE
        pred = spherical_model(bin_centers_valid, nugget, sill, range_val)
        mse = torch.mean((pred - gamma_valid) ** 2).item()

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
    Fit a spherical variogram model using GPU acceleration with haversine distances.

    Args:
        coords (np.ndarray): Coordinates array of shape (n_samples, 2)
        values (np.ndarray): Values array of shape (n_samples,)
        nlags (int): Number of distance lags
        max_dist (float | None): Maximum distance to consider
        gpu_id (int): GPU device ID to use
        crs (str): Coordinate reference system of input coordinates
        log_binning (bool): Whether to use logarithmic binning

    Returns:
        Tuple[float, float, float, float]: Estimated range, nugget, sill, and MSE
    """
    # Set the GPU device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Convert to PyTorch tensors and move to GPU
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)

    # Transform coordinates to WGS84 (lat/lon) if needed
    coords_tensor = _transform_coords_to_wgs84(coords_tensor, crs, device)

    # Calculate pairwise distances
    n_samples = coords_tensor.shape[0]

    # Process in chunks to handle large datasets
    chunk_size = 30000

    # Initialize bins for the experimental variogram
    if max_dist is None:
        max_dist = _estimate_max_distance(coords_tensor, n_samples)

    # Create distance bins - either linear or logarithmic
    if log_binning:
        # Use logarithmic binning for adaptive bin sizes
        # Add a small value to avoid log(0)
        min_dist = max(
            1000.0, max_dist * 0.001
        )  # Minimum distance (1km or 0.1% of max)
        bin_edges = torch.logspace(
            torch.log10(torch.tensor(min_dist)),
            torch.log10(torch.tensor(max_dist)),
            nlags + 1,
            device=device,
        )
    else:
        # Use linear binning (original implementation)
        bin_edges = torch.linspace(0, max_dist, nlags + 1, device=device)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute experimental variogram
    gamma_sum, gamma_counts = _compute_experimental_variogram(
        coords_tensor, values_tensor, bin_edges, nlags, chunk_size, device
    )

    # Calculate mean semivariance for each bin
    with torch.no_grad():
        valid_lags = gamma_counts > 0
        gamma = torch.zeros_like(gamma_counts)
        gamma[valid_lags] = gamma_sum[valid_lags] / gamma_counts[valid_lags]

    # Fit variogram model
    return _fit_variogram_model(bin_centers, gamma, valid_lags, device)


@delayed
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
        group = group.sample(n_max)

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
        # Fit variogram using GPU with haversine distances
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
        log.info("Falling back to CPU calculation...")
        # Fallback to CPU calculation if GPU fails
        from pykrige.ok import OrdinaryKriging

        orig_kwargs = {
            "variogram_model": "spherical",
            "nlags": nlags,
            "anisotropy_scaling": 1,
            "anisotropy_angle": 0,
        }
        orig_kwargs.update(**kwargs)

        ok_vgram = OrdinaryKriging(
            group["easting"], group["northing"], group[data_col], **orig_kwargs
        )
        return ok_vgram.variogram_model_parameters[1], len(group)


def copy_ref_to_dvc(
    cfg: ConfigBox,
) -> None:
    """Copy the reference ranges file to the DVC-tracked location."""
    fn = (
        f"{Path(cfg.train.spatial_autocorr).stem}_"
        f"{cfg.calc_spatial_autocorr.use_existing}"
        f"{Path(cfg.train.spatial_autocorr).suffix}"
    )
    ranges_fn_ref = Path("reference", fn)
    log.info("Using existing spatial autocorrelation ranges from %s...", ranges_fn_ref)
    ranges_fn_dvc = get_autocorr_ranges_fn(cfg)

    if ranges_fn_dvc.exists():
        log.info("Overwriting existing spatial autocorrelation ranges...")
        ranges_fn_dvc.unlink()
    shutil.copy(ranges_fn_ref, ranges_fn_dvc)


# def transform_coords_to_equidistant(
#     coords: np.ndarray, from_crs: str, to_crs: str
# ) -> np.ndarray:
#     """
#     Transform coordinates to equidistant projection.

#     Args:
#         coords (np.ndarray): The coordinates to transform.

#     Returns:
#         np.ndarray: The transformed coordinates.
#     """
#     # Transform coordinates to equidistant projection
#     transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
#     return np.asarray(transformer.transform(coords[:, 0], coords[:, 1]))


def _assign_zones(df: pd.DataFrame, n_zones: int) -> pd.DataFrame:
    """
    Assigns zones to the DataFrame based on x and y coordinates.

    Args:
        df (pd.DataFrame): The DataFrame with x and y coordinates.
        n_sectors (int): The number of sectors to divide the data into.

    Returns:
        pd.DataFrame: The DataFrame with an additional 'zone' column.
    """
    df = df.assign(
        easting=df.x + abs(df.x.min()),
        northing=df.y + abs(df.y.min()),
    )

    x_bins = np.linspace(df.easting.min(), df.easting.max(), n_zones + 1)
    y_bins = np.linspace(df.northing.min(), df.northing.max(), n_zones // 2 + 1)

    x_zones = np.digitize(df.easting, x_bins) - 1
    y_zones = np.digitize(df.northing, y_bins) - 1

    df["zone"] = [f"{x}_{y}" for x, y in zip(x_zones, y_zones)]
    return df.drop(columns=["easting", "northing"])


@delayed
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

    if cfg.crs == "EPSG:4326":
        log.info("Adding UTM coordinates...")

        trait_df = add_utm(trait_df)

        vgram_kwargs["crs"] = "EPSG:4326"
        results = [
            calculate_variogram_gpu(group, trait_col, **vgram_kwargs)
            for _, group in trait_df.groupby("zone")
        ]

    elif cfg.crs == "EPSG:6933":
        vgram_kwargs["crs"] = "EPSG:6933"

        # Calculate global variogram first (with smaller sample)
        global_kwargs = vgram_kwargs.copy()
        global_kwargs["n_max"] = min(30000, len(trait_df))
        global_sample = (
            trait_df.sample(global_kwargs["n_max"], random_state=42)
            if len(trait_df) > global_kwargs["n_max"]
            else trait_df
        )
        global_result = calculate_variogram_gpu(
            global_sample, trait_col, **global_kwargs
        )

        # Then calculate by chunks for local patterns
        if syscfg.n_chunks > 1:
            log.info("Using latitude bands for more consistent results...")
            # Use latitude bands instead of arbitrary zones for more consistent results
            trait_df = trait_df.assign(
                lat_band=pd.cut(trait_df.y, bins=syscfg.n_chunks, labels=False)
            )

            # Set max_samples per chunk to balance representation
            samples_per_chunk = min(10000, len(trait_df) // syscfg.n_chunks)
            vgram_kwargs["n_max"] = samples_per_chunk

            results = [
                calculate_variogram_gpu(group, trait_col, **vgram_kwargs)
                for _, group in trait_df.groupby("lat_band")
            ]

            # Add global result to the chunk results
            results.append(global_result)
        else:
            results = [global_result]

    else:
        raise ValueError(f"Unknown CRS: {cfg.crs}")

    autocorr_ranges = list(compute(*results))
    if len(autocorr_ranges) == 1:
        global_range, global_n = autocorr_ranges[0]
    else:
        global_range, global_n = autocorr_ranges[-1]
        autocorr_ranges = autocorr_ranges[:-1]

    filt_ranges = [(r, n) for r, n in autocorr_ranges if n > 0]

    # Weight the ranges by the number of samples used to calculate them
    sample_sizes = np.array([n for _, n in filt_ranges])
    weights = sample_sizes / sample_sizes.sum()
    ranges = np.array([r for r, _ in filt_ranges])

    # Create a new row and append it to the DataFrame
    new_ranges = pd.DataFrame(
        [
            {
                "trait": trait_col,
                "mean": np.average(ranges, weights=weights),
                "std": np.sqrt(
                    np.average((ranges - ranges.mean()) ** 2, weights=weights)
                ),
                "median": np.median(ranges),
                "q05": np.quantile(ranges, 0.05),
                "q95": np.quantile(ranges, 0.95),
                "n": sample_sizes.sum(),
                "n_chunks": len(ranges),
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
