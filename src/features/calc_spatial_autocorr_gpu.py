"""Calculates spatial autocorrelation for each trait in a feature set using GPU acceleration."""

import shutil
from pathlib import Path
from typing import Optional

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import utm
from box import ConfigBox
from dask import compute, delayed

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import get_autocorr_ranges_fn, get_y_fn

# Available GPUs
GPU_DEVICES = []  # Will be populated from syscfg
CURRENT_GPU_IDX = 0


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
    df["easting"] = [e for result in results for e in result[0]]
    df["northing"] = [n for result in results for n in result[1]]
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


def fit_variogram_gpu(
    coords: np.ndarray,
    values: np.ndarray,
    nlags: int = 30,
    max_dist: Optional[float] = None,
    gpu_id: int = 0,
) -> tuple[float, float, float, float]:
    """
    Fit a spherical variogram model using GPU acceleration.

    Args:
        coords (np.ndarray): Coordinates array of shape (n_samples, 2)
        values (np.ndarray): Values array of shape (n_samples,)
        nlags (int): Number of distance lags
        max_dist (Optional[float]): Maximum distance to consider
        gpu_id (int): GPU device ID to use

    Returns:
        Tuple[float, float, float, float]: Estimated range, nugget, sill, and MSE
    """
    # Set the GPU device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Convert to PyTorch tensors and move to GPU
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)

    # Calculate pairwise distances
    n_samples = coords_tensor.shape[0]

    # Process in chunks to handle large datasets
    chunk_size = 30000

    # Initialize bins for the experimental variogram
    if max_dist is None:
        with torch.no_grad():
            # Estimate max distance from a subset of data
            subset_size = min(1000, n_samples)
            indices = torch.randperm(n_samples)[:subset_size]
            subset_coords = coords_tensor[indices]

            # Calculate pairwise distances for the subset
            x1 = subset_coords.unsqueeze(1)
            x2 = subset_coords.unsqueeze(0)
            subset_dists = torch.sqrt(torch.sum((x1 - x2) ** 2, dim=2))

            max_dist = float(subset_dists.max().item() * 0.7)  # Use 70% of max distance

    # Create distance bins
    bin_edges = torch.linspace(0, max_dist, nlags + 1, device=device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

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

            # Calculate pairwise distances between chunks
            x1 = chunk_coords.unsqueeze(1)
            x2 = coords_tensor[j:j_end].unsqueeze(0)
            dists = torch.sqrt(torch.sum((x1 - x2) ** 2, dim=2))

            # Calculate pairwise semivariances
            v1 = chunk_values.unsqueeze(1)
            v2 = values_tensor[j:j_end].unsqueeze(0)
            sv = 0.5 * (v1 - v2) ** 2

            # Bin the semivariances by distance
            for lag in range(nlags):
                mask = (dists >= bin_edges[lag]) & (dists < bin_edges[lag + 1])
                gamma_sum[lag] += sv[mask].sum()
                gamma_counts[lag] += mask.sum()

    # Calculate mean semivariance for each bin
    with torch.no_grad():
        valid_lags = gamma_counts > 0
        gamma = torch.zeros_like(gamma_counts)
        gamma[valid_lags] = gamma_sum[valid_lags] / gamma_counts[valid_lags]

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

    optimizer = torch.optim.Adam([params], lr=0.01)

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
    if not isinstance(group, pd.DataFrame) or len(group) < 200:
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
    coords = group[["easting", "northing"]].values.astype(np.float32)
    values = group[data_col].values.astype(np.float32)

    # Get next available GPU
    gpu_id = get_next_gpu()

    try:
        # Fit variogram using GPU
        range_param, _, _, _ = fit_variogram_gpu(
            coords=coords, values=values, nlags=nlags, gpu_id=gpu_id
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


def copy_ref_to_dvc(cfg: ConfigBox) -> None:
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
    if cfg.target_resolution > 0.2 and cfg.crs == "EPSG:4326":
        log.info(
            "Target resolution of > 0.2 deg detected. Using web mercator "
            "coordinates instead of UTM zones..."
        )
        trait_df_wmerc = (
            gpd.GeoDataFrame(
                trait_df.drop(columns=["x", "y"]),
                geometry=gpd.points_from_xy(trait_df.x, trait_df.y),
                crs="EPSG:4326",
            )
            .to_crs("EPSG:3857")
            .pipe(
                lambda _df: _df.assign(
                    easting=_df.geometry.x, northing=_df.geometry.y
                ).drop(columns=["geometry"])
            )
        )

        results = [calculate_variogram_gpu(trait_df_wmerc, trait_col, **vgram_kwargs)]

    elif cfg.crs == "EPSG:4326":
        log.info("Adding UTM coordinates...")

        trait_df = add_utm(trait_df).drop(columns=["x", "y"])

        results = [
            calculate_variogram_gpu(group, trait_col, **vgram_kwargs)
            for _, group in trait_df.groupby("zone")
        ]

    elif cfg.crs == "EPSG:6933":

        def _assign_zones(df: pd.DataFrame, n_zones: int) -> pd.DataFrame:
            """
            Assigns zones to the DataFrame based on x and y coordinates.

            Args:
                df (pd.DataFrame): The DataFrame with x and y coordinates.
                n_sectors (int): The number of sectors to divide the data into.

            Returns:
                pd.DataFrame: The DataFrame with an additional 'zone' column.
            """

            x_bins = np.linspace(df.easting.min(), df.easting.max(), n_zones + 1)
            y_bins = np.linspace(df.northing.min(), df.northing.max(), n_zones // 2 + 1)

            x_zones = np.digitize(df.easting, x_bins) - 1
            y_zones = np.digitize(df.northing, y_bins) - 1

            df["zone"] = [f"{x}_{y}" for x, y in zip(x_zones, y_zones)]
            return df

        # Convert x and y to easting and northing (simple shift) as EPSG:6933 contains
        # negative x and y coordinates
        trait_df = trait_df.assign(
            easting=trait_df.x + abs(trait_df.x.min()),
            northing=trait_df.y + abs(trait_df.y.min()),
        ).drop(columns=["x", "y"])

        if syscfg.n_chunks > 1:
            trait_df_grouped = trait_df.pipe(
                _assign_zones, n_zones=syscfg.n_chunks
            ).groupby("zone")

            vgram_kwargs["n_max"] = len(trait_df)
            results = [
                calculate_variogram_gpu(group, trait_col, **vgram_kwargs)
                for _, group in trait_df_grouped
            ]
        else:
            vgram_kwargs["n_max"] = len(trait_df)

            results = [calculate_variogram_gpu(trait_df, trait_col, **vgram_kwargs)]

    else:
        raise ValueError(f"Unknown CRS: {cfg.crs}")

    autocorr_ranges = list(compute(*results))

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


def main(cfg: ConfigBox = get_config()) -> None:
    """Main function for calculating spatial autocorrelation."""
    syscfg = cfg[detect_system()][cfg.model_res]["calc_spatial_autocorr"]

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
    y_ddf = dd.read_parquet(y_fn).query("source == 's'").drop(columns=["source"])
    y_cols = y_ddf.columns.difference(["x", "y"]).to_list()

    vgram_kwargs = {
        "n_max": 18000,
        "nlags": 30,
    }

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
    ranges_df = pd.concat(compute(*results), ignore_index=True)  # type: ignore

    close_dask(client)

    log.info("Saving range statistics to DataFrame...")
    # Path to be checked into DVC
    ranges_fn_dvc = get_autocorr_ranges_fn(cfg)

    # Path to be used as reference when computing ranges for other resolutions.
    # Tracked with git.
    ranges_fn_ref = Path(
        "reference",
        f"{ranges_fn_dvc.stem}_{cfg.PFT}_{cfg.model_res}{ranges_fn_dvc.suffix}",
    )

    ranges_df.to_parquet(ranges_fn_dvc)
    ranges_df.to_parquet(ranges_fn_ref)


if __name__ == "__main__":
    main()
