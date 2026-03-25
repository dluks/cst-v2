"""
Match sPlot data with filtered trait data, compute community-weighted statistics
per grid cell (dissolving plot boundaries), and write per-trait output to
GeoTIFF or Zarr.

Cell-level aggregation
----------------------
Observations from all plots landing in the same grid cell are pooled.  Each
(plot, species) row is weighted by ``Rel_Abund_Plot * weight`` — matching the
weighting scheme used by the histogram target builder in
``build_histogram_targets._process_splot``.  Statistics (CWM, std, quantiles)
are then computed once per cell from these pooled, weighted observations.
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
import zarr

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.df_utils import reproject_geo_to_xy
from src.utils.raster_utils import xr_to_raster
from src.utils.trait_utils import filter_pft


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build sPlot trait maps with cell-level community-weighted "
        "statistics. Output one file per trait in GeoTIFF or Zarr format."
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        help="Path to the parameters file.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "-t",
        "--trait",
        type=str,
        required=True,
        help="Trait name to process (e.g. 'X11', 'PC1', or 'gsmax').",
    )
    parser.add_argument(
        "-f",
        "--output-format",
        type=str,
        choices=["tif", "zarr"],
        default="tif",
        help="Output format: 'tif' (GeoTIFF) or 'zarr'. Default: tif.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace | None = None) -> None:
    """Main function."""
    args = cli() if args is None else args
    cfg = get_config(params_path=args.params)

    n_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "1")
    os.environ["GDAL_NUM_THREADS"] = n_cpus
    log.info("Setting GDAL_NUM_THREADS=%s", n_cpus)

    out_format = args.output_format
    out_dir = Path(cfg.splot.maps.out_dir, cfg.product_code)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = "zarr" if out_format == "zarr" else "tif"
    out_fn = out_dir / f"{args.trait}.{ext}"
    plot_fn = out_dir / f"{args.trait}_diagnostic.pdf"

    if out_fn.exists() and plot_fn.exists() and not args.overwrite:
        log.info("%s and diagnostic already exist. Skipping...", out_fn.name)
        return

    # ------------------------------------------------------------------
    # 1. Load & join
    # ------------------------------------------------------------------
    log.info("Loading filtered sPlot data...")
    splot_obs = _load_splot(
        Path(cfg.splot.filtered.out_dir, cfg.trait_type, cfg.splot.filtered.fp),
        cfg.PFT,
    )

    log.info("Loading trait data...")
    trait_df = _load_trait(Path(cfg.traits.interim_out), args.trait)

    log.info("Joining sPlot and trait data...")
    merged = splot_obs.merge(
        trait_df, left_on="speciesname", right_on="nameOutWCVP", how="inner"
    ).drop(columns=["nameOutWCVP"])

    if len(merged) == 0:
        log.error("No data after joining sPlot with trait %s. Skipping...", args.trait)
        return

    log.info("Merged %d observation rows for trait %s", len(merged), args.trait)

    # Drop rows with NaN trait values
    merged = merged.dropna(subset=[args.trait])

    # ------------------------------------------------------------------
    # 2. Reproject & assign grid cells
    # ------------------------------------------------------------------
    log.info("Reprojecting coordinates...")
    merged = _reproject(cfg.crs, merged)

    log.info("Assigning grid cell IDs at %dm resolution...", cfg.target_resolution)
    merged = _assign_cell_ids(merged, cfg.target_resolution)

    # Combined weight: relative abundance × survey weight
    # (consistent with build_histogram_targets._process_splot)
    merged["combined_weight"] = merged["Rel_Abund_Plot"] * merged["weight"]

    # ------------------------------------------------------------------
    # 3. Cell-level community-weighted statistics
    # ------------------------------------------------------------------
    log.info("Computing cell-level statistics for trait %s...", args.trait)
    cell_stats = _compute_cell_stats(merged, args.trait)
    log.info("Computed statistics for %d grid cells", len(cell_stats))

    # ------------------------------------------------------------------
    # 4. Write output
    # ------------------------------------------------------------------
    if out_fn.exists() and not args.overwrite:
        log.info("%s already exists, skipping data write.", out_fn.name)
    else:
        log.info("Writing to disk...")
        if out_format == "zarr":
            _write_zarr(cell_stats, out_fn, args.trait, cfg)
        else:
            _write_tif(cell_stats, out_fn, args.trait, cfg, int(n_cpus))

    # ------------------------------------------------------------------
    # 5. Diagnostic plot
    # ------------------------------------------------------------------
    if plot_fn.exists() and not args.overwrite:
        log.info("%s already exists, skipping plot.", plot_fn.name)
    else:
        log.info("Writing diagnostic plot to %s...", plot_fn.name)
        _write_diagnostic_plot(cell_stats, plot_fn, args.trait)

    log.info("Done!")


# ---------------------------------------------------------------------------
# Cell-level statistics (vectorized)
# ---------------------------------------------------------------------------


def _compute_cell_stats(df: pd.DataFrame, trait: str) -> pd.DataFrame:
    """Compute community-weighted statistics per grid cell.

    All (plot, species) rows within a cell are pooled and weighted by
    ``combined_weight = Rel_Abund_Plot * weight``.

    Returns a DataFrame with one row per cell and columns:
    cell_x, cell_y, cwm, cw_std, cw_med, cw_q02 … cw_q98, n_obs, total_weight.
    """
    g = df.groupby("cell_id")

    trait_vals = df[trait].values
    weights = df["combined_weight"].values

    # Weighted sum and weight sum per cell
    df["_tw"] = trait_vals * weights
    agg = g.agg(
        cell_x=("cell_x", "first"),
        cell_y=("cell_y", "first"),
        _sum_tw=("_tw", "sum"),
        _sum_w=("combined_weight", "sum"),
        n_obs=("combined_weight", "size"),
    )

    agg["cwm"] = agg["_sum_tw"] / agg["_sum_w"]

    # For std and quantiles we need per-cell iteration, but over cells
    # (thousands) not plots (hundreds of thousands) — much faster.
    stds = np.empty(len(agg), dtype=np.float64)
    quantile_levels = [0.02, 0.05, 0.25, 0.50, 0.75, 0.95, 0.98]
    quantiles = np.empty((len(agg), len(quantile_levels)), dtype=np.float64)

    cell_ids_ordered = agg.index.values
    cell_id_to_pos = {cid: i for i, cid in enumerate(cell_ids_ordered)}
    cwm_vals = agg["cwm"].values

    # Group once, iterate over cells
    for cell_id, grp in g:
        pos = cell_id_to_pos[cell_id]
        tv = grp[trait].values
        w = grp["combined_weight"].values
        w_norm = w / w.sum()

        # Weighted std
        mean = cwm_vals[pos]
        stds[pos] = np.sqrt(np.sum(w_norm * (tv - mean) ** 2))

        # Weighted quantiles
        order = np.argsort(tv)
        tv_sorted = tv[order]
        w_cumsum = np.cumsum(w_norm[order])
        for qi, q in enumerate(quantile_levels):
            idx = np.searchsorted(w_cumsum, q)
            idx = min(idx, len(tv_sorted) - 1)
            quantiles[pos, qi] = tv_sorted[idx]

    agg["cw_std"] = stds
    q_names = ["cw_q02", "cw_q05", "cw_q25", "cw_med", "cw_q75", "cw_q95", "cw_q98"]
    for i, name in enumerate(q_names):
        agg[name] = quantiles[:, i]

    agg["total_weight"] = agg["_sum_w"]
    agg = agg.drop(columns=["_sum_tw", "_sum_w"])

    return agg.reset_index()


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------


def _write_diagnostic_plot(
    cell_stats: pd.DataFrame, out_fn: Path, trait: str,
) -> None:
    """Write a diagnostic figure with spatial map + histogram for each stat layer."""
    stat_cols = [
        ("cwm", "CWM"),
        ("cw_std", "CW Std"),
        ("cw_med", "Median"),
        ("cw_q02", "Q02"),
        ("cw_q05", "Q05"),
        ("cw_q25", "Q25"),
        ("cw_q75", "Q75"),
        ("cw_q95", "Q95"),
        ("cw_q98", "Q98"),
        ("n_obs", "N obs"),
    ]

    n_rows = len(stat_cols)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows))

    # Subsample for plotting if too many cells (keeps plots fast at 1km resolution)
    max_plot_cells = 80_000
    if len(cell_stats) > max_plot_cells:
        rng = np.random.default_rng(42)
        plot_idx = rng.choice(len(cell_stats), size=max_plot_cells, replace=False)
        plot_df = cell_stats.iloc[plot_idx]
        subsample_note = f" (subsampled {max_plot_cells:,}/{len(cell_stats):,} cells)"
    else:
        plot_df = cell_stats
        subsample_note = ""

    x = plot_df["cell_x"].values
    y = plot_df["cell_y"].values

    for i, (col, label) in enumerate(stat_cols):
        vals = plot_df[col].values
        ax_map = axes[i, 0]
        ax_hist = axes[i, 1]

        # Spatial scatter map
        vmin, vmax = np.nanpercentile(vals, [2, 98])
        sc = ax_map.scatter(
            x, y, c=vals, s=0.5, cmap="viridis", vmin=vmin, vmax=vmax,
            rasterized=True,
        )
        ax_map.set_aspect("equal")
        ax_map.set_title(f"{label}", fontsize=10)
        ax_map.tick_params(labelsize=7)
        fig.colorbar(sc, ax=ax_map, fraction=0.046, pad=0.04)

        # Histogram
        finite = vals[np.isfinite(vals)]
        ax_hist.hist(finite, bins=150, color="#5e81ac", edgecolor="none", alpha=0.85)
        ax_hist.set_title(f"{label} distribution", fontsize=10)
        ax_hist.tick_params(labelsize=7)
        ax_hist.axvline(np.median(finite), color="#bf616a", linewidth=1, linestyle="--",
                        label=f"median={np.median(finite):.3g}")
        ax_hist.legend(fontsize=7)

    fig.suptitle(f"Trait {trait} — {len(cell_stats):,} grid cells{subsample_note}", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_fn, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote diagnostic plot: %s", out_fn.name)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_zarr(
    cell_stats: pd.DataFrame, out_fn: Path, trait: str, cfg: object,
) -> None:
    """Write cell statistics to a Zarr store."""
    store = zarr.open_group(str(out_fn), mode="w")

    coords = cell_stats[["cell_x", "cell_y"]].values.astype(np.float64)
    store.create_array("coords", data=coords)

    stat_cols = ["cwm", "cw_std", "cw_med", "cw_q02", "cw_q05",
                 "cw_q25", "cw_q75", "cw_q95", "cw_q98"]
    for col in stat_cols:
        store.create_array(col, data=cell_stats[col].values.astype(np.float32))

    store.create_array("n_obs", data=cell_stats["n_obs"].values.astype(np.int32))
    store.create_array("total_weight", data=cell_stats["total_weight"].values.astype(np.float32))

    store.attrs["trait"] = trait
    store.attrs["crs"] = cfg.crs
    store.attrs["resolution"] = cfg.target_resolution
    store.attrs["n_cells"] = len(cell_stats)
    store.attrs["stat_names"] = stat_cols

    log.info("Wrote %s (%d cells)", out_fn.name, len(cell_stats))


def _write_tif(
    cell_stats: pd.DataFrame, out_fn: Path, trait: str, cfg: object, n_cpus: int,
) -> None:
    """Write cell statistics to a multi-band GeoTIFF via rasterize_points."""
    from src.utils.df_utils import rasterize_points

    stat_cols = ["cwm", "cw_std", "cw_med", "cw_q02", "cw_q05",
                 "cw_q25", "cw_q75", "cw_q95", "cw_q98"]
    stat_names = ["mean", "std", "median", "q02", "q05", "q25", "q75", "q95", "q98"]

    # Build a DataFrame with x, y and all stat columns for rasterization
    raster_df = cell_stats[["cell_x", "cell_y", *stat_cols]].copy()
    raster_df = raster_df.rename(columns={"cell_x": "x", "cell_y": "y"})

    grids = []
    for stat_col, stat_name in zip(stat_cols, stat_names):
        ds = rasterize_points(
            raster_df[["x", "y", stat_col]],
            data_cols=stat_col,
            res=cfg.target_resolution,
            crs=cfg.crs,
            agg=False,
        )
        ds = ds.rename({stat_col: stat_name})
        grids.append(ds)

    # Add count
    count_df = raster_df[["x", "y"]].copy()
    count_df["n_obs"] = cell_stats["n_obs"].values
    ds_count = rasterize_points(
        count_df,
        data_cols="n_obs",
        res=cfg.target_resolution,
        crs=cfg.crs,
        agg=False,
    )
    ds_count = ds_count.rename({"n_obs": "count"})
    grids.append(ds_count)

    gridded = xr.merge(grids)
    xr_to_raster(gridded, out_fn, num_threads=n_cpus)
    log.info("Wrote %s", out_fn.name)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_splot(fp: Path, pfts: list[str]) -> pd.DataFrame:
    """Load filtered sPlot data and filter by PFT."""
    splot = pd.read_parquet(fp).pipe(filter_pft, pfts).drop(columns=["pft"])
    return splot


def _load_trait(fp: Path, trait_name: str) -> pd.DataFrame:
    """Load trait data for a single trait."""
    trait_df = pd.read_parquet(fp, columns=["nameOutWCVP", trait_name])
    return trait_df


def _reproject(crs: str, df: pd.DataFrame) -> pd.DataFrame:
    """Reproject coordinates if necessary."""
    if crs != "EPSG:4326":
        if crs != "EPSG:6933":
            raise ValueError(f"Unsupported CRS: {crs}")

        df = reproject_geo_to_xy(
            df,
            to_crs=crs,
            x="Longitude",
            y="Latitude",
        ).drop(columns=["Latitude", "Longitude"])
    else:
        df = df.rename(columns={"Longitude": "x", "Latitude": "y"})

    return df


def _assign_cell_ids(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """Assign grid cell IDs based on coordinates and resolution.

    Matches the cell assignment logic in ``build_histogram_targets``.
    """
    cx = (df["x"].values // resolution).astype(np.int64)
    cy = (df["y"].values // resolution).astype(np.int64)
    df["cell_x"] = cx * resolution
    df["cell_y"] = cy * resolution
    df["cell_id"] = cx * 10_000_000 + cy
    return df


if __name__ == "__main__":
    main()
