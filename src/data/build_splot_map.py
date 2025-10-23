"""
Match sPlot data with filtered trait data, calculate community-weighted statistics
per plot, grid it, and write each trait's corresponding raster stack to GeoTIFF files.
"""

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.df_utils import rasterize_points, reproject_geo_to_xy
from src.utils.raster_utils import xr_to_raster
from src.utils.trait_aggregation_utils import cw_stats
from src.utils.trait_utils import filter_pft


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Match sPlot data with trait data, calculate community-weighted "
        "statistics, grid it, and write each trait's corresponding raster stack to "
        "GeoTIFF files."
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
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """Main function."""
    args = cli() if args is None else args
    cfg = get_config(params_path=args.params)

    out_dir = Path(cfg.splot.maps.out_dir, cfg.product_code)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_fn = out_dir / f"{args.trait}.tif"
    if out_fn.exists() and not args.overwrite:
        log.info("%s.tif already exists. Skipping...", args.trait)
        return

    # Load filtered sPlot data
    log.info("Loading filtered sPlot data...")
    splot_obs = _load_splot(
        Path(cfg.splot.filtered.out_dir, cfg.trait_type, cfg.splot.filtered.fp),
        cfg.PFT,
    )

    log.info("Loading trait data...")
    trait_df = _load_trait(Path(cfg.traits.interim_out), args.trait)

    log.info("Joining sPlot and trait data...")
    splot_traits = splot_obs.merge(
        trait_df, left_on="speciesname", right_on="nameOutWCVP", how="inner"
    ).drop(columns=["nameOutWCVP"])

    if len(splot_traits) == 0:
        log.error("No data after joining sPlot with trait %s. Skipping...", args.trait)
        return

    log.info("Calculating community-weighted statistics per plot...")
    plot_stats = (
        splot_traits.groupby("PlotObservationID")
        .apply(
            lambda g: cw_stats(g, args.trait, "Rel_Abund_Plot"),
            include_groups=False,
        )
        .reset_index()
    )

    # Merge with coordinates and weights
    plot_stats = plot_stats.merge(
        splot_traits[
            ["PlotObservationID", "Latitude", "Longitude", "weight"]
        ].drop_duplicates(subset=["PlotObservationID"]),
        on="PlotObservationID",
        how="left",
    )

    log.info("Reprojecting coordinates...")
    plot_stats = _reproject(cfg.crs, plot_stats)

    log.info("Processing trait %s...", args.trait)

    # Rasterize each statistic
    stat_cols = ["cwm", "cw_std", "cw_med", "cw_q05", "cw_q95", "cw_range"]
    stat_names = ["mean", "std", "median", "q05", "q95", "range"]

    log.info("Rasterizing statistics...")
    grids = []
    for stat_col, stat_name in zip(stat_cols, stat_names):
        log.info("Rasterizing %s...", stat_col)

        # Determine which statistics to compute
        requested_funcs = ["mean"]

        # Add count on the last statistic
        if stat_col == "cw_range":
            requested_funcs.append("count")
            requested_funcs.append("count_weighted")

        # Select columns for rasterization
        raster_df = pd.DataFrame(plot_stats[["x", "y", stat_col, "weight"]])

        ds = rasterize_points(
            raster_df,
            data_cols=stat_col,
            res=cfg.target_resolution,
            crs=cfg.crs,
            agg=True,
            funcs=requested_funcs,
            weights="weight",
        )

        # Rename 'mean' to the statistic name (except for cwm which stays as 'mean')
        if stat_col != "cwm":
            ds = ds.rename({"mean": stat_name})

        grids.append(ds)
        log.info("✓ %s rasterized", stat_col)

    # Merge all of the gridded data into a single dataset
    log.info("Merging gridded data...")
    gridded_trait = xr.merge(grids)
    log.info("Gridded data merged")

    log.info("Writing to disk...")
    xr_to_raster(gridded_trait, out_fn)
    log.info("Wrote %s.tif.", args.trait)

    log.info("Done!")


def _load_splot(fp: Path, pfts: list[str]) -> pd.DataFrame:
    """Load filtered sPlot data and filter by PFT."""
    splot = pd.read_parquet(fp).pipe(filter_pft, pfts).drop(columns=["pft"])
    return splot


def _load_trait(fp: Path, trait_name: str) -> pd.DataFrame:
    """Load trait data for a single trait."""
    trait_df = pd.read_parquet(fp, columns=["nameOutWCVP", trait_name])
    return trait_df


def _reproject(crs: str, splot_traits: pd.DataFrame) -> pd.DataFrame:
    """Reproject coordinates if necessary."""
    if crs != "EPSG:4326":
        if crs != "EPSG:6933":
            raise ValueError(f"Unsupported CRS: {crs}")

        splot_traits = reproject_geo_to_xy(
            splot_traits,
            to_crs=crs,
            x="Longitude",
            y="Latitude",
        ).drop(columns=["Latitude", "Longitude"])
    else:
        # Rename columns to x and y for consistency
        splot_traits = splot_traits.rename(columns={"Longitude": "x", "Latitude": "y"})

    return splot_traits


if __name__ == "__main__":
    main()
