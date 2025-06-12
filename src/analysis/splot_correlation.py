"""Calculate Pearson correlation coefficient between sPlot and GBIF sparse CWM trait
grids."""

from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import get_trait_map_fns
from src.utils.raster_utils import open_raster


def process_pair(
    splot_fn: Path, gbif_fn: Path, resolution: str
) -> tuple[str, str, float]:
    """Process a single pair of sPlot and GBIF files to calculate correlation.

    Args:
        splot_fn: Path to sPlot file
        gbif_fn: Path to GBIF file
        resolution: Model resolution

    Returns:
        Tuple containing (trait_id, resolution, correlation)
    """
    assert splot_fn.stem == gbif_fn.stem, f"{splot_fn.stem} != {gbif_fn.stem}"
    log.info("Processing trait %s...", splot_fn.stem)

    splot_map = open_raster(splot_fn).sel(band=1)
    gbif_map = open_raster(gbif_fn).sel(band=1)

    splot_df = (
        splot_map.to_dataframe(name=f"{splot_fn.stem}")
        .drop(columns=["band", "spatial_ref"])
        .dropna()
    )
    gbif_df = (
        gbif_map.to_dataframe(name=f"{gbif_fn.stem}")
        .drop(columns=["band", "spatial_ref"])
        .dropna()
    )

    correlation = splot_df.corrwith(gbif_df).iloc[0]
    log.info("✓ Trait %s correlation: %.3f", splot_fn.stem, correlation)
    return (splot_fn.stem, resolution, correlation)


def main() -> None:
    cfg = get_config()
    log.info("Starting correlation calculation for resolution %s...", cfg.model_res)

    splot_fns = get_trait_map_fns("splot", cfg)
    gbif_fns = get_trait_map_fns("gbif", cfg)
    log.info("Found %d trait pairs to process", len(splot_fns))

    # Process files in parallel
    results = Parallel(n_jobs=1)(
        delayed(process_pair)(splot_fn, gbif_fn, cfg.model_res)
        for splot_fn, gbif_fn in zip(splot_fns, gbif_fns)
    )

    corr_df_this_res = pd.DataFrame(
        list(results), columns=["trait_id", "resolution", "pearsonr"]
    )
    log.info("Calculated correlations for %d traits", len(corr_df_this_res))

    corr_results_fn = Path("results", cfg.datasets.Y.correlation_fn)
    log.info("Writing results to %s", corr_results_fn)

    if corr_results_fn.exists():
        corr_df = pd.read_csv(corr_results_fn)
        corr_df = pd.concat(
            [corr_df, corr_df_this_res], ignore_index=True
        ).drop_duplicates(subset=["trait_id", "resolution"])
    else:
        corr_df = corr_df_this_res

    corr_df.to_csv(corr_results_fn, index=False)
    log.info("✓ Results written successfully")


if __name__ == "__main__":
    main()
