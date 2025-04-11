"""Match GBIF and PFT data and save to disk."""

from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from box import ConfigBox
from distributed import Client

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.trait_utils import clean_species_name


def main(cfg: ConfigBox = get_config()):
    """Match GBIF and PFT data and save to disk."""
    syscfg = cfg[detect_system()][cfg.model_res]["match_gbif_pfts"]
    # 00. Initialize Dask client

    # 01. Load data
    pft_path = Path(cfg.raw_dir, cfg.trydb.raw.pfts)
    pft_columns = ["AccSpeciesName", "pft"]
    log.info(f"Loading PFT data from {pft_path}...")
    if pft_path.suffix == ".csv":
        pfts = pd.read_csv(
            Path(cfg.raw_dir, cfg.trydb.raw.pfts),
            encoding="latin-1",
            usecols=pft_columns,
        )
    elif pft_path.suffix == ".parquet":
        pfts = pd.read_parquet(
            Path(cfg.raw_dir, cfg.trydb.raw.pfts), columns=pft_columns
        )
    else:
        raise ValueError(f"Unsupported PFT file format: {pft_path.suffix}")

    pfts = pfts.astype(
        {
            "AccSpeciesName": "string[pyarrow]",
            "pft": "category",
        }
    )

    gbif_raw_dir = Path(cfg.raw_dir, cfg.gbif.raw.dir)
    gbif_prep_dir = Path(cfg.interim_dir, cfg.gbif.interim.dir)
    gbif_prep_dir.mkdir(parents=True, exist_ok=True)
    gbif_columns = [
        "species",
        "taxonrank",
        "decimallatitude",
        "decimallongitude",
        "occurrencestatus",
    ]

    with Client(dashboard_address=cfg.dask_dashboard, n_workers=syscfg.n_workers):
        log.info(f"Loading GBIF data from {gbif_raw_dir}...")
        gbif = dd.read_parquet(
            gbif_raw_dir / "all_tracheophyta_non-cult_2024-04-10.parquet/*",
            columns=gbif_columns,
        ).astype(
            {
                "species": "string[pyarrow]",
                "taxonrank": "category",
                "decimallatitude": "float64[pyarrow]",
                "decimallongitude": "float64[pyarrow]",
                "occurrencestatus": "category",
            }
        )

        # 02. Preprocess GBIF data
        gbif = (
            gbif.dropna(subset=["species"])
            .query("taxonrank == 'SPECIES' and occurrencestatus == 'PRESENT'")
            .drop(columns=["taxonrank", "occurrencestatus"])
            .pipe(clean_species_name, "species", "speciesname")
            .drop(columns=["species"])
            .sort_values(by="speciesname")
            .set_index("speciesname")
        )

        # 03. Preprocess PFT data
        pfts = (
            pfts.dropna(subset=["AccSpeciesName"])
            .pipe(clean_species_name, "AccSpeciesName", "speciesname")
            .drop(columns=["AccSpeciesName"])
            .sort_values(by="speciesname")
            .drop_duplicates(subset=["speciesname"])
            .set_index("speciesname")
        )

        log.info("Matching GBIF and PFT data and saving to disk...")
        # 04. Merge GBIF and PFT data and save to disk
        gbif = (
            gbif.join(pfts, how="inner")
            .reset_index()
            .to_parquet(gbif_prep_dir / cfg.gbif.interim.matched, write_index=False)
        )


if __name__ == "__main__":
    main()
    log.info("Done!")
