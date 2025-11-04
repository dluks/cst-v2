"""Utility functions for cleaning and processing trait data."""

import json
import re
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dask_utils import repartition_if_set

# from src.utils.dataset_utils import get_try_traits_interim_fn


def genus_species_caps(col: pd.Series) -> pd.Series:
    """
    Converts the values in the given pandas Series to a format where the genus is
    capitalized and the species is lowercase.
    """
    col = col.str.title()

    return (
        col.str.split()
        .map(lambda x: x[0] + " " + x[1].lower())
        .astype("string[pyarrow]")
    )


def trim_species_name(col: pd.Series) -> pd.Series:
    """
    Trims the species name in the given column.
    """
    return col.str.extract("([A-Za-z]+ [A-Za-z]+)", expand=False).astype(
        "string[pyarrow]"
    )


def clean_species_name(
    df: pd.DataFrame, sp_col: str, new_sp_col: str | None = None
) -> pd.DataFrame:
    """
    Cleans a column containing species names by trimming them to the leading two words
    and converting them to lowercase.
    """
    if new_sp_col is None:
        new_sp_col = sp_col

    # Perform all operations in one chain
    result = df.assign(
        **{new_sp_col: trim_species_name(df[sp_col]).str.lower()},
    ).dropna(subset=[new_sp_col])

    return result


def filter_pft(df: pd.DataFrame, pft_set: str, pft_col: str = "pft") -> pd.DataFrame:
    """
    Filter the DataFrame based on the specified plant functional types (PFTs).

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        pft_set (str): The plant functional types to filter by, separated by underscores.
        pft_col (str, optional): The column name in the DataFrame that contains the PFTs.
            Defaults to "pft".

    Returns:
        pd.DataFrame: The filtered DataFrame.

    Raises:
        ValueError: If an invalid PFT designation is provided.
    """
    pfts = pft_set.split("_")
    if not any(pft in ["Shrub", "Tree", "Grass"] for pft in pfts):
        raise ValueError(f"Invalid PFT designation: {pft_set}")

    return df[df[pft_col].isin(pfts)]


def get_active_traits(cfg: ConfigBox | None = None) -> list[str]:
    """Returns a list of full names of the active traits. E.g. ['X1_mean', 'X2_mean']"""
    if cfg is None:
        cfg = get_config()
    y_cfg = cfg.datasets.Y
    return [f"X{i}_{y_cfg.trait_stats[y_cfg.trait_stat - 1]}" for i in y_cfg.traits]


def get_trait_number_from_id(trait_id: str) -> str:
    """Parses the trait number from a trait id string."""
    tnum = re.search(r"\d+", trait_id)
    if tnum is None:
        raise ValueError(f"Could not extract trait number from {trait_id}")
    return tnum.group()


def load_trait_mapping(cfg: ConfigBox | None = None) -> dict:
    if cfg is None:
        cfg = get_config()
    with open(cfg.trait_mapping, encoding="utf-8") as f:
        return json.load(f)


def get_trait_name_from_id(
    trait_id: str, length: str = "short", cfg: ConfigBox | None = None
) -> tuple[str, str]:
    """Returns the name of a trait from its id as well as the unit of the trait."""
    if cfg is None:
        cfg = get_config()

    mapping = load_trait_mapping(cfg)

    tnum = get_trait_number_from_id(trait_id)

    if tnum not in mapping:
        raise ValueError(f"Trait number {tnum} not in mapping")

    if length not in mapping[tnum]:
        raise ValueError(f"Length {length} not in mapping for trait {trait_id}")

    return mapping[tnum][length], mapping[tnum]["unit"]


# def get_traits_to_process(
#     valid_traits: list[int],
#     using_pca: bool,
#     trait_id: int | None,
# ) -> list[str]:
#     """Get the traits to process."""
#     if using_pca and trait_id is not None:
#         raise ValueError("Cannot specify a trait ID when using PCA")

#     if using_pca:
#         # In this case, we're going to be creating PCA maps
#         trait_cols = dd.read_parquet(get_try_traits_interim_fn()).columns
#         pca_components = [c for c in trait_cols if c.startswith("PC")]
#         return pca_components

#     else:
#         # In this case, we're going to be creating trait maps of one or more traits
#         return format_traits_to_process(trait_id, valid_traits)


def format_traits_to_process(
    specific_trait: int | None, valid_traits: list[int]
) -> list[str]:
    # If trait is specified, only process that one
    if specific_trait:
        if specific_trait not in valid_traits:
            raise ValueError(
                f"Invalid trait ID: {specific_trait}. Valid traits are: {', '.join(map(str, valid_traits))}"
            )
        log.info("Processing single trait: %s", specific_trait)
        return [f"X{specific_trait}"]
    else:
        log.info("Processing all valid traits")
        return [f"X{t}" for t in valid_traits]


def check_for_existing_maps(
    out_dir: Path, traits_to_process: list[str], fd_stats_to_process: list[str]
) -> tuple[list[str], list[str]]:
    """
    Check for existing maps and return the traits to process. If we're in FD mode, and
    there are still FD metrics to process, we return all of the traits along with the
    remaining FD metrics to process.

    Args:
        out_dir (Path): The output directory.
        traits_to_process (list[str]): The traits to process.
        fd_mode (bool): Whether we're in FD mode.
        cfg (ConfigBox): The configuration.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
            - The first list contains the traits that need to be processed.
            - The second list contains the FD metrics that need to be processed.
    """

    def _check_list(trait_list: list[str]) -> list[str]:
        traits_to_process_filtered = []
        for trait in trait_list:
            out_path = out_dir / f"{trait}.tif"
            if out_path.exists():
                log.info("✓ %s.tif already exists, skipping...", trait)
                continue
            traits_to_process_filtered.append(trait)
            log.info("✗ %s.tif needs processing", trait)
        return traits_to_process_filtered

    log.info("Checking for existing output files...")

    if fd_stats_to_process:
        stats_to_process = _check_list(fd_stats_to_process)

        if stats_to_process:
            return traits_to_process, stats_to_process
        else:
            return [], []
    else:
        return _check_list(traits_to_process), []


# def load_try_traits(npartitions: int, traits_to_process: list[str]) -> dd.DataFrame:
#     needed_columns = ["speciesname"]
#     needed_columns.extend(traits_to_process)
#     log.info("Loading trait data for columns: %s", ", ".join(needed_columns))

#     # Load pre-cleaned and filtered TRY traits and set species as index
#     traits = (
#         dd.read_parquet(
#             get_try_traits_interim_fn(),
#             columns=needed_columns,
#         )
#         .pipe(repartition_if_set, npartitions)
#         .set_index("speciesname")
#     )
#     return traits
