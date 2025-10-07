"""Extract scale_factor and add_offset metadata from interim EO data files."""

import json
from pathlib import Path

import rasterio

from src.conf.conf import get_config
from src.conf.environment import log


def extract_eo_metadata(resolution: str = "22km") -> dict[str, dict]:
    """
    Extract scale_factor and add_offset metadata from all EO data geotiffs.

    Args:
        resolution: The resolution directory to process (e.g., "22km", "1km")

    Returns:
        Dictionary mapping filenames to their metadata (scale_factor, add_offset, nodata)
    """
    cfg = get_config()
    eo_data_dir = Path(cfg.interim_dir) / cfg.eo_data.interim.dir / resolution

    if not eo_data_dir.exists():
        raise ValueError(f"Directory does not exist: {eo_data_dir}")

    metadata = {}

    # Iterate over all subdirectories (datasets)
    for dataset_dir in sorted(eo_data_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        log.info(f"Processing dataset: {dataset_name}")

        # Iterate over all geotiff files in the dataset
        for tif_file in sorted(dataset_dir.glob("*.tif")):
            filename = f"{dataset_name}/{tif_file.name}"

            with rasterio.open(tif_file) as src:
                # Get scales and offsets for all bands
                scales = src.scales
                offsets = src.offsets
                nodata = src.nodata

                # For single-band files, extract the first value
                scale_factor = scales[0] if scales and scales[0] is not None else 1.0
                add_offset = offsets[0] if offsets and offsets[0] is not None else 0.0

                metadata[filename] = {
                    "scale_factor": scale_factor,
                    "add_offset": add_offset,
                }

                # If multiple bands have different scales/offsets, store all of them
                if src.count > 1 and (len(set(scales)) > 1 or len(set(offsets)) > 1):
                    metadata[filename]["scales"] = scales
                    metadata[filename]["offsets"] = offsets

                log.info(
                    f"  {tif_file.name}: scale={scale_factor}, "
                    f"offset={add_offset}, nodata={nodata}"
                )

    return metadata


def save_eo_metadata(
    metadata: dict[str, dict],
    output_path: Path | str | None = None,
    resolution: str = "22km",
) -> None:
    """
    Save EO metadata to a JSON file.

    Args:
        metadata: Dictionary of metadata extracted from extract_eo_metadata()
        output_path: Path to save the JSON file. If None, saves to reference directory.
        resolution: Resolution identifier to include in default filename.
    """
    if output_path is None:
        output_path = Path("reference") / f"eo_data_metadata_{resolution}.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Saved metadata to {output_path}")


def main() -> None:
    """Main function to extract and save EO metadata."""
    cfg = get_config()
    resolution = cfg.model_res

    log.info(f"Extracting metadata for resolution: {resolution}")
    metadata = extract_eo_metadata(resolution)

    log.info(f"Found metadata for {len(metadata)} files")

    output_path = Path("reference") / f"eo_data_metadata_{resolution}.json"
    save_eo_metadata(metadata, output_path, resolution)

    log.info("Done!")


if __name__ == "__main__":
    main()
