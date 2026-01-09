"""Build final product for a single trait.

This module creates a Cloud-Optimized GeoTIFF (COG) with 3 bands:
1. Trait prediction
2. Coefficient of Variation (CoV)
3. Area of Applicability (AoA) mask

The output file includes comprehensive metadata and is optimized for cloud access.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.io.upload_sftp import upload_file_sftp
from src.utils.dataset_utils import (
    get_aoa_dir,
    get_cov_dir,
    get_model_performance,
    get_predict_dir,
    get_processed_dir,
)
from src.utils.raster_utils import pack_data
from src.utils.spatial_utils import interpolate_like
from src.utils.trait_utils import get_trait_number_from_id


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build final product for a single trait."
    )
    parser.add_argument(
        "--trait",
        type=str,
        required=True,
        help="Trait to process (e.g., 'leaf_N_per_dry_mass')",
    )
    parser.add_argument(
        "--trait-set",
        type=str,
        required=True,
        help="Trait set to use (e.g., 'Shrub_Tree_Grass')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="local",
        choices=["local", "sftp", "both"],
        help="Destination for output (default: local)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to params.yaml file (default: uses active config)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode"
    )
    return parser.parse_args()


def _check_is_packed(ds: rasterio.DatasetReader) -> bool:
    """Check if a raster dataset is packed (has scales/offsets).

    Args:
        ds: Rasterio dataset reader

    Returns:
        True if dataset is packed, False otherwise
    """
    return (
        ds.scales[0] is not None
        and ds.offsets[0] is not None
        and not np.isnan(ds.nodata)
    )


def load_trait_metadata(cfg: ConfigBox) -> dict:
    """Load trait metadata from reference files.

    Args:
        cfg: Configuration

    Returns:
        Dictionary with trait_mapping and trait_stats
    """
    with open("reference/trait_mapping.json", "rb") as f:
        trait_mapping = json.load(f)

    with open("reference/trait_stat_mapping.json", "rb") as f:
        trait_stats = json.load(f)

    return {"trait_mapping": trait_mapping, "trait_stats": trait_stats}


def check_existing_bands(output_path: Path) -> dict[str, bool]:
    """Check which bands exist in an existing final product raster.

    Args:
        output_path: Path to the final product raster

    Returns:
        Dictionary with keys 'predict', 'cov', 'aoa' and boolean values indicating presence
    """
    if not output_path.exists():
        return {"predict": False, "cov": False, "aoa": False}

    try:
        with rasterio.open(output_path, "r") as ds:
            num_bands = ds.count
            band_descriptions = [ds.descriptions[i] for i in range(num_bands)]

            # Check for expected band descriptions
            has_predict = num_bands >= 1 and band_descriptions[0] is not None
            has_cov = num_bands >= 2 and band_descriptions[1] is not None and "Coefficient of Variation" in band_descriptions[1]
            has_aoa = num_bands >= 3 and band_descriptions[2] is not None and "Area of Applicability" in band_descriptions[2]

            return {
                "predict": has_predict,
                "cov": has_cov,
                "aoa": has_aoa,
            }
    except Exception as e:
        log.warning(f"Error checking existing bands: {e}")
        return {"predict": False, "cov": False, "aoa": False}


def get_missing_bands(
    trait: str,
    trait_set: str,
    predict_dir: Path,
    cov_dir: Path,
    aoa_dir: Path,
    output_path: Path,
) -> set[str]:
    """Determine which bands need to be generated based on available inputs and existing output.

    Args:
        trait: Trait name
        trait_set: Trait set name
        predict_dir: Directory containing prediction rasters
        cov_dir: Directory containing CoV rasters
        aoa_dir: Directory containing AoA rasters
        output_path: Path to final product raster

    Returns:
        Set of band names that need to be generated ('predict', 'cov', 'aoa')
    """
    # Check which bands exist in the output
    existing_bands = check_existing_bands(output_path)

    # Check which input files are available
    predict_path = predict_dir / trait / trait_set / f"{trait}_{trait_set}_predict.tif"
    cov_path = cov_dir / trait / trait_set / f"{trait}_{trait_set}_cov.tif"
    aoa_path = aoa_dir / trait / trait_set / f"{trait}_{trait_set}_aoa.tif"

    available_inputs = {
        "predict": predict_path.exists(),
        "cov": cov_path.exists(),
        "aoa": aoa_path.exists(),
    }

    # Determine which bands need to be added/updated
    missing_bands = set()

    for band_name in ["predict", "cov", "aoa"]:
        # Need to generate if:
        # 1. Band doesn't exist in output AND input is available
        # 2. OR we're in overwrite mode and input is available
        if available_inputs[band_name] and not existing_bands[band_name]:
            missing_bands.add(band_name)

    return missing_bands


def build_final_product_single_trait(
    trait: str,
    trait_set: str,
    cfg: ConfigBox,
    metadata: dict,
    trait_metadata: dict,
    destination: str = "local",
    overwrite: bool = False,
) -> Path:
    """Build final product for a single trait.

    Supports smart partial updates: if final raster exists and overwrite=False,
    only missing bands will be added/updated.

    Args:
        trait: Trait name
        trait_set: Trait set name
        cfg: Configuration
        metadata: General metadata dictionary
        trait_metadata: Trait-specific metadata (trait_mapping and trait_stats)
        destination: Output destination ('local', 'sftp', or 'both')
        overwrite: Whether to overwrite existing output (if False, performs partial update)

    Returns:
        Path to output file

    Raises:
        FileNotFoundError: If required input files not found
        ValueError: If destination is invalid
    """
    if destination not in ["sftp", "local", "both"]:
        raise ValueError("Invalid destination. Must be one of 'sftp', 'local', 'both'.")

    output_file_name = f"{trait}_{cfg.PFT}_{cfg.model_res}.tif"

    # Determine output path
    local_dest_dir = get_processed_dir() / cfg.public.local_dir
    output_path = local_dest_dir / output_file_name

    # Paths to input directories
    predict_dir = get_predict_dir()
    cov_dir = get_cov_dir(cfg)
    aoa_dir = get_aoa_dir()

    # Check for missing bands if not overwriting
    if not overwrite and output_path.exists():
        missing_bands = get_missing_bands(
            trait, trait_set, predict_dir, cov_dir, aoa_dir, output_path
        )

        if not missing_bands:
            log.info("All bands present in %s, nothing to update", output_path)
            return output_path

        log.info("Missing bands in %s: %s", output_path, ", ".join(missing_bands))
        log.info("Performing partial update...")
    elif overwrite and output_path.exists():
        log.info("Overwriting existing file: %s", output_path)

    log.info("Processing %s (%s)...", trait, trait_set)

    # Get trait metadata
    trait_num = get_trait_number_from_id(trait)
    trait_mapping = trait_metadata["trait_mapping"]
    trait_stats = trait_metadata["trait_stats"]
    trait_meta = trait_mapping[trait_num]

    # Update metadata with trait info
    metadata = metadata.copy()
    metadata["trait_short_name"] = trait_meta["short"]
    metadata["trait_long_name"] = trait_meta["long"]
    metadata["trait_unit"] = trait_meta["unit"]

    # Paths to input files
    predict_dir = get_predict_dir()
    cov_dir = get_cov_dir(cfg)
    aoa_dir = get_aoa_dir()

    predict_path = predict_dir / trait / trait_set / f"{trait}_{trait_set}_predict.tif"
    cov_path = cov_dir / trait / trait_set / f"{trait}_{trait_set}_cov.tif"
    aoa_path = aoa_dir / trait / trait_set / f"{trait}_{trait_set}_aoa.tif"

    # Determine if we're doing a partial update
    doing_partial_update = not overwrite and output_path.exists()
    if doing_partial_update:
        missing_bands = get_missing_bands(
            trait, trait_set, predict_dir, cov_dir, aoa_dir, output_path
        )
    else:
        missing_bands = {"predict", "cov", "aoa"}

    # Check that required files exist
    required_paths = []
    if "predict" in missing_bands:
        required_paths.append(predict_path)
    if "cov" in missing_bands:
        required_paths.append(cov_path)
    if "aoa" in missing_bands:
        required_paths.append(aoa_path)

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Initialize band data storage
    scales = []
    offsets = []
    predict = None
    cov = None
    aoa = None
    predict_valid_mask = None
    nodata = None

    # Load existing bands if doing partial update
    if doing_partial_update:
        log.info("Loading existing bands from %s...", output_path)
        with rasterio.open(output_path, "r") as existing_ds:
            existing_bands = check_existing_bands(output_path)

            if existing_bands["predict"]:
                predict = existing_ds.read(1)
                scales.append(existing_ds.scales[0])
                offsets.append(existing_ds.offsets[0])
                nodata = existing_ds.nodata
                predict_valid_mask = predict != nodata
                log.info("Loaded existing predict band")

            if existing_bands["cov"]:
                cov = existing_ds.read(2)
                scales.append(existing_ds.scales[1])
                offsets.append(existing_ds.offsets[1])
                log.info("Loaded existing CoV band")

            if existing_bands["aoa"]:
                aoa = existing_ds.read(3)
                scales.append(existing_ds.scales[2])
                offsets.append(existing_ds.offsets[2])
                log.info("Loaded existing AoA band")

    # Load missing bands
    if "predict" in missing_bands:
        log.info("Reading and packing predict...")
        predict_ds = rasterio.open(predict_path, "r")
        predict_is_packed = _check_is_packed(predict_ds)
        predict = predict_ds.read(1)

        if predict_is_packed:
            predict_valid_mask = predict != predict_ds.nodata
        else:
            predict_valid_mask = ~np.isnan(predict)

        if not predict_is_packed:
            predict = pack_data(predict_ds.read(1))
            if doing_partial_update and len(scales) >= 1:
                scales[0] = predict[0]
                offsets[0] = predict[1]
            else:
                scales.insert(0, predict[0])
                offsets.insert(0, predict[1])
            nodata = predict[2]
            predict = predict[3]
        else:
            if doing_partial_update and len(scales) >= 1:
                scales[0] = predict_ds.scales[0]
                offsets[0] = predict_ds.offsets[0]
            else:
                scales.insert(0, predict_ds.scales[0])
                offsets.insert(0, predict_ds.offsets[0])
            nodata = predict_ds.nodata
        predict_ds.close()

    if "cov" in missing_bands:
        log.info("Reading and packing CoV...")
        cov_dataset = rasterio.open(cov_path, "r")
        cov_is_packed = _check_is_packed(cov_dataset)
        cov = cov_dataset.read(1)

        if cov_is_packed:
            if doing_partial_update and len(scales) >= 2:
                scales[1] = cov_dataset.scales[0]
                offsets[1] = cov_dataset.offsets[0]
            else:
                scales.append(cov_dataset.scales[0])
                offsets.append(cov_dataset.offsets[0])
        else:
            cov_packed = pack_data(cov_dataset.read(1))
            if doing_partial_update and len(scales) >= 2:
                scales[1] = cov_packed[0]
                offsets[1] = cov_packed[1]
            else:
                scales.append(cov_packed[0])
                offsets.append(cov_packed[1])
            cov = cov_packed[3]
        cov_dataset.close()

    if "aoa" in missing_bands:
        log.info("Reading AoA...")
        aoa_dataset = rasterio.open(aoa_path, "r")
        aoa_is_packed = _check_is_packed(aoa_dataset)
        log.info("Interpolating AoA...")
        aoa_nodata = aoa_dataset.nodata if aoa_is_packed else None
        aoa_interpolated = interpolate_like(
            aoa_dataset.read(2), predict_valid_mask, use_gpu=False, nodata_value=aoa_nodata
        )
        log.info("Masking and downcasting AoA...")
        aoa = np.where(np.isnan(aoa_interpolated), nodata, aoa_interpolated).astype("int16")
        if doing_partial_update and len(scales) >= 3:
            scales[2] = 1
            offsets[2] = 0
        else:
            scales.append(1)
            offsets.append(0)
        aoa_dataset.close()

    # Get model performance metrics
    log.info("Gathering model performance metrics...")
    # Check if using new per-trait transformation system
    if hasattr(cfg.traits, "power_transform"):
        # New system: check trait-specific transformation metadata
        transformer_dir = Path(cfg.traits.transformer_dir)
        metadata_file = transformer_dir / f"{trait}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                transform_meta = json.load(f)
            xform = (
                cfg.traits.transform_method if transform_meta["is_transformed"] else "none"
            )
        else:
            xform = "none"
    else:
        # Old system: global transform config
        xform = cfg.trydb.interim.transform
        xform = xform if xform is not None else "none"

    mp = get_model_performance(trait, trait_set).query(f"transform == '{xform}'")
    pearson_r = "pearsonr_wt" if cfg.crs == "EPSG:4326" else "pearsonr"
    mp_dict = {
        "R2": mp["r2"].values[0].round(2),
        "Pearson's r": mp[pearson_r].values[0].round(2),
        "nRMSE": mp["norm_root_mean_squared_error"].values[0].round(2),
        "RMSE": mp["root_mean_squared_error"].values[0].round(2),
        "MAE": mp["mean_absolute_error"].values[0].round(2),
    }

    # Generate raster metadata - use predict raster as reference
    log.info("Generating metadata...")
    # Open predict raster for metadata (use existing output if doing partial update)
    if doing_partial_update:
        ref_ds = rasterio.open(output_path, "r")
    else:
        ref_ds = rasterio.open(predict_path, "r")

    bounds = ref_ds.bounds
    spatial_extent = (
        f"min_x: {bounds.left}, min_y: {bounds.bottom}, "
        f"max_x: {bounds.right}, max_y: {bounds.top}"
    )

    raster_meta = {
        "crs": ref_ds.crs,
        "resolution": ref_ds.res,
        "geospatial_units": "degrees",
        "grid_coordinate_system": "WGS 84",
        "transform": ref_ds.transform,
        "nodata": nodata,
        "spatial_extent": spatial_extent,
        "model_performance": json.dumps(mp_dict),
    }

    # Create COG profile
    cog_profile = ref_ds.profile.copy()
    cog_profile.update(count=3)

    # Ensure new profile is configured to write as a COG
    cog_profile.update(
        driver="GTiff",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="ZSTD",
        copy_src_overviews=True,
        interleave="band",
        nodata=nodata,
    )

    new_tags = ref_ds.tags().copy()

    for key, value in metadata.items():
        new_tags[key] = value

    for key, value in raster_meta.items():
        new_tags[key] = value

    # Close reference dataset
    ref_ds.close()

    # Write output file
    log.info("Writing new file...")
    with tempfile.TemporaryDirectory() as temp_dir:
        new_file_path = Path(temp_dir, output_file_name)

        # Create a new file with updated metadata and data
        with rasterio.open(new_file_path, "w", **cog_profile) as new_dataset:
            new_dataset.update_tags(**new_tags)

            log.info("Setting scales and offsets...")
            scales_arr = np.array(scales, dtype=np.float64)
            offsets_arr = np.array(offsets, dtype=np.float64)
            new_dataset._set_all_scales(scales_arr)  # pylint: disable=protected-access
            new_dataset._set_all_offsets(offsets_arr)  # pylint: disable=protected-access

            for i in range(1, 4):
                new_dataset.update_tags(i, _FillValue=nodata)

            log.info("Writing bands...")
            new_dataset.write(predict, 1)
            new_dataset.write(cov, 2)
            new_dataset.write(aoa, 3)

            new_dataset.set_band_description(
                1,
                f"{trait_meta['short']} ({trait_stats[str(cfg.datasets.Y.trait_stat)]})",
            )
            new_dataset.set_band_description(2, "Coefficient of Variation")
            new_dataset.set_band_description(3, "Area of Applicability mask")

        cog_path = (
            new_file_path.parent / f"{new_file_path.stem}_cog{new_file_path.suffix}"
        )
        log.info("Writing COG...")
        subprocess.run(
            [
                "rio",
                "cogeo",
                "create",
                "--overview-resampling",
                "average",
                "--in-memory",
                str(new_file_path),
                str(cog_path),
            ],
            check=True,
        )

        output_path = None
        if destination in ("local", "both"):
            local_dest_dir = get_processed_dir() / cfg.public.local_dir
            local_dest_dir.mkdir(parents=True, exist_ok=True)
            log.info("Copying to local directory...")
            output_path = local_dest_dir / new_file_path.name
            shutil.copy2(cog_path, output_path)

        if destination in ("sftp", "both"):
            log.info("Uploading to the server...")
            upload_file_sftp(
                cog_path,
                str(
                    Path(
                        cfg.public.sftp_dir,
                        cfg.PFT,
                        cfg.model_res,
                        new_file_path.name,
                    )
                ),
            )

    log.info("Done! ✅")
    return output_path if output_path else cog_path


def main(args: argparse.Namespace, cfg: ConfigBox | None = None) -> Path:
    """Main function for single trait final product building.

    Args:
        args: Command-line arguments
        cfg: Configuration (if None, loads from get_config())

    Returns:
        Path to output file
    """
    if cfg is None:
        cfg = get_config()

    if not args.verbose:
        log.setLevel("WARNING")

    today = datetime.today().strftime("%Y-%m-%d")

    metadata = {
        "author": "Daniel Lusk",
        "organization": "Department for Sensor-based Geoinformatics, University of Freiburg",
        "contact": "daniel.lusk@geosense.uni-freiburg.de",
        "creation_date": today,
        "version": cfg.version,
        "type": "dataset",
        "language": "en",
        "keywords": "citizen-science, plant functional traits, global, 1km, "
        "earth observation, gbif, splot, try, modis, soilgrids, vodca, worldclim",
        "license": "This dataset is provided for research purposes only. Redistribution "
        "or commercial use is prohibited without permission.",
        "rights": "This dataset is the intellectual property of the Department for "
        "Sensor-based Geoinformatics, University of Freiburg. Publication pending. "
        "Do not distribute or use for commercial purposes without express permission "
        "from the authors.",
        "PFTs": cfg.PFT.replace("_", ", "),
        "usage_notes": """This dataset contains extrapolations of trait data from the
        TRY Trait Database matched with geotagged species observations from GBIF and
        species abundances from sPlot. Extrapolations are the result of modeling gridded
        trait values as a function of publicly available Earth observation datasets. All
        model performance metrics (R2, Pearson's r, nRMSE, RMSE) are based on spatial
        K-fold cross-validation against unseen sPlot observations only.""",
    }

    trait_metadata = load_trait_metadata(cfg)

    return build_final_product_single_trait(
        trait=args.trait,
        trait_set=args.trait_set,
        cfg=cfg,
        metadata=metadata,
        trait_metadata=trait_metadata,
        destination=args.dest,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main(cli())
