from pathlib import Path
from src.conf.conf import get_config
from src.utils.raster_utils import create_sample_raster, open_raster, xr_to_raster
from src.conf.environment import log

cfg = get_config()


def main() -> None:
    ref_r = create_sample_raster(resolution=1, crs="EPSG:4326")

    fns = list(Path(cfg.interim_dir, "other_trait_maps", "111km").glob("*.tif"))

    for fn in fns:
        log.info(f"Reprojecting {fn} to WGS84...")
        r = open_raster(fn)
        if "wolf" in fn.stem:
            r = r.sel(band=2)
            r.attrs["long_name"] = fn.stem
        r_reproj = r.rio.reproject_match(ref_r)
        xr_to_raster(r_reproj, Path("tmp") / fn.name)
        log.info(f"Reprojected {fn} to WGS84.")

    log.info("Done.")


if __name__ == "__main__":
    main()
