# Histogram Modeling Input Predictors

**Pipeline:** `try6_hist_pow-xf_22km`
**Resolution:** 22 km grid cells (EPSG:6933, Equal Area Cylindrical)
**Total predictors:** 151

---

## 1. Canopy Height (2 features)

Source: ETH Global Canopy Height, 2020 v1
Citation:
Lang, N., Jetz, W., Schindler, K. & Wegner, J. D. A high-resolution canopy height model of the Earth. Nat Ecol Evol 7, 1778–1789 (2023).


| ID | Description |
|----|-------------|
| `ETH_GlobalCanopyHeight_2020_v1` | Mean canopy height (m) |
| `ETH_GlobalCanopyHeightSD_2020_v1` | Standard deviation of canopy height (m) |

## 2. MODIS Surface Reflectance (72 features)

Source: NASA MODIS MOD09A1.061 Terra Surface Reflectance, monthly means over 2001–2024 (m1–m12)
Citation:
Vermote, E. & Wolfe, R. MODIS/Terra Surface Reflectance Daily L2G Global 1km and 500m SIN Grid V061. NASA EOSDIS Land Processes DAAC https://doi.org/10.5067/MODIS/MOD09GA.061 (2021).


| Band pattern | Description |
|--------------|-------------|
| `sur_refl_b01_2001-2024_m{1-12}_mean` | Red band (620–670 nm) |
| `sur_refl_b02_2001-2024_m{1-12}_mean` | Near-infrared band (841–876 nm) |
| `sur_refl_b03_2001-2024_m{1-12}_mean` | Blue band (459–479 nm) |
| `sur_refl_b04_2001-2024_m{1-12}_mean` | Green band (545–565 nm) |
| `sur_refl_b05_2001-2024_m{1-12}_mean` | SWIR band (1230–1250 nm) |
| `sur_refl_ndvi_2001-2024_m{1-12}_mean` | NDVI (Normalized Difference Vegetation Index) |

Reflectance values scaled by 10,000 (range 0–16,000).

6 bands × 12 months = 72 features

## 3. SoilGrids v2.0 (61 features)

Source: ISRIC SoilGrids v2.0
Citation:
Poggio, L. et al. SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty. SOIL 7, 217–240 (2021).


Each property measured at 6 depth intervals: 0–5 cm, 5–15 cm, 15–30 cm, 30–60 cm, 60–100 cm, 100–200 cm.

Values are in native SoilGrids integer-scaled units (no conversion applied).

| Property prefix | Description | Units |
|-----------------|-------------|-------|
| `bdod_{depth}_mean` | Bulk density of fine earth | cg/cm³ |
| `cec_{depth}_mean` | Cation exchange capacity | mmol(c)/kg |
| `cfvo_{depth}_mean` | Coarse fragments volume | cm³/dm³ |
| `clay_{depth}_mean` | Clay content | g/kg |
| `nitrogen_{depth}_mean` | Total nitrogen | cg/kg |
| `ocd_{depth}_mean` | Organic carbon density | hg/m³ |
| `phh2o_{depth}_mean` | Soil pH in water | pH × 10 |
| `sand_{depth}_mean` | Sand content | g/kg |
| `silt_{depth}_mean` | Silt content | g/kg |
| `soc_{depth}_mean` | Soil organic carbon content | dg/kg |
| `ocs_0-30cm_mean` | Organic carbon stocks (0–30 cm only) | t/ha |

10 properties × 6 depths + 1 = 61 features

## 4. WorldClim Bioclimatic Variables (6 features)

Source: WorldClim v2.1 (30 arc-second resolution, aggregated to 22 km)
Citation:
Fick, S. E. & Hijmans, R. J. WorldClim 2: new 1-km spatial resolution climate surfaces for global land areas. International Journal of Climatology 37, 4302–4315 (2017).


| ID | Description |
|----|-------------|
| `wc2.1_30s_bio_1` | Annual mean temperature (°C × 10) |
| `wc2.1_30s_bio_4` | Temperature seasonality (std dev × 100) |
| `wc2.1_30s_bio_7` | Temperature annual range (°C × 10) |
| `wc2.1_30s_bio_12` | Annual precipitation (mm) |
| `wc2.1_30s_bio_13-14` | Precipitation range: wettest minus driest month (mm) — derived as bio_13 − bio_14 |
| `wc2.1_30s_bio_15` | Precipitation seasonality (coefficient of variation, %) |

## 5. VODCA – Vegetation Optical Depth (9 features)

Source: Vegetation Optical Depth Climate Archive
Citation:
Moesinger, L. et al. The global long-term microwave Vegetation Optical Depth Climate Archive (VODCA). Earth System Science Data 12, 177–196 (2020).


Summary statistics of vegetation optical depth at 3 microwave frequencies:

| ID | Description |
|----|-------------|
| `vodca_c-band_mean` | C-band (~6.9 GHz) VOD mean |
| `vodca_c-band_p5` | C-band VOD 5th percentile |
| `vodca_c-band_p95` | C-band VOD 95th percentile |
| `vodca_x-band_mean` | X-band (~10.7 GHz) VOD mean |
| `vodca_x-band_p5` | X-band VOD 5th percentile |
| `vodca_x-band_p95` | X-band VOD 95th percentile |
| `vodca_k-band_mean` | Ku-band (~18.7 GHz) VOD mean |
| `vodca_k-band_p5` | Ku-band VOD 5th percentile |
| `vodca_k-band_p95` | Ku-band VOD 95th percentile |

## 6. ALOS Terrain (1 feature)

Source: ALOS DEM

| ID | Description |
|----|-------------|
| `ALOS_CHILI_constant` | Continuous Heat-Insolation Load Index (0–255) |

---

## Key File Locations

| Resource | Path |
|----------|------|
| Feature data | `data/features/predict/modis_wc2_soil_canopy_vodca_alos_22km/X.parquet` |
| Feature report | `data/features/predict/modis_wc2_soil_canopy_vodca_alos_22km/report.md` |
| Pipeline params | `pipeline/products/try6_hist_pow-xf_22km/params.yaml` |
| Training data | `data/features/try6_hist_pow-xf_22km/xy_data/train.zarr` |
