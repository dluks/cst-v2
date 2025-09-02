# Trait Matching Statistics

This module contains scripts for analyzing trait matching statistics between GBIF citizen science observations, sPlot vegetation survey data, and TRY trait database records.

## Overview

The `trait_matching_stats.py` script generates statistics showing how many species and observations are available before and after matching with TRY trait data, for each trait individually. This analysis is essential for understanding data coverage and transparency in trait mapping studies.

## What the Script Does

1. **Loads datasets**: Reads GBIF citizen science data, sPlot vegetation survey data, and TRY trait data
2. **Applies filters**: Filters data by plant functional type (PFT) as configured
3. **Analyzes trait matching**: For each trait, counts:
   - Total species and observations in the original dataset
   - Species and observations remaining after PFT filtering
   - Species and observations that successfully match with TRY trait data
   - Percentage of species/observations retained after trait matching
4. **Generates output**: Creates a CSV file with formatted results

## Output Format

The output CSV contains the following columns:

- **Trait**: Human-readable trait name (e.g., "SSD", "Leaf N (mass)")
- **Dataset**: Either "GBIF" or "sPlot"
- **Species**: Number and percentage of species with trait data in format "N (X.X%)"
- **Observations**: Number and percentage of observations with trait data in format "N (X.X%)"

Example output:
```
Trait,Dataset,Species,Observations
SSD,GBIF,12543 (45.2%),1256789 (38.7%)
SSD,sPlot,8432 (67.8%),89456 (72.1%)
Leaf N (mass),GBIF,18765 (67.5%),1789234 (55.1%)
...
```

## Usage

### Direct execution:

```bash
cd cst-repo
python -m src.analysis.trait_matching_stats -o trait_matching_stats.csv
```

### Using the wrapper script:

```bash
cd cst-repo
python scripts/generate_trait_matching_stats.py
```

The wrapper script automatically saves results to `results/trait_matching_stats.csv`.

### Command-line options:

- `-o, --output`: Specify output CSV filename (default: `trait_matching_stats.csv`)

## Requirements

- Processed GBIF data (`data/interim/gbif/gbif_pfts.parquet`)
- Processed sPlot data (`data/interim/splot/extracted/`)
- TRY trait data (`data/interim/try/traits.parquet`)
- Valid configuration in `params.yaml`

## Configuration Dependencies

The script reads the following configuration parameters:

- `datasets.Y.traits`: List of trait IDs to analyze
- `PFT`: Plant functional type filter
- `trydb.interim.perform_pca`: Whether PCA is being used
- Dataset paths and interim directories

## Technical Details

### Data Processing Flow

1. **GBIF Data**:
   - Loads from `gbif_pfts.parquet` (already matched with PFT data)
   - Filters by configured PFT
   - Sets species name as index for trait joining

2. **sPlot Data**:
   - Loads vegetation data from `vegetation.parquet`
   - Loads and filters PFT data
   - Joins vegetation with PFT data
   - Filters by configured PFT

3. **TRY Traits**:
   - Loads preprocessed trait data
   - For each trait, creates subset with non-null values
   - Joins with GBIF and sPlot data separately

4. **Statistics Calculation**:
   - Counts unique species and total observations before/after matching
   - Calculates percentages based on post-PFT filtering counts
   - Formats results for output

### Performance Considerations

- Uses Dask for distributed computing with configurable client
- Processes one trait at a time to manage memory usage
- Computes statistics lazily until needed

## Integration with Manuscript

This script directly addresses reviewer comments requesting transparency about trait data coverage. The results support statements in the rebuttal letter about providing "a table in the appendix that breaks this information down by trait" showing "the number of TRY observations and the number of unique species used after filtering and matching with GBIF and sPlot." 