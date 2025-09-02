# Trait Pearson R Dot Plot Visualization

This script creates a landscape-oriented dot plot showing fold-wise mean Pearson R and standard deviation for each trait across trait sets (SCI, COMB, CIT) for 1 km resolution models with power transformation.

## Usage

### Basic Usage
```bash
python -m src.visualization.figures.trait_pearsonr_dotplot
```

### Advanced Usage
```bash
python -m src.visualization.figures.trait_pearsonr_dotplot \
    --out_path ./results/figures/custom_trait_plot.png \
    --dpi 300 \
    --figsize 24,10 \
    --error_type ci95
```

## Arguments

- `--out_path` or `-o`: Output file path (default: `./results/figures/trait-pearsonr-dotplot.png`)
- `--dpi`: DPI for output figure (default: 300)
- `--figsize`: Figure size as 'width,height' in inches (default: '20,8')
- `--error_type`: Type of error bars - 'std' for standard deviation, 'ci95' for 95% confidence interval (default: 'std')

## Output

The script generates a landscape-oriented dot plot with:
- X-axis: All 31 traits with names rotated 45 degrees for readability
- Y-axis: Pearson's r (fold-wise mean ± error bars)
- Three dots per trait representing the three trait sets (SCI, COMB, CIT)
- Error bars showing either standard deviation or 95% confidence intervals
- Color scheme matching other project figures:
  - SCI (sPlot): #b0b257 (olive/yellow-green)
  - COMB (sPlot+GBIF): #66a9aa (teal)
  - CIT (GBIF): #b95fa1 (purple/magenta)

## Requirements

- The script must be run from the project root directory
- Requires `results/all_results.parquet` file to be present
- Uses traits defined in `params.yaml` under `datasets.Y.traits`

## Data Filters

The script automatically filters the data for:
- Resolution: 1 km only
- Transform: power transformation only  
- Traits: Only those listed in the configuration

## Scientific Context

This visualization is designed to show the fold-wise cross-validation performance (Pearson correlation) for each trait across different trait sets, providing insight into:
- Which traits are most predictable across trait sets
- How trait set composition affects predictive performance
- The variability in performance across spatial folds

### Error Bar Options

The script supports two types of error bars:

1. **Standard Deviation (`--error_type std`)**: Shows the standard deviation of Pearson R across the 5 cross-validation folds
2. **95% Confidence Interval (`--error_type ci95`)**: Shows the 95% confidence interval calculated using the t-distribution with 4 degrees of freedom (n_folds - 1), which provides a more statistically interpretable measure of uncertainty

The 95% confidence interval is often preferred in scientific publications as it indicates the range within which the true population mean is likely to fall with 95% confidence.

The figure is optimized for inclusion in academic papers in landscape format to accommodate the 31 traits along the x-axis. 