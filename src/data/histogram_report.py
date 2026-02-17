"""Generate sanity-check report for histogram target Zarr stores.

Produces PNG figures and a Markdown summary to verify:
1. Spatial coverage is geographically sensible
2. Per-trait validity patterns are plausible
3. Individual cell distributions look like reasonable probability distributions
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import zarr

log = logging.getLogger(__name__)

# CRS used by the histogram pipeline (EPSG:6933 — Equal Area Cylindrical)
_SRC_CRS = ccrs.EqualEarth()

# Representative traits for detailed maps
_KEY_TRAITS = ["X4", "X14", "X3106", "X3117", "X26", "X6"]

# Latitude bands (in EPSG:6933 y-coordinates, approximate)
_REGION_BANDS = {
    "Boreal": (5_500_000, 7_500_000),
    "N. Temperate": (2_500_000, 5_500_000),
    "N. Subtropical": (500_000, 2_500_000),
    "Tropical": (-1_500_000, 500_000),
    "S. Subtropical": (-3_500_000, -1_500_000),
    "S. Temperate": (-6_000_000, -3_500_000),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_trait_descriptions(params_path: Path) -> dict[str, str]:
    """Parse trait IDs and descriptions from params.yaml comments."""
    descriptions: dict[str, str] = {}
    with open(params_path) as f:
        content = f.read()
    for match in re.finditer(r"^\s*-\s*(X\d+)\s*#\s*(.+)$", content, re.MULTILINE):
        descriptions[match.group(1)] = match.group(2).strip()
    return descriptions


def _trait_label(trait: str, descriptions: dict[str, str], max_len: int = 40) -> str:
    """Short human-readable label for a trait."""
    desc = descriptions.get(trait, "")
    if len(desc) > max_len:
        desc = desc[: max_len - 3] + "..."
    return f"{trait} — {desc}" if desc else trait


def _weighted_mean(hist: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Compute histogram weighted mean per cell.

    Parameters
    ----------
    hist : np.ndarray
        (N, n_bins) probability histograms.
    bin_edges : np.ndarray
        (n_bins + 1,) bin edge values.

    Returns
    -------
    np.ndarray
        (N,) weighted mean values.
    """
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return (hist * centers[np.newaxis, :]).sum(axis=1)


def _shannon_entropy(hist: np.ndarray) -> np.ndarray:
    """Shannon entropy per row (bits). Zeros are ignored."""
    h = np.where(hist > 0, hist * np.log2(hist), 0.0)
    return -h.sum(axis=1)


def _pick_sample_cells(
    coords: np.ndarray, masks: np.ndarray, n_per_band: int = 1
) -> list[int]:
    """Pick sample cell indices from distinct latitude bands."""
    indices: list[int] = []
    for _name, (y_lo, y_hi) in _REGION_BANDS.items():
        in_band = (coords[:, 1] >= y_lo) & (coords[:, 1] < y_hi)
        # Prefer cells with many valid traits
        candidates = np.where(in_band)[0]
        if len(candidates) == 0:
            continue
        n_valid = masks[candidates].sum(axis=1)
        best = candidates[np.argsort(-n_valid)[:n_per_band]]
        indices.extend(best.tolist())
    return indices


def _to_lonlat(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert EPSG:6933 (x, y) to (lon, lat) for cartopy plotting."""
    import pyproj

    transformer = pyproj.Transformer.from_crs(
        "EPSG:6933", "EPSG:4326", always_xy=True
    )
    lon, lat = transformer.transform(coords[:, 0], coords[:, 1])
    return lon, lat


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def _plot_spatial_coverage(
    lon: np.ndarray,
    lat: np.ndarray,
    n_valid: np.ndarray,
    out_path: Path,
) -> None:
    """Scatter map of cells colored by number of valid traits."""
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="gray")

    sc = ax.scatter(
        lon, lat,
        c=n_valid,
        s=1,
        cmap="viridis",
        transform=ccrs.PlateCarree(),
        rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Valid traits per cell")
    ax.set_title("Spatial coverage — valid traits per cell")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def _plot_trait_validity_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    masks: np.ndarray,
    trait_names: list[str],
    descriptions: dict[str, str],
    out_dir: Path,
) -> None:
    """Small-multiple grid of per-trait validity maps."""
    n_traits = len(trait_names)
    ncols = 4
    nrows = (n_traits + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 3 * nrows),
        subplot_kw={"projection": ccrs.EqualEarth()},
    )
    axes = np.asarray(axes).flatten()

    for j, trait in enumerate(trait_names):
        ax = axes[j]
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.2, color="gray")
        valid = masks[:, j]
        ax.scatter(
            lon[valid], lat[valid],
            c="#2ecc71", s=0.3, transform=ccrs.PlateCarree(), rasterized=True,
        )
        ax.scatter(
            lon[~valid], lat[~valid],
            c="#e74c3c", s=0.1, alpha=0.3, transform=ccrs.PlateCarree(), rasterized=True,
        )
        n_ok = int(valid.sum())
        ax.set_title(f"{trait} ({n_ok})", fontsize=8)

    # Hide unused axes
    for j in range(n_traits, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-trait validity (green = valid, red = invalid)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / "trait_validity_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def _plot_trait_mean_maps(
    lon: np.ndarray,
    lat: np.ndarray,
    histograms: np.ndarray,
    masks: np.ndarray,
    bin_edges_arr: np.ndarray,
    trait_names: list[str],
    descriptions: dict[str, str],
    out_dir: Path,
) -> None:
    """Spatial maps of histogram-weighted mean for key traits."""
    key_indices = [
        (j, t) for j, t in enumerate(trait_names) if t in _KEY_TRAITS
    ]
    if not key_indices:
        key_indices = [(j, t) for j, t in enumerate(trait_names)][:6]

    ncols = 3
    nrows = (len(key_indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 3.5 * nrows),
        subplot_kw={"projection": ccrs.EqualEarth()},
    )
    axes = np.asarray(axes).flatten()

    for i, (j, trait) in enumerate(key_indices):
        ax = axes[i]
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.2, color="gray")

        valid = masks[:, j]
        if not valid.any():
            ax.set_title(f"{trait} — no data", fontsize=9)
            continue

        wmean = _weighted_mean(histograms[valid, j, :], bin_edges_arr[j])
        sc = ax.scatter(
            lon[valid], lat[valid],
            c=wmean, s=0.5, cmap="plasma",
            transform=ccrs.PlateCarree(), rasterized=True,
        )
        fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
        label = _trait_label(trait, descriptions, max_len=35)
        ax.set_title(label, fontsize=8)

    for i in range(len(key_indices), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Histogram weighted mean by trait", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / "trait_mean_maps.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def _plot_entropy_map(
    lon: np.ndarray,
    lat: np.ndarray,
    histograms: np.ndarray,
    masks: np.ndarray,
    out_path: Path,
) -> None:
    """Spatial map of mean Shannon entropy across valid traits."""
    n_cells, n_traits, _ = histograms.shape
    entropy_per_trait = np.zeros((n_cells, n_traits), dtype=np.float32)
    for j in range(n_traits):
        entropy_per_trait[:, j] = _shannon_entropy(histograms[:, j, :])

    # Mean entropy across valid traits only
    masked_entropy = np.where(masks, entropy_per_trait, np.nan)
    mean_entropy = np.nanmean(masked_entropy, axis=1)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="gray")

    sc = ax.scatter(
        lon, lat,
        c=mean_entropy,
        s=1,
        cmap="coolwarm",
        transform=ccrs.PlateCarree(),
        rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Mean Shannon entropy (bits)")
    ax.set_title("Mean histogram entropy across valid traits")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def _plot_sample_distributions(
    histograms: np.ndarray,
    masks: np.ndarray,
    coords: np.ndarray,
    bin_edges_arr: np.ndarray,
    trait_names: list[str],
    descriptions: dict[str, str],
    out_path: Path,
) -> None:
    """Bar charts for a sample of cells across latitude bands."""
    sample_idx = _pick_sample_cells(coords, masks, n_per_band=1)
    if not sample_idx:
        log.warning("No sample cells found — skipping distribution plot")
        return

    # Determine which traits to show
    key_indices = [
        (j, t) for j, t in enumerate(trait_names) if t in _KEY_TRAITS
    ]
    if not key_indices:
        key_indices = [(j, t) for j, t in enumerate(trait_names)][:6]

    n_cells_sample = len(sample_idx)
    n_traits_sample = len(key_indices)
    fig, axes = plt.subplots(
        n_cells_sample, n_traits_sample,
        figsize=(3 * n_traits_sample, 2.5 * n_cells_sample),
        squeeze=False,
    )

    # Region labels for row titles
    region_names = list(_REGION_BANDS.keys())

    for row, cell_i in enumerate(sample_idx):
        region = region_names[row] if row < len(region_names) else f"Cell {cell_i}"
        for col, (j, trait) in enumerate(key_indices):
            ax = axes[row, col]
            edges = bin_edges_arr[j]
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = edges[1:] - edges[:-1]
            probs = histograms[cell_i, j, :]

            color = "#2ecc71" if masks[cell_i, j] else "#e74c3c"
            ax.bar(centers, probs, width=widths * 0.9, color=color, edgecolor="none")

            if row == 0:
                ax.set_title(_trait_label(trait, descriptions, max_len=25), fontsize=7)
            if col == 0:
                ax.set_ylabel(region, fontsize=8)
            ax.tick_params(labelsize=5)
            ax.set_ylim(0, None)

    fig.suptitle("Sample cell distributions (green = valid, red = invalid)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _write_markdown_report(
    report_dir: Path,
    attrs: dict,
    masks: np.ndarray,
    trait_names: list[str],
    descriptions: dict[str, str],
) -> None:
    """Write a summary markdown report with embedded figures."""
    stats = attrs.get("statistics", {})
    report_path = report_dir / "histogram_report.md"

    with open(report_path, "w") as f:
        f.write("# Histogram Target Sanity Check\n\n")
        f.write(f"**Source:** {attrs.get('source', 'unknown')}  \n")
        f.write(f"**CRS:** {attrs.get('crs', '')}  \n")
        f.write(f"**Resolution:** {attrs.get('target_resolution', '')}m  \n")
        f.write(f"**Bins:** {attrs.get('n_bins', '')}  \n")
        f.write(f"**Label smoothing:** {attrs.get('label_smoothing_epsilon', '')}  \n\n")

        f.write("## Cell statistics\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total cells | {stats.get('n_cells_total', 'N/A')} |\n")
        f.write(f"| Valid cells | {stats.get('n_cells_valid', 'N/A')} |\n")
        f.write(f"| Filtered (observations) | {stats.get('cells_filtered_observations', 'N/A')} |\n")
        f.write(f"| Filtered (species) | {stats.get('cells_filtered_species', 'N/A')} |\n")
        f.write(f"| Filtered (abundance) | {stats.get('cells_filtered_abundance', 'N/A')} |\n\n")

        f.write("## Per-trait valid cell counts\n\n")
        f.write("| Trait | Description | Valid cells | % |\n")
        f.write("|-------|-------------|-------------|---|\n")
        trait_counts = stats.get("trait_valid_counts", {})
        n_valid = stats.get("n_cells_valid", 1)
        for j, trait in enumerate(trait_names):
            desc = descriptions.get(trait, "")
            if len(desc) > 50:
                desc = desc[:47] + "..."
            count = trait_counts.get(trait, int(masks[:, j].sum()))
            pct = 100 * count / n_valid if n_valid > 0 else 0
            f.write(f"| {trait} | {desc} | {count} | {pct:.1f}% |\n")
        f.write("\n")

        f.write("## Figures\n\n")
        f.write("### Spatial coverage\n\n")
        f.write("![Spatial coverage](spatial_coverage.png)\n\n")
        f.write("### Trait validity grid\n\n")
        f.write("![Trait validity](trait_validity_grid.png)\n\n")
        f.write("### Trait mean maps (key traits)\n\n")
        f.write("![Trait means](trait_mean_maps.png)\n\n")
        f.write("### Entropy map\n\n")
        f.write("![Entropy](entropy_map.png)\n\n")
        f.write("### Sample cell distributions\n\n")
        f.write("![Sample distributions](sample_distributions.png)\n\n")

    log.info("Saved %s", report_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_histogram_report(
    zarr_path: Path,
    out_dir: Path,
    params_path: Path | None = None,
) -> None:
    """Generate a sanity-check report for a histogram Zarr store.

    Parameters
    ----------
    zarr_path : Path
        Path to ``histograms.zarr``.
    out_dir : Path
        Parent output directory. Report is written to ``out_dir/report/``.
    params_path : Path | None
        Path to params.yaml for parsing trait descriptions.
    """
    log.info("Generating histogram report for %s", zarr_path)

    root = zarr.open_group(zarr_path, mode="r")
    histograms = np.asarray(root["histograms"])
    masks = np.asarray(root["masks"])
    coords = np.asarray(root["coords"])
    bin_edges_arr = np.asarray(root["bin_edges"])
    attrs = dict(root.attrs)

    trait_names: list[str] = attrs.get("trait_names", [])
    descriptions: dict[str, str] = {}
    if params_path is not None and params_path.exists():
        descriptions = _parse_trait_descriptions(params_path)

    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Reproject to lon/lat once for all map plots
    lon, lat = _to_lonlat(coords)
    n_valid_per_cell = masks.sum(axis=1)

    # Generate figures
    _plot_spatial_coverage(lon, lat, n_valid_per_cell, report_dir / "spatial_coverage.png")
    _plot_trait_validity_grid(lon, lat, masks, trait_names, descriptions, report_dir)
    _plot_trait_mean_maps(
        lon, lat, histograms, masks, bin_edges_arr,
        trait_names, descriptions, report_dir,
    )
    _plot_entropy_map(lon, lat, histograms, masks, report_dir / "entropy_map.png")
    _plot_sample_distributions(
        histograms, masks, coords, bin_edges_arr,
        trait_names, descriptions, report_dir / "sample_distributions.png",
    )

    # Write markdown
    _write_markdown_report(report_dir, attrs, masks, trait_names, descriptions)

    log.info("Report complete: %s", report_dir)
