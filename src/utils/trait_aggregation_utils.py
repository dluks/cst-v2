from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

from src.conf.environment import file_log, log

########################################################
### Community-weighted stats ###
########################################################


def _cwm(data: pd.Series, weights: pd.Series) -> float | Any:
    """Calculate the community-weighted mean."""
    return np.average(data, weights=weights)


def _cw_std(data: pd.Series, weights: pd.Series) -> float:
    """Calculate the community-weighted standard deviation."""
    return np.sqrt(np.average((data - _cwm(data, weights)) ** 2, weights=weights))


def _cw_quantile(data: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Calculate the community-weighted quantile."""
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]

    cumsum = np.cumsum(sorted_weights)
    quantile_value = sorted_data[cumsum >= quantile][0]
    return quantile_value


def cw_stats(g: pd.DataFrame, col: str, abund_col: str) -> pd.Series:
    """Calculate all community-weighted stats per plot."""
    # Normalize the abundances to sum to 1. Important when not all species in a plot are
    # present in the trait data.
    normalized_abund = g[abund_col] / g[abund_col].sum()
    if g.empty:
        log.warning("Empty group detected, returning NaNs...")
        return pd.Series(
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            index=["cwm", "cw_std", "cw_med", "cw_q05", "cw_q95"],
        )
    return pd.Series(
        [
            _cwm(g[col], normalized_abund),
            _cw_std(g[col], normalized_abund),
            _cw_quantile(g[col].to_numpy(), normalized_abund.to_numpy(), 0.5),
            _cw_quantile(g[col].to_numpy(), normalized_abund.to_numpy(), 0.05),
            _cw_quantile(g[col].to_numpy(), normalized_abund.to_numpy(), 0.95),
        ],
        index=["cwm", "cw_std", "cw_med", "cw_q05", "cw_q95"],
    )


########################################################
### Functional diversity metrics ###
########################################################


def fd_metrics(
    df: pd.DataFrame,
    trait_cols: list,
    stats: list[str],
    species_col: str,
    abundance_col: str | None = None,
    use_ses: bool = False,
    random_seed: int | None = None,
) -> pd.Series:
    """Calculate all functional diversity stats per plot.

    Args:
        df: DataFrame containing species observations for a single plot
        trait_cols: List of column names containing traits or PCA components
        stats: List of stats to calculate
        abundance_col: Column name containing abundances
        use_ses: Whether to include standardized effect size for functional richness
        random_seed: Random seed for SES calculations

    Returns:
        Series with functional diversity metrics
    """
    # Check if we have enough data to calculate metrics
    if df.empty or len(df) < 2:
        return pd.Series({s: np.nan for s in stats})

    try:
        group_name = str(df.name)
    except AttributeError:
        group_name = "no_group_name"

    # Extract trait matrix and normalize abundances
    trait_matrix = df[trait_cols].values

    if abundance_col is not None:
        # Convert abundances to numpy array before normalization to avoid ArrowExtensionArray issues
        df = df.drop(columns=[species_col])  # We no longer need this column
        abundances = df[abundance_col].to_numpy()

        if not np.isclose(np.sum(abundances), 1.0):
            normalized_abund = abundances / np.sum(abundances)
        else:
            normalized_abund = abundances.copy()

    else:
        # No abundance column provided, so we identify multiple observations of the same
        # species and calculate proportional abundances.
        # TODO: This might interfere with grid cell-level f_ric...
        total_obs = len(df)
        df = (
            df.groupby([species_col, *trait_cols])
            .size()
            .div(total_obs)
            .reset_index(name="Rel_Abund")
            .drop(columns=[species_col])
        )
        trait_matrix = df[trait_cols].to_numpy()
        normalized_abund = df["Rel_Abund"].to_numpy()

        if len(trait_matrix) != len(normalized_abund):
            raise ValueError(
                "Trait matrix and normalized abundances have different lengths"
            )

    calculated_stats = {s: 0.0 for s in stats}
    if "sp_ric" in stats:
        # Calculate species richness (number of species in the plot)
        calculated_stats["sp_ric"] = len(df)
    if "f_ric" in stats:
        # Calculate functional richness
        f_ric = _calculate_fric(trait_matrix)
        calculated_stats["f_ric"] = f_ric
    if "f_eve" in stats:
        # Calculate functional evenness
        f_eve = calculate_feve(trait_matrix, normalized_abund)
        calculated_stats["f_eve"] = f_eve
    if "f_div" in stats:
        # Calculate functional divergence
        f_div = calculate_mean_pairwise_dissimilarity(trait_matrix, normalized_abund)
        calculated_stats["f_div"] = f_div
    if "f_red" in stats:
        # Calculate functional redundancy (1 - functional divergence)
        f_red = 1 - f_div if not np.isnan(f_div) else np.nan
        calculated_stats["f_red"] = f_red

    result = pd.Series(calculated_stats)

    return result


def _calculate_fric(trait_matrix: np.ndarray) -> float:
    """Calculate functional richness (FRic)."""
    if len(trait_matrix) < 3:
        return np.nan

    if trait_matrix.shape[0] < trait_matrix.shape[1]:
        return np.nan

    convex_hull = ConvexHull(trait_matrix)
    return convex_hull.volume


def calculate_feve(trait_matrix: np.ndarray, abundances: np.ndarray) -> float:
    """
    Calculate functional evenness (FEve).

    Parameters
    ----------
    trait_matrix : numpy.ndarray
        Matrix of species trait values (species × traits)
    abundances : numpy.ndarray
        Abundance weights for each species

    Returns
    -------
    float
        Functional evenness value between 0 and 1
    """
    # Ensure abundances is a 1D array
    abundances = abundances.flatten()

    # FEve requires at least 3 species
    if len(trait_matrix) < 3:
        return np.nan

    # Calculate distance matrix
    dist_matrix = squareform(pdist(trait_matrix, metric="euclidean"))

    # Get minimum spanning tree
    mst = minimum_spanning_tree(dist_matrix).toarray()

    # Get branch lengths and their corresponding nodes from MST
    branch_lengths = []
    parent_nodes = []
    child_nodes = []

    for i in range(len(trait_matrix)):
        for j in range(i + 1, len(trait_matrix)):
            if mst[i, j] > 0:
                branch_lengths.append(mst[i, j])
                parent_nodes.append(i)
                child_nodes.append(j)

    # Calculate EW (weighted branch lengths) - divide branch length by the sum of abundances
    ew = np.array(branch_lengths) / (abundances[parent_nodes] + abundances[child_nodes])

    # Calculate PEW (proportion of each branch in total)
    pew = ew / np.sum(ew)

    # Number of species
    S = len(trait_matrix)
    one_over_s_minus_one = 1 / (S - 1)

    # Calculate functional evenness using the formula from Villéger et al. (2008)
    feve = (np.sum(np.minimum(pew, one_over_s_minus_one)) - one_over_s_minus_one) / (
        1 - one_over_s_minus_one
    )

    return feve


def calculate_mean_pairwise_dissimilarity(
    trait_matrix: np.ndarray, abundances: np.ndarray, metric: Any = "euclidean"
) -> float:
    """
    Calculate Mean Pairwise Dissimilarity (MPD).
    MPD = Rao / Simpson

    Args:
        trait_matrix: Matrix of species trait values (species × traits)
        abundances: Abundance weights for each species
        metric: Distance metric to use (default: "euclidean")

    Returns:
        Mean Pairwise Dissimilarity (MPD)
    """
    rao = calculate_rao_quadratic_entropy(trait_matrix, abundances, metric)
    simpson = calculate_simpson_diversity(abundances)

    # Avoid division by zero
    if simpson > 0:
        return rao / simpson
    else:
        return np.nan


def calculate_simpson_diversity(abundances: np.ndarray) -> float:
    """
    Calculate Simpson diversity index.

    Args:
        abundances: Array of species abundances

    Returns:
        Simpson diversity (1 - sum of squared relative abundances)
    """
    # Simpson index = 1 - sum(p_i^2)
    simpson = 1 - np.sum(abundances**2)

    assert isinstance(simpson, float)
    return simpson


def calculate_rao_quadratic_entropy(
    trait_matrix: np.ndarray,
    abundances: np.ndarray,
    metric: Any = "euclidean",
) -> float:
    """
    Calculate Rao's quadratic entropy.

    Args:
        trait_matrix: Matrix of species trait values (species × traits)
        abundances: Abundance weights for each species
        metric: Distance metric to use (default: "euclidean")

    Returns:
        Rao's quadratic entropy (abundance-weighted mean of trait distances)
    """
    if len(trait_matrix) < 2:
        return np.nan

    # Calculate distance matrix
    dist_matrix = squareform(pdist(trait_matrix, metric=metric))

    # Calculate Rao's quadratic entropy
    rao_qe = abundances @ dist_matrix @ abundances

    # The result is a scalar, but we use item() to extract the float value
    rao_qe = rao_qe.item() / 2  # Divide by 2 because we double-count each pair

    return rao_qe
