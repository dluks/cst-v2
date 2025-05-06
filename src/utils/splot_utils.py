import pandas as pd


def filter_certain_plots(df: pd.DataFrame, givd_nu: str) -> pd.DataFrame:
    """Filter out certain plots."""
    return df[df["GIVD_NU"] != givd_nu]
