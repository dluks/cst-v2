import pandas as pd


def filter_certain_plots(df: pd.DataFrame, givd_col: str, givd: str) -> pd.DataFrame:
    """Filter out certain plots."""
    return df[df[givd_col] != givd]
