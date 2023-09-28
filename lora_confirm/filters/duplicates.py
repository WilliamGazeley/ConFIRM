import pandas as pd
from typing import List

def _duplicates_axis_0(
        df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Filter out generated rows that are duplicates based on specific columns.
    """
    return df.drop_duplicates(subset=columns, keep='first')

def _duplicates_axis_1(
        df: pd.DataFrame, rows: List[str]) -> pd.DataFrame:
    """
    Filter out generated columns that are duplicates based on specific rows.
    """
    df_T = df.T
    df_T['tup'] = df_T.loc[rows].apply(tuple, axis=1)
    df_T = df_T.drop_duplicates('tup')
    df_T = df_T.drop('tup', axis=1)
    return df_T.T

def duplicates(df: pd.DataFrame, columns: List[str], axis=0) -> pd.DataFrame:
    """
    Filter out generated questions that are duplicates.
    """
    if axis == 0:
        return _duplicates_axis_0(df, columns)
    elif axis == 1:
        return _duplicates_axis_1(df, columns)
    else:
        raise ValueError("Axis must be 0 or 1")
