"""
Cleans raw data: handles missing values, categorical encoding, and numeric normalization.
"""

import pandas as pd
import numpy as np

def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing or invalid entries with appropriate replacements."""
    df = df.replace({"?": np.nan, "Unknown/Invalid": np.nan})
    df = df.fillna("Unknown")
    return df
