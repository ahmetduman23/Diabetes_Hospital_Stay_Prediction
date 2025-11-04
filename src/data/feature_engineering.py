"""
Feature construction and transformation functions used before training.
"""

import numpy as np
import pandas as pd

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new aggregated or ratio-based features."""
    df["meds_per_day"] = df["num_medications"] / df["time_in_hospital"].clip(lower=1)
    df["labs_per_day"] = df["num_lab_procedures"] / df["time_in_hospital"].clip(lower=1)
    return df
