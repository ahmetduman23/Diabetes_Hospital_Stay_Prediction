"""
Handles all dataset loading operations.
Used in both data exploration and model training notebooks.
"""

import pandas as pd

def load_processed_data(path: str) -> pd.DataFrame:
    """Load the cleaned dataset (CSV) for modeling."""
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset from {path} — shape: {df.shape}")
    return df
