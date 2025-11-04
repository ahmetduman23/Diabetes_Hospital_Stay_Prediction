"""
Metric aggregation and comparison tools across models.
"""

import pandas as pd

def summarize_results(results_dict):
    """Turn model evaluation results into a DataFrame."""
    df = pd.DataFrame(results_dict).T
    return df.sort_values(by="r2", ascending=False)
