"""
Model explainability utilities — SHAP & feature importance.
"""

import shap
import matplotlib.pyplot as plt
import pandas as pd

def explain_global(model, X_sample, top_n=15):
    """Plot SHAP summary (bar + swarm) for global importance."""
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=True)
    shap.summary_plot(shap_values, X_sample, show=True)
    return shap_values

def explain_local(model, X_sample, index=0):
    """Generate SHAP force plot for a single instance."""
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample.iloc[index:index+1])
    shap.initjs()
    return shap.force_plot(explainer.expected_value, shap_values.values, X_sample.iloc[index:index+1])
