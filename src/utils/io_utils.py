"""
Input–output utilities for saving/loading models and datasets.
"""

import joblib

def save_model(model, path):
    """Save a trained model to disk."""
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

def load_model(path):
    """Load a trained model from disk."""
    return joblib.load(path)
