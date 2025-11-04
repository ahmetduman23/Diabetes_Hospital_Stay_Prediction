"""
src — Core package for the Hospital Stay Prediction project.

This package provides a modular implementation for:
- 📦 Data loading & preprocessing
- 🧩 Feature engineering
- 🤖 Model training & evaluation
- 🔍 Explainability (SHAP, feature importance)
- 📊 Visualization & utility functions

Typical usage example:
----------------------
from src.data.load_data import load_processed_data
from src.models.train_model import train_lgbm
from src.utils.visualization import plot_learning_curve

df = load_processed_data("../data/processed/cleaned_data.csv")
model = train_lgbm(X_train, y_train)
plot_learning_curve(model, X_train, y_train)
"""

# Expose subpackages
from . import data
from . import models
from . import utils

__all__ = ["data", "models", "utils"]

# Optional: metadata
__version__ = "1.0.0"
__author__ = "Ahmet Yasir Duman"
__email__ = "ahmetyasirduman@outlook.com"
__project__ = "Hospital Stay Prediction — LightGBM Explainability Pipeline"
