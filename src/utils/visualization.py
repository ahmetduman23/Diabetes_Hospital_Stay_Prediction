"""
Reusable visualization helpers for scatter plots, learning curves, and feature importances.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, cv=5):
    """Draw train–validation learning curves."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring="r2", n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 8)
    )
    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train", color="royalblue")
    plt.plot(train_sizes, np.mean(val_scores, axis=1), label="Validation", color="darkorange")
    plt.title("Learning Curve")
    plt.xlabel("Training Samples")
    plt.ylabel("R² Score")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
