"""
Handles model training and hyperparameter optimization for RandomForest, XGBoost, and LightGBM.
"""

from lightgbm import LGBMRegressor

def train_lgbm(X_train, y_train, params=None):
    """Train the final LightGBM model using tuned parameters."""
    if params is None:
        params = {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "num_leaves": 127,
            "subsample": 0.6,
            "colsample_bytree": 1.0,
            "random_state": 42,
            "n_jobs": -1
        }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model
