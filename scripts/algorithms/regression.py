from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from algorithms.classification import build_tabular_preprocessor


REGRESSION_MODEL_NAMES = [
    "hist_gradient_boosting_regressor",
    "random_forest_regressor",
    "neural_network_mlp_regressor",
]

REGRESSION_ALGORITHM_SUMMARY = {
    "hist_gradient_boosting_regressor": "Boosted tree model that predicts raw finishing position.",
    "random_forest_regressor": "Tree ensemble rank baseline for finish-position prediction.",
    "neural_network_mlp_regressor": "Neural-network regressor used to rank every driver in a race.",
}


def build_regression_estimator(model_name: str, random_state: int = 42):
    if model_name == "hist_gradient_boosting_regressor":
        return HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=random_state,
        )
    if model_name == "random_forest_regressor":
        return RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "neural_network_mlp_regressor":
        return MLPRegressor(
            hidden_layer_sizes=(96, 48),
            activation="relu",
            alpha=0.003,
            batch_size=64,
            early_stopping=True,
            learning_rate_init=0.0008,
            max_iter=450,
            n_iter_no_change=30,
            random_state=random_state,
        )

    raise ValueError(f"Unknown position model: {model_name}")


def build_regression_pipeline(
    X: pd.DataFrame,
    model_name: str,
    random_state: int = 42,
) -> Pipeline:
    is_neural = model_name == "neural_network_mlp_regressor"
    dense_output = model_name in {"hist_gradient_boosting_regressor", "neural_network_mlp_regressor"}
    preprocessor = build_tabular_preprocessor(
        X,
        scale_numeric=is_neural,
        dense_output=dense_output,
    )
    regressor = build_regression_estimator(model_name, random_state=random_state)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )
