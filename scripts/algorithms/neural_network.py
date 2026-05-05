from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from algorithms.classification import build_tabular_preprocessor


MLP_CONFIGS: list[dict[str, Any]] = [
    {
        "config_name": "mlp_64_32_baseline",
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "alpha": 0.001,
        "learning_rate_init": 0.001,
        "batch_size": 64,
    },
    {
        "config_name": "mlp_96_48_regularized",
        "hidden_layer_sizes": (96, 48),
        "activation": "relu",
        "alpha": 0.003,
        "learning_rate_init": 0.0008,
        "batch_size": 64,
    },
    {
        "config_name": "mlp_128_64_small_lr",
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "alpha": 0.002,
        "learning_rate_init": 0.0005,
        "batch_size": 64,
    },
    {
        "config_name": "mlp_128_64_32_deep",
        "hidden_layer_sizes": (128, 64, 32),
        "activation": "relu",
        "alpha": 0.004,
        "learning_rate_init": 0.0006,
        "batch_size": 64,
    },
    {
        "config_name": "mlp_80_40_tanh",
        "hidden_layer_sizes": (80, 40),
        "activation": "tanh",
        "alpha": 0.002,
        "learning_rate_init": 0.0007,
        "batch_size": 64,
    },
]


def build_mlp_pipeline(
    X: pd.DataFrame,
    config: dict[str, Any],
    max_iter: int,
    validation_fraction: float,
    random_state: int = 42,
) -> Pipeline:
    preprocessor = build_tabular_preprocessor(
        X,
        scale_numeric=True,
        dense_output=True,
    )
    classifier = MLPClassifier(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        activation=config["activation"],
        alpha=config["alpha"],
        batch_size=config["batch_size"],
        early_stopping=True,
        learning_rate_init=config["learning_rate_init"],
        max_iter=max_iter,
        n_iter_no_change=35,
        random_state=random_state,
        validation_fraction=validation_fraction,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
