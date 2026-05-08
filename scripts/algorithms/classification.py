from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CLASSIFICATION_MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "neural_network_mlp",
]

CLASSIFICATION_ALGORITHM_SUMMARY = {
    "logistic_regression": "Linear baseline with class balancing and scaled numeric features.",
    "random_forest": "Bagging tree ensemble used as the current holdout champion.",
    "extra_trees": "More randomized tree ensemble used as a second ensemble comparison.",
    "hist_gradient_boosting": "Boosted tree model used as a strong non-linear comparison.",
    "neural_network_mlp": "Simple multilayer perceptron classifier used as the neural-network baseline.",
}


def build_tabular_preprocessor(
    X: pd.DataFrame,
    *,
    scale_numeric: bool,
    dense_output: bool,
) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=not dense_output,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0.0 if dense_output else 0.3,
    )


def build_classification_estimator(model_name: str, random_state: int = 42):
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=random_state,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=random_state,
        )
    if model_name == "neural_network_mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.001,
            batch_size=64,
            early_stopping=True,
            learning_rate_init=0.001,
            max_iter=350,
            n_iter_no_change=25,
            random_state=random_state,
        )

    raise ValueError(f"Unknown classification model: {model_name}")


def build_classification_pipeline(
    X: pd.DataFrame,
    model_name: str = "random_forest",
    random_state: int = 42,
) -> Pipeline:
    scale_numeric = model_name in {"logistic_regression", "neural_network_mlp"}
    dense_output = model_name in {"hist_gradient_boosting", "neural_network_mlp"}
    preprocessor = build_tabular_preprocessor(
        X,
        scale_numeric=scale_numeric,
        dense_output=dense_output,
    )
    classifier = build_classification_estimator(model_name, random_state=random_state)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
