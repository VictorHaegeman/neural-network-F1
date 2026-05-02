from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_model import (
    DATA_PATH,
    FIGURES_PATH,
    TARGET,
    build_features,
    latest_season_split,
    top10_by_race_score,
)


OUTPUT_PATH = Path("outputs")
TUNING_PATH = OUTPUT_PATH / "neural_network_tuning.csv"
SUMMARY_PATH = OUTPUT_PATH / "neural_network_summary.json"
MODEL_PATH = OUTPUT_PATH / "models" / "top10_neural_network_mlp.joblib"

RANDOM_STATE = 42

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune and train a dedicated neural-network MLP baseline.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--test-season", type=int, default=None)
    parser.add_argument("--min-test-rows", type=int, default=200)
    parser.add_argument("--max-iter", type=int, default=650)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--force", action="store_true", help="Re-run every MLP config even if tuning CSV exists.")
    parser.add_argument(
        "--refit-full",
        action="store_true",
        help="Refit the best MLP on the full dataset after tuning. Slower; off by default.",
    )
    return parser.parse_args()


def build_mlp_pipeline(
    X: pd.DataFrame,
    config: dict[str, Any],
    max_iter: int,
    validation_fraction: float,
) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=2,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        sparse_threshold=0.0,
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
        random_state=RANDOM_STATE,
        validation_fraction=validation_fraction,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_config(
    config: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_iter: int,
    validation_fraction: float,
) -> tuple[dict[str, Any], Pipeline]:
    X_train = build_features(train_df)
    y_train = train_df[TARGET].astype(int)
    X_test = build_features(test_df)
    y_test = test_df[TARGET].astype(int)

    model = build_mlp_pipeline(X_train, config, max_iter, validation_fraction)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    classifier = model.named_steps["classifier"]

    metrics: dict[str, Any] = {
        "config_name": config["config_name"],
        "hidden_layer_sizes": str(config["hidden_layer_sizes"]),
        "activation": config["activation"],
        "alpha": float(config["alpha"]),
        "learning_rate_init": float(config["learning_rate_init"]),
        "batch_size": int(config["batch_size"]),
        "n_iter": int(getattr(classifier, "n_iter_", 0)),
        "best_validation_score": float(getattr(classifier, "best_validation_score_", 0.0) or 0.0),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start_season": int(train_df["season"].min()),
        "train_end_season": int(train_df["season"].max()),
        "test_season": int(test_df["season"].iloc[0]),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
    }
    metrics.update(top10_by_race_score(test_df, probabilities))
    return metrics, model


def save_tuning_figure(results: pd.DataFrame) -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    plot_df = results.melt(
        id_vars=["config_name"],
        value_vars=["f1", "roc_auc", "race_precision_at_10"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(11, 5))
    sns.barplot(data=plot_df, x="config_name", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.title("Neural Network MLP Tuning")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "neural_network_tuning.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data}")

    df = pd.read_csv(args.data)
    if df.empty:
        raise ValueError(f"Dataset is empty: {args.data}")

    train_df, test_df = latest_season_split(df, args.test_season, args.min_test_rows)

    fitted_models: dict[str, Pipeline] = {}

    if TUNING_PATH.exists() and not args.force:
        results = pd.read_csv(TUNING_PATH)
        print(f"Reusing existing tuning results from {TUNING_PATH}. Use --force to recompute.")
    else:
        rows: list[dict[str, Any]] = []
        for config in MLP_CONFIGS:
            metrics, model = evaluate_config(
                config=config,
                train_df=train_df,
                test_df=test_df,
                max_iter=args.max_iter,
                validation_fraction=args.validation_fraction,
            )
            rows.append(metrics)
            fitted_models[config["config_name"]] = model
            print(json.dumps(metrics, indent=2), flush=True)

        results = pd.DataFrame(rows).sort_values(
            ["race_precision_at_10", "f1", "roc_auc"],
            ascending=[False, False, False],
        )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(TUNING_PATH, index=False)
    save_tuning_figure(results)

    best = results.iloc[0].to_dict()
    best_config_name = str(best["config_name"])
    best_config = next(config for config in MLP_CONFIGS if config["config_name"] == best_config_name)

    if args.refit_full:
        final_X = build_features(df)
        final_y = df[TARGET].astype(int)
        saved_model = build_mlp_pipeline(final_X, best_config, args.max_iter, args.validation_fraction)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            saved_model.fit(final_X, final_y)
        training_scope = "full_dataset"
    elif best_config_name in fitted_models:
        saved_model = fitted_models[best_config_name]
        training_scope = "holdout_train_split"
    else:
        X_train = build_features(train_df)
        y_train = train_df[TARGET].astype(int)
        saved_model = build_mlp_pipeline(X_train, best_config, args.max_iter, args.validation_fraction)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            saved_model.fit(X_train, y_train)
        training_scope = "holdout_train_split_rebuilt_from_existing_results"

    joblib.dump(saved_model, MODEL_PATH)

    summary = {
        "best_neural_network_config": best_config_name,
        "best_holdout": best,
        "model_path": str(MODEL_PATH),
        "model_training_scope": training_scope,
        "tuning_path": str(TUNING_PATH),
        "figure_path": str(FIGURES_PATH / "neural_network_tuning.png"),
        "note": "Dedicated neural-network artifact is trained separately from the main champion model.",
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nNeural network tuning complete.")
    print(f"Tuning results: {TUNING_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
