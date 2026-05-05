from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from algorithms.regression import REGRESSION_MODEL_NAMES, build_regression_pipeline as build_position_pipeline
from train_model import DATA_PATH, RANDOM_STATE, build_features, latest_season_split


OUTPUT_PATH = Path("outputs")
FIGURES_PATH = OUTPUT_PATH / "figures"
MODEL_PATH = OUTPUT_PATH / "models" / "finish_position_regressor.joblib"
NEURAL_MODEL_PATH = OUTPUT_PATH / "models" / "finish_position_neural_network_mlp.joblib"
METRICS_PATH = OUTPUT_PATH / "position_model_metrics.json"
COMPARISON_PATH = OUTPUT_PATH / "position_model_comparison.csv"
PREDICTION_PATH = OUTPUT_PATH / "predictions" / "position_predictions_holdout.csv"
TARGET = "final_position"

MODEL_NAMES = REGRESSION_MODEL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a finish-position ranking model.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--test-season", type=int, default=None)
    parser.add_argument("--min-test-rows", type=int, default=200)
    return parser.parse_args()


def build_regression_pipeline(X: pd.DataFrame, model_name: str) -> Pipeline:
    return build_position_pipeline(X, model_name=model_name, random_state=RANDOM_STATE)


def add_rank_predictions(test_df: pd.DataFrame, raw_predictions: np.ndarray) -> pd.DataFrame:
    scored = test_df[
        [
            "race_id",
            "season",
            "round",
            "grand_prix",
            "driver_id",
            "driver_code",
            "driver_name",
            "constructor_name",
            "grid",
            "top10_finish",
            "final_position",
        ]
    ].copy()
    scored["predicted_finish_position_raw"] = raw_predictions
    scored["predicted_finish_rank"] = (
        scored.groupby("race_id")["predicted_finish_position_raw"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    scored["predicted_top10_position"] = scored["predicted_finish_rank"].where(
        scored["predicted_finish_rank"] <= 10,
        0,
    )
    return scored.sort_values(["season", "round", "predicted_finish_rank", "driver_id"])


def ranking_metrics(scored: pd.DataFrame) -> dict[str, float]:
    race_precision = []
    all_rank_mae = []
    actual_top10_rank_mae = []
    predicted_top10_rank_mae = []
    spearman_scores = []

    for _, group in scored.groupby("race_id"):
        actual_position = pd.to_numeric(group["final_position"], errors="coerce")
        predicted_rank = pd.to_numeric(group["predicted_finish_rank"], errors="coerce")
        actual_top10 = group[actual_position <= 10]
        predicted_top10 = group.nsmallest(min(10, len(group)), "predicted_finish_rank")

        race_precision.append(float((predicted_top10["final_position"] <= 10).mean()))
        all_rank_mae.append(float((predicted_rank - actual_position).abs().mean()))

        if not actual_top10.empty:
            actual_top10_rank_mae.append(
                float((actual_top10["predicted_finish_rank"] - actual_top10["final_position"]).abs().mean())
            )
        if not predicted_top10.empty:
            predicted_top10_rank_mae.append(
                float((predicted_top10["predicted_finish_rank"] - predicted_top10["final_position"]).abs().mean())
            )

        if len(group) > 1:
            corr = group[["final_position", "predicted_finish_rank"]].corr(method="spearman").iloc[0, 1]
            if not pd.isna(corr):
                spearman_scores.append(float(corr))

    return {
        "race_precision_at_10": float(np.mean(race_precision)) if race_precision else 0.0,
        "all_rank_mae": float(np.mean(all_rank_mae)) if all_rank_mae else 0.0,
        "actual_top10_rank_mae": float(np.mean(actual_top10_rank_mae)) if actual_top10_rank_mae else 0.0,
        "predicted_top10_rank_mae": float(np.mean(predicted_top10_rank_mae)) if predicted_top10_rank_mae else 0.0,
        "mean_spearman_by_race": float(np.mean(spearman_scores)) if spearman_scores else 0.0,
    }


def evaluate_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, float | int | str], Pipeline, pd.DataFrame]:
    X_train = build_features(train_df)
    y_train = train_df[TARGET].astype(float)
    X_test = build_features(test_df)
    y_test = test_df[TARGET].astype(float)

    model = build_regression_pipeline(X_train, model_name)
    model.fit(X_train, y_train)
    raw_predictions = np.clip(model.predict(X_test), 1, 30)
    scored = add_rank_predictions(test_df, raw_predictions)

    metrics: dict[str, float | int | str] = {
        "model": model_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_season": int(test_df["season"].iloc[0]),
        "position_mae_raw": float(mean_absolute_error(y_test, raw_predictions)),
        "position_rmse_raw": float(np.sqrt(mean_squared_error(y_test, raw_predictions))),
    }
    metrics.update(ranking_metrics(scored))
    return metrics, model, scored


def save_comparison_chart(comparison: pd.DataFrame) -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    plot_df = comparison.melt(
        id_vars=["model"],
        value_vars=["race_precision_at_10", "actual_top10_rank_mae", "mean_spearman_by_race"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric")
    plt.xticks(rotation=15, ha="right")
    plt.title("Finish Position Model Comparison")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "position_model_comparison.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data}")

    df = pd.read_csv(args.data)
    df = df[pd.to_numeric(df[TARGET], errors="coerce") > 0].copy()
    train_df, test_df = latest_season_split(df, args.test_season, args.min_test_rows)

    rows: list[dict[str, float | int | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    scored_outputs: dict[str, pd.DataFrame] = {}
    for model_name in MODEL_NAMES:
        metrics, model, scored = evaluate_model(model_name, train_df, test_df)
        rows.append(metrics)
        fitted_models[model_name] = model
        scored_outputs[model_name] = scored
        print(json.dumps(metrics, indent=2), flush=True)

    comparison = pd.DataFrame(rows).sort_values(
        ["race_precision_at_10", "actual_top10_rank_mae", "position_mae_raw"],
        ascending=[False, True, True],
    )
    best = comparison.iloc[0].to_dict()
    best_model_name = str(best["model"])

    final_X = build_features(df)
    final_y = df[TARGET].astype(float)
    best_model = build_regression_pipeline(final_X, best_model_name)
    best_model.fit(final_X, final_y)

    neural_model = build_regression_pipeline(final_X, "neural_network_mlp_regressor")
    neural_model.fit(final_X, final_y)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(neural_model, NEURAL_MODEL_PATH)
    comparison.to_csv(COMPARISON_PATH, index=False)
    scored_outputs[best_model_name].to_csv(PREDICTION_PATH, index=False)
    save_comparison_chart(comparison)

    summary = {
        "best_position_model": best_model_name,
        "best_holdout": best,
        "model_path": str(MODEL_PATH),
        "neural_model_path": str(NEURAL_MODEL_PATH),
        "comparison_path": str(COMPARISON_PATH),
        "holdout_predictions_path": str(PREDICTION_PATH),
        "figure_path": str(FIGURES_PATH / "position_model_comparison.png"),
        "note": "This model predicts race finishing order/rank; it complements the binary top-10 classifier.",
    }
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Finish-position training complete.")
    print(f"Model: {MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
