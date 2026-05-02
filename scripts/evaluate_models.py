from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_model import (
    DATA_PATH,
    FIGURES_PATH,
    MODEL_NAMES,
    TARGET,
    build_features,
    build_pipeline,
    latest_season_split,
    top10_by_race_score,
)


OUTPUT_PATH = Path("outputs")
MODEL_COMPARISON_PATH = OUTPUT_PATH / "model_comparison.csv"
ROLLING_BACKTEST_PATH = OUTPUT_PATH / "rolling_backtest.csv"
SUMMARY_PATH = OUTPUT_PATH / "model_selection_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare several top-10 prediction algorithms.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--test-season", type=int, default=None)
    parser.add_argument("--min-test-rows", type=int, default=200)
    parser.add_argument("--min-train-seasons", type=int, default=2)
    parser.add_argument("--skip-rolling", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    return parser.parse_args()


def evaluate_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, float | int | str | None]:
    X_train = build_features(train_df)
    y_train = train_df[TARGET].astype(int)
    X_test = build_features(test_df)
    y_test = test_df[TARGET].astype(int)

    model = build_pipeline(X_train, model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics: dict[str, float | int | str | None] = {
        "model": model_name,
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
    return metrics


def run_holdout_comparison(
    df: pd.DataFrame,
    model_names: list[str],
    test_season: int | None,
    min_test_rows: int,
) -> pd.DataFrame:
    train_df, test_df = latest_season_split(df, test_season, min_test_rows)
    rows = [evaluate_model(model_name, train_df, test_df) for model_name in model_names]
    return pd.DataFrame(rows).sort_values(
        ["race_precision_at_10", "f1", "roc_auc"],
        ascending=[False, False, False],
    )


def run_rolling_backtest(
    df: pd.DataFrame,
    model_names: list[str],
    min_train_seasons: int,
) -> pd.DataFrame:
    seasons = sorted(df["season"].dropna().astype(int).unique())
    rows = []

    for season in seasons:
        train_df = df[df["season"] < season].copy()
        test_df = df[df["season"] == season].copy()
        train_seasons = sorted(train_df["season"].dropna().astype(int).unique())

        if len(train_seasons) < min_train_seasons or test_df.empty:
            continue

        for model_name in model_names:
            rows.append(evaluate_model(model_name, train_df, test_df))

    return pd.DataFrame(rows)


def save_comparison_figure(holdout_df: pd.DataFrame) -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    plot_df = holdout_df.melt(
        id_vars=["model"],
        value_vars=["f1", "roc_auc", "race_precision_at_10"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.title("Holdout Model Comparison")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "model_comparison.png", dpi=160)
    plt.close()


def save_rolling_figure(rolling_df: pd.DataFrame) -> None:
    if rolling_df.empty:
        return

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=rolling_df,
        x="test_season",
        y="race_precision_at_10",
        hue="model",
        marker="o",
    )
    plt.ylim(0, 1)
    plt.title("Rolling Season Backtest")
    plt.xlabel("Test season")
    plt.ylabel("Race precision@10")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "rolling_backtest.png", dpi=160)
    plt.close()


def summarize_results(holdout_df: pd.DataFrame, rolling_df: pd.DataFrame) -> dict[str, object]:
    best_holdout = holdout_df.iloc[0].to_dict()
    summary: dict[str, object] = {
        "best_holdout_model": best_holdout["model"],
        "best_holdout": best_holdout,
    }

    if not rolling_df.empty:
        rolling_summary = (
            rolling_df.groupby("model", as_index=False)[
                ["accuracy", "precision", "recall", "f1", "roc_auc", "race_precision_at_10"]
            ]
            .mean(numeric_only=True)
            .sort_values(["race_precision_at_10", "f1", "roc_auc"], ascending=[False, False, False])
        )
        summary["best_rolling_model"] = rolling_summary.iloc[0]["model"]
        summary["rolling_average"] = rolling_summary.to_dict(orient="records")

    return summary


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data}")

    df = pd.read_csv(args.data)
    if df.empty:
        raise ValueError(f"Dataset is empty: {args.data}")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    holdout_df = run_holdout_comparison(
        df=df,
        model_names=args.models,
        test_season=args.test_season,
        min_test_rows=args.min_test_rows,
    )
    holdout_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    save_comparison_figure(holdout_df)

    if args.skip_rolling:
        rolling_df = pd.DataFrame()
    else:
        rolling_df = run_rolling_backtest(
            df=df,
            model_names=args.models,
            min_train_seasons=args.min_train_seasons,
        )
        rolling_df.to_csv(ROLLING_BACKTEST_PATH, index=False)
        save_rolling_figure(rolling_df)

    summary = summarize_results(holdout_df, rolling_df)
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Model evaluation complete.")
    print(f"Holdout comparison: {MODEL_COMPARISON_PATH}")
    if not args.skip_rolling:
        print(f"Rolling backtest: {ROLLING_BACKTEST_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
