from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from train_model import DATA_PATH, FIGURES_PATH, TARGET, save_dataset_figures


OUTPUTS_PATH = Path("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate project figures from datasets and metrics.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model-comparison", type=Path, default=OUTPUTS_PATH / "model_comparison.csv")
    parser.add_argument("--rolling-backtest", type=Path, default=OUTPUTS_PATH / "rolling_backtest.csv")
    return parser.parse_args()


def save_rows_by_season(df: pd.DataFrame) -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    season_counts = df.groupby("season", as_index=False).size().rename(columns={"size": "rows"})

    plt.figure(figsize=(10, 4))
    sns.barplot(data=season_counts, x="season", y="rows", color="#3b7a78")
    plt.title("Rows by Season")
    plt.xlabel("Season")
    plt.ylabel("Driver-race rows")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "rows_by_season.png", dpi=160)
    plt.close()


def save_top10_rate_by_season(df: pd.DataFrame) -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    season_target = df.groupby("season", as_index=False)[TARGET].mean()

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=season_target, x="season", y=TARGET, marker="o", color="#7a4f9f")
    plt.ylim(0, 1)
    plt.title("Top 10 Rate by Season")
    plt.xlabel("Season")
    plt.ylabel("Top 10 rate")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "top10_rate_by_season.png", dpi=160)
    plt.close()


def save_grid_vs_finish(df: pd.DataFrame) -> None:
    if "grid" not in df.columns or "final_position" not in df.columns:
        return

    plot_df = df[(df["grid"] > 0) & (df["final_position"] > 0)].copy()
    if plot_df.empty:
        return

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=plot_df,
        x="grid",
        y="final_position",
        hue=TARGET,
        alpha=0.35,
        palette="Set2",
        edgecolor=None,
    )
    plt.title("Grid Position vs Final Position")
    plt.xlabel("Grid position")
    plt.ylabel("Final position")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "grid_vs_finish.png", dpi=160)
    plt.close()


def save_model_comparison(model_comparison_path: Path) -> None:
    if not model_comparison_path.exists():
        return

    comparison = pd.read_csv(model_comparison_path)
    if comparison.empty:
        return

    plot_df = comparison.melt(
        id_vars=["model"],
        value_vars=["f1", "roc_auc", "race_precision_at_10"],
        var_name="metric",
        value_name="score",
    )

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.title("Model Comparison")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "model_comparison.png", dpi=160)
    plt.close()


def save_rolling_backtest(rolling_backtest_path: Path) -> None:
    if not rolling_backtest_path.exists():
        return

    rolling = pd.read_csv(rolling_backtest_path)
    if rolling.empty:
        return

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=rolling,
        x="test_season",
        y="race_precision_at_10",
        hue="model",
        marker="o",
    )
    plt.ylim(0, 1)
    plt.title("Rolling Backtest Precision@10")
    plt.xlabel("Test season")
    plt.ylabel("Race precision@10")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "rolling_backtest.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data}")

    df = pd.read_csv(args.data)
    if df.empty:
        raise ValueError(f"Dataset is empty: {args.data}")

    save_dataset_figures(df)
    save_rows_by_season(df)
    save_top10_rate_by_season(df)
    save_grid_vs_finish(df)
    save_model_comparison(args.model_comparison)
    save_rolling_backtest(args.rolling_backtest)

    print(f"Figures written to {FIGURES_PATH}")


if __name__ == "__main__":
    main()
