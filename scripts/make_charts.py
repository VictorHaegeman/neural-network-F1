from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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


def save_algorithm_holdout_summary(model_comparison_path: Path) -> None:
    if not model_comparison_path.exists():
        return

    comparison = pd.read_csv(model_comparison_path)
    if comparison.empty:
        return

    comparison = comparison.sort_values("race_precision_at_10", ascending=False)
    plot_df = comparison.melt(
        id_vars=["model"],
        value_vars=["race_precision_at_10", "f1", "roc_auc"],
        var_name="metric",
        value_name="score",
    )

    metric_labels = {
        "race_precision_at_10": "Race precision@10",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
    }
    plot_df["metric"] = plot_df["metric"].map(metric_labels)

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    fig, (ax, note_ax) = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        gridspec_kw={"width_ratios": [2.4, 1]},
    )
    sns.barplot(data=plot_df, x="model", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Holdout Algorithm Comparison (2025)")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(title="")

    best = comparison.iloc[0]
    note_ax.axis("off")
    note = (
        "How to read this figure:\n\n"
        "Each algorithm is trained on seasons 2010-2024 and tested on 2025.\n\n"
        "Race precision@10 measures how many of the predicted top 10 drivers "
        "actually finished in the top 10 for each race.\n\n"
        f"Best holdout model: {best['model']}\n"
        f"Race precision@10: {best['race_precision_at_10']:.3f}\n"
        f"F1: {best['f1']:.3f}\n"
        f"ROC-AUC: {best['roc_auc']:.3f}"
    )
    note_ax.text(
        0,
        1,
        note,
        va="top",
        ha="left",
        fontsize=10,
        linespacing=1.25,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f5f2e8", "edgecolor": "#c7bfa7"},
    )
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "algorithm_holdout_summary.png", dpi=180)
    plt.close(fig)


def save_model_metrics_table(model_comparison_path: Path) -> None:
    if not model_comparison_path.exists():
        return

    comparison = pd.read_csv(model_comparison_path)
    if comparison.empty:
        return

    columns = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "race_precision_at_10"]
    table_df = comparison[columns].copy()
    table_df = table_df.sort_values("race_precision_at_10", ascending=False)
    for column in columns[1:]:
        table_df[column] = table_df[column].map(lambda value: f"{value:.3f}")
    table_df = table_df.rename(
        columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "roc_auc": "ROC-AUC",
            "race_precision_at_10": "Race P@10",
        }
    )

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    ax.set_title("Model Validation Metrics on 2025 Holdout", pad=18, fontsize=14, weight="bold")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#b8b8b8")
        if row == 0:
            cell.set_facecolor("#2f5d62")
            cell.set_text_props(weight="bold", color="white")
        elif row == 1:
            cell.set_facecolor("#eef6f4")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f7f7f7")

    fig.text(
        0.5,
        0.04,
        "The table is sorted by race precision@10, the metric most aligned with predicting a race-level top 10.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "model_metrics_table.png", dpi=180)
    plt.close(fig)


def save_assignment_pipeline_overview() -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    stages = [
        ("Raw sources", "Jolpica F1 API\nOpen-Meteo\nFastF1 optional"),
        ("Cleaning", "Identifier matching\nmissing data handling\navailability flags"),
        ("Final dataset", "6,999 driver-race rows\n198 variables\n0 missing values"),
        ("Algorithms", "Logistic regression\nRandom forest\nExtra trees\nHGB\nMLP"),
        ("Validation", "2025 holdout\nrolling backtest\nrace precision@10"),
        ("Outputs", "metrics CSV/JSON\nPNG figures\nDOCX/PDF report"),
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x_positions = [0.08, 0.245, 0.41, 0.575, 0.74, 0.905]
    colors = ["#dbe9f6", "#e6f2df", "#fff0c9", "#f1e4f6", "#e8edf7", "#f7e2d4"]

    for index, ((title, body), x, color) in enumerate(zip(stages, x_positions, colors)):
        width = 0.13
        box = FancyBboxPatch(
            (x - width / 2, 0.36),
            width,
            0.32,
            boxstyle="round,pad=0.025,rounding_size=0.02",
            facecolor=color,
            edgecolor="#555555",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x, 0.61, title, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(x, 0.48, body, ha="center", va="center", fontsize=9, linespacing=1.2)
        if index < len(x_positions) - 1:
            arrow = FancyArrowPatch(
                (x + width / 2 + 0.012, 0.52),
                (x_positions[index + 1] - width / 2 - 0.012, 0.52),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.2,
                color="#555555",
            )
            ax.add_patch(arrow)

    ax.text(
        0.5,
        0.82,
        "Reproducible Machine Learning Pipeline for the Assignment",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
    )
    ax.text(
        0.5,
        0.2,
        "This figure links the required assignment steps: data collection, preprocessing, model implementation, validation and final report evidence.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "assignment_pipeline_overview.png", dpi=180)
    plt.close(fig)


def save_algorithm_validation_summary(rolling_backtest_path: Path) -> None:
    if not rolling_backtest_path.exists():
        return

    rolling = pd.read_csv(rolling_backtest_path)
    if rolling.empty:
        return

    average = (
        rolling.groupby("model", as_index=False)["race_precision_at_10"]
        .mean()
        .sort_values("race_precision_at_10", ascending=False)
    )
    best = average.iloc[0]

    fig, (line_ax, bar_ax) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [2, 1]})
    sns.lineplot(
        data=rolling,
        x="test_season",
        y="race_precision_at_10",
        hue="model",
        marker="o",
        ax=line_ax,
    )
    line_ax.set_ylim(0, 1)
    line_ax.set_title("Expanding-Window Backtest")
    line_ax.set_xlabel("Test season")
    line_ax.set_ylabel("Race precision@10")
    line_ax.legend(title="", fontsize=8)

    sns.barplot(data=average, x="race_precision_at_10", y="model", color="#637db4", ax=bar_ax)
    bar_ax.set_xlim(0, 1)
    bar_ax.set_title("Average Stability")
    bar_ax.set_xlabel("Average race precision@10")
    bar_ax.set_ylabel("")
    bar_ax.text(
        0,
        -0.8,
        textwrap.fill(
            f"Best average stability: {best['model']} ({best['race_precision_at_10']:.3f}). "
            "This protects the project from relying only on a single random split.",
            width=42,
        ),
        fontsize=9,
        ha="left",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "algorithm_validation_summary.png", dpi=180)
    plt.close(fig)


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
    save_algorithm_holdout_summary(args.model_comparison)
    save_model_metrics_table(args.model_comparison)
    save_rolling_backtest(args.rolling_backtest)
    save_algorithm_validation_summary(args.rolling_backtest)
    save_assignment_pipeline_overview()

    print(f"Figures written to {FIGURES_PATH}")


if __name__ == "__main__":
    main()
