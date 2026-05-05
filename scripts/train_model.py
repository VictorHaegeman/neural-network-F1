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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from algorithms.classification import CLASSIFICATION_MODEL_NAMES, build_classification_pipeline


DATA_PATH = Path("data/final/f1_top10_model_dataset.csv")
MODEL_PATH = Path("outputs/models/top10_classifier.joblib")
METRICS_PATH = Path("outputs/metrics.json")
FIGURES_PATH = Path("outputs/figures")

TARGET = "top10_finish"
RANDOM_STATE = 42
MODEL_NAMES = CLASSIFICATION_MODEL_NAMES

LEAKAGE_COLUMNS = [
    TARGET,
    "final_position",
    "position_text",
    "points",
    "laps",
    "status",
    "fastest_lap_rank",
    "fastest_lap_number",
    "fastest_lap_time",
    "fastest_lap_seconds",
    "fastest_lap_avg_speed_kph",
    "safety_car_count",
    "virtual_safety_car_count",
    "red_flag_count",
    "yellow_flag_count",
    "double_yellow_count",
    "black_flag_count",
    "track_limits_count",
    "investigation_count",
    "penalty_count",
    "incident_count",
    "race_control_messages_count",
    "race_control_data_available",
    "total_dnf_count",
    "classified_driver_count",
    "race_disruption_score",
    "fastf1_race_control_available",
    "fastf1_race_control_messages_count",
    "fastf1_safety_car_count",
    "fastf1_virtual_safety_car_count",
    "fastf1_red_flag_count",
    "fastf1_yellow_flag_count",
    "fastf1_double_yellow_count",
    "fastf1_black_flag_count",
    "fastf1_track_limits_count",
    "fastf1_investigation_count",
    "fastf1_penalty_count",
    "fastf1_incident_count",
    "fastf1_drs_disabled_count",
    "fastf1_deleted_lap_count",
    "fastf1_slippery_track_count",
    "fastf1_clear_count",
    "fastf1_race_disruption_score",
]

NON_FEATURE_COLUMNS = [
    "race_id",
    "race_date",
    "driver_name",
    "driver_date_of_birth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a V0 model for F1 top-10 prediction.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--test-season", type=int, default=None)
    parser.add_argument("--model", choices=MODEL_NAMES, default="random_forest")
    parser.add_argument(
        "--min-test-rows",
        type=int,
        default=200,
        help="Minimum rows needed for the automatic temporal test season.",
    )
    return parser.parse_args()


def choose_test_season(df: pd.DataFrame, test_season: int | None, min_test_rows: int) -> int:
    if test_season is not None:
        return test_season

    season_counts = df.groupby("season").size().sort_index(ascending=False)
    for season, row_count in season_counts.items():
        previous_rows = int((df["season"] < season).sum())
        if row_count >= min_test_rows and previous_rows > 0:
            return int(season)

    return int(season_counts.index[0])


def latest_season_split(
    df: pd.DataFrame,
    test_season: int | None,
    min_test_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "season" not in df.columns:
        raise ValueError("The dataset needs a 'season' column for temporal validation.")

    season = choose_test_season(df, test_season, min_test_rows)
    train_df = df[df["season"] < season].copy()
    test_df = df[df["season"] == season].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(f"Unable to create train/test split with test season {season}.")

    return train_df, test_df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_columns = [col for col in LEAKAGE_COLUMNS + NON_FEATURE_COLUMNS if col in df.columns]
    return df.drop(columns=drop_columns)


def build_pipeline(X: pd.DataFrame, model_name: str = "random_forest") -> Pipeline:
    return build_classification_pipeline(X, model_name=model_name, random_state=RANDOM_STATE)


def top10_by_race_score(test_df: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    scored = test_df[["race_id", TARGET]].copy()
    scored["top10_probability"] = probabilities
    race_scores = []
    exact_matches = []

    for _, group in scored.groupby("race_id"):
        predicted_top10 = group.nlargest(min(10, len(group)), "top10_probability")
        hits = int(predicted_top10[TARGET].sum())
        actual_top10 = int(group[TARGET].sum())
        race_scores.append(hits / max(1, min(10, len(group))))
        exact_matches.append(int(hits == actual_top10 == min(10, len(group))))

    return {
        "race_precision_at_10": float(np.mean(race_scores)) if race_scores else 0.0,
        "exact_top10_set_rate": float(np.mean(exact_matches)) if exact_matches else 0.0,
    }


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Top 10 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "confusion_matrix.png", dpi=160)
    plt.close()


def save_feature_importance(model: Pipeline) -> None:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_).ravel()
    else:
        return

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(25)
    )

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    sns.barplot(data=importance_df, x="importance", y="feature", color="#2f6f9f")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "feature_importance.png", dpi=160)
    plt.close()


def save_dataset_figures(df: pd.DataFrame) -> None:
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x=TARGET, hue=TARGET, palette="Set2", legend=False)
    plt.title("Target Distribution")
    plt.xlabel("Top 10 Finish")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "target_distribution.png", dpi=160)
    plt.close()

    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].head(25)
    plt.figure(figsize=(8, 5))
    if missing.empty:
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
        plt.axis("off")
    else:
        sns.barplot(x=missing.values, y=missing.index, color="#8f5f5f")
        plt.xlabel("Missing values")
        plt.ylabel("")
    plt.title("Missing Values")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "missing_values.png", dpi=160)
    plt.close()

    numeric_df = df.select_dtypes(include=["number", "bool"])
    if TARGET in numeric_df.columns and len(numeric_df.columns) > 1:
        strongest = (
            numeric_df.corr(numeric_only=True)[TARGET]
            .abs()
            .sort_values(ascending=False)
            .head(25)
            .index
            .tolist()
        )
        corr = numeric_df[strongest].corr(numeric_only=True)
    else:
        corr = numeric_df.iloc[:, :25].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "correlation_matrix.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data}")

    df = pd.read_csv(args.data)
    if df.empty:
        raise ValueError(f"Dataset is empty: {args.data}")

    save_dataset_figures(df)

    train_df, test_df = latest_season_split(df, args.test_season, args.min_test_rows)
    X_train = build_features(train_df)
    y_train = train_df[TARGET].astype(int)
    X_test = build_features(test_df)
    y_test = test_df[TARGET].astype(int)

    evaluation_model = build_pipeline(X_train, args.model)
    evaluation_model.fit(X_train, y_train)

    predictions = evaluation_model.predict(X_test)
    probabilities = evaluation_model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": args.model,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_season": int(test_df["season"].iloc[0]),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
    }
    metrics.update(top10_by_race_score(test_df, probabilities))

    final_X = build_features(df)
    final_y = df[TARGET].astype(int)
    final_model = build_pipeline(final_X, args.model)
    final_model.fit(final_X, final_y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    save_confusion_matrix(y_test, predictions)
    save_feature_importance(evaluation_model)

    print("Training complete.")
    print(f"Model: {MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
