from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_NONEMPTY_FILES = [
    "README.md",
    "requirements.txt",
    "docs/ASSIGNMENT_ALIGNMENT.md",
    "docs/PROJECT_PLAN.md",
    "docs/REMAINING_TASKS.md",
    "data/raw/fastf1_race_control.csv",
    "data/raw/race_control_history.csv",
    "data/raw/weather_data.csv",
    "data/raw/upcoming_qualifying_results.csv",
    "data/final/f1_top10_model_dataset.csv",
    "outputs/data_coverage_report.csv",
    "outputs/data_coverage_summary.json",
    "outputs/metrics.json",
    "outputs/model_comparison.csv",
    "outputs/neural_network_tuning.csv",
    "outputs/neural_network_summary.json",
    "outputs/position_model_comparison.csv",
    "outputs/position_model_metrics.json",
    "outputs/rolling_backtest.csv",
    "outputs/model_selection_summary.json",
    "outputs/neural_network_embedding_3d.csv",
    "outputs/figures/confusion_matrix.png",
    "outputs/figures/feature_importance.png",
    "outputs/figures/algorithm_holdout_summary.png",
    "outputs/figures/algorithm_validation_summary.png",
    "outputs/figures/assignment_pipeline_overview.png",
    "outputs/figures/model_metrics_table.png",
    "outputs/figures/model_comparison.png",
    "outputs/figures/neural_network_embedding_3d.html",
    "outputs/figures/neural_network_tuning.png",
    "outputs/figures/position_model_comparison.png",
    "outputs/figures/rolling_backtest.png",
    "outputs/predictions/position_predictions_holdout.csv",
    "outputs/predictions/top10_predictions_2026_04.csv",
    "outputs/predictions/upcoming_top10_predictions.csv",
    "outputs/predictions/upcoming_prediction_notes.json",
    "outputs/predictions/neural_network/upcoming_top10_predictions.csv",
    "outputs/predictions/neural_network/upcoming_prediction_notes.json",
    "notebooks/ML_Project_Code.ipynb",
    "report/Report.md",
    "report/Report.docx",
    "report/Report.pdf",
    "submission/IML_Assignment_GroupX.zip",
]

REQUIRED_SCRIPTS = [
    "scripts/run_pipeline.py",
    "scripts/generate_raw_data.py",
    "scripts/import_missing_completed_races.py",
    "scripts/generate_historical_weather.py",
    "scripts/generate_fastf1_features.py",
    "scripts/generate_fastf1_race_control.py",
    "scripts/generate_final_dataset.py",
    "scripts/train_model.py",
    "scripts/train_position_model.py",
    "scripts/tune_neural_network.py",
    "scripts/evaluate_models.py",
    "scripts/make_charts.py",
    "scripts/visualize_neural_network_3d.py",
    "scripts/open_neural_network_3d.py",
    "scripts/audit_data_coverage.py",
    "scripts/fetch_upcoming_qualifying.py",
    "scripts/predict_top10.py",
    "scripts/predict_upcoming_races.py",
    "scripts/build_report_docx.py",
    "scripts/build_report_pdf.py",
    "scripts/build_submission.py",
    "scripts/validate_project.py",
]

REQUIRED_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "race_precision_at_10",
]


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"[FAIL] {message}")


def pass_check(message: str) -> None:
    print(f"[OK] {message}")


def check_nonempty_files(failures: list[str]) -> None:
    for relative_path in REQUIRED_NONEMPTY_FILES:
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            fail(f"Missing file: {relative_path}", failures)
        elif path.stat().st_size == 0:
            fail(f"Empty file: {relative_path}", failures)
        else:
            pass_check(f"{relative_path} ({path.stat().st_size} bytes)")


def check_scripts(failures: list[str]) -> None:
    for relative_path in REQUIRED_SCRIPTS:
        path = PROJECT_ROOT / relative_path
        if not path.exists() or path.stat().st_size == 0:
            fail(f"Missing or empty script: {relative_path}", failures)
        else:
            pass_check(f"{relative_path}")


def check_dataset(failures: list[str]) -> None:
    path = PROJECT_ROOT / "data/final/f1_top10_model_dataset.csv"
    if not path.exists() or path.stat().st_size == 0:
        return

    df = pd.read_csv(path)
    if len(df) < 6000:
        fail(f"Dataset has too few rows: {len(df)}", failures)
    else:
        pass_check(f"Dataset rows: {len(df)}")

    if len(df.columns) < 190:
        fail(f"Dataset has too few columns: {len(df.columns)}", failures)
    else:
        pass_check(f"Dataset columns: {len(df.columns)}")

    missing_count = int(df.isna().sum().sum())
    if missing_count:
        fail(f"Dataset contains missing values: {missing_count}", failures)
    else:
        pass_check("Dataset has no missing values")

    seasons = sorted(df["season"].dropna().astype(int).unique())
    if seasons[0] > 2011 or seasons[-1] < 2025:
        fail(f"Unexpected season coverage: {seasons[0]}-{seasons[-1]}", failures)
    else:
        pass_check(f"Season coverage: {seasons[0]}-{seasons[-1]}")


def check_weather_data(failures: list[str]) -> None:
    path = PROJECT_ROOT / "data/raw/weather_data.csv"
    if not path.exists() or path.stat().st_size == 0:
        return

    weather = pd.read_csv(path)
    if "weather_data_available" not in weather.columns:
        fail("weather_data.csv is missing weather_data_available", failures)
        return

    available = int(weather["weather_data_available"].fillna(0).sum())
    if len(weather) < 300:
        fail(f"Weather table has too few races: {len(weather)}", failures)
    else:
        pass_check(f"Weather race rows: {len(weather)}")

    if available < 300:
        fail(f"Weather table has too few available rows: {available}", failures)
    else:
        pass_check(f"Weather rows available: {available}")


def check_race_control_data(failures: list[str]) -> None:
    path = PROJECT_ROOT / "data/raw/fastf1_race_control.csv"
    if not path.exists() or path.stat().st_size == 0:
        return

    race_control = pd.read_csv(path)
    if "fastf1_race_control_available" not in race_control.columns:
        fail("fastf1_race_control.csv is missing fastf1_race_control_available", failures)
        return

    available = int(race_control["fastf1_race_control_available"].fillna(0).sum())
    if available < 150:
        fail(f"FastF1 race-control has too few available rows: {available}", failures)
    else:
        pass_check(f"FastF1 race-control rows available: {available}")

    history_path = PROJECT_ROOT / "data/raw/race_control_history.csv"
    if history_path.exists() and history_path.stat().st_size > 0:
        history = pd.read_csv(history_path)
        if len(history) < 300:
            fail(f"Race-control history has too few rows: {len(history)}", failures)
        else:
            pass_check(f"Race-control history rows: {len(history)}")


def check_position_model_outputs(failures: list[str]) -> None:
    path = PROJECT_ROOT / "outputs/position_model_metrics.json"
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open(encoding="utf-8") as file:
        metrics = json.load(file)

    best = metrics.get("best_holdout", {})
    required = ["race_precision_at_10", "actual_top10_rank_mae", "mean_spearman_by_race"]
    for metric in required:
        if metric not in best:
            fail(f"Position model missing metric: {metric}", failures)
        else:
            pass_check(f"position {metric}: {best[metric]}")

    predictions_path = PROJECT_ROOT / "outputs/predictions/upcoming_top10_predictions.csv"
    if predictions_path.exists() and predictions_path.stat().st_size > 0:
        predictions = pd.read_csv(predictions_path)
        if "predicted_finish_rank" not in predictions.columns:
            fail("Upcoming predictions are missing predicted_finish_rank", failures)
        else:
            pass_check("Upcoming predictions include predicted_finish_rank")


def check_metrics(failures: list[str]) -> None:
    path = PROJECT_ROOT / "outputs/metrics.json"
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open(encoding="utf-8") as file:
        metrics = json.load(file)

    for metric in REQUIRED_METRICS:
        if metric not in metrics:
            fail(f"Missing metric: {metric}", failures)
        elif metrics[metric] is None:
            fail(f"Metric is null: {metric}", failures)
        else:
            pass_check(f"{metric}: {metrics[metric]}")


def check_submission_zip(failures: list[str]) -> None:
    path = PROJECT_ROOT / "submission/IML_Assignment_GroupX.zip"
    if not path.exists() or path.stat().st_size == 0:
        return

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())

    expected = {
        "README.md",
        "requirements.txt",
        "report/Report.md",
        "report/Report.docx",
        "report/Report.pdf",
        "notebooks/ML_Project_Code.ipynb",
        "data/final/f1_top10_model_dataset.csv",
        "scripts/run_pipeline.py",
    }
    for name in expected:
        if name not in names:
            fail(f"ZIP missing: {name}", failures)
        else:
            pass_check(f"ZIP includes {name}")

    forbidden_fragments = [".venv/", ".fastf1_cache/", "outputs/models/", "__pycache__/"]
    for name in names:
        if any(fragment in name for fragment in forbidden_fragments):
            fail(f"ZIP contains forbidden path: {name}", failures)
            break
    else:
        pass_check("ZIP excludes caches, venv and generated model binaries")


def main() -> None:
    failures: list[str] = []

    check_nonempty_files(failures)
    check_scripts(failures)
    check_dataset(failures)
    check_weather_data(failures)
    check_race_control_data(failures)
    check_metrics(failures)
    check_position_model_outputs(failures)
    check_submission_zip(failures)

    if failures:
        print(f"\nValidation failed with {len(failures)} issue(s).")
        sys.exit(1)

    print("\nProject validation passed.")


if __name__ == "__main__":
    main()
