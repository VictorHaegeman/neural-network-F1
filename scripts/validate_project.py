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
    "docs/PROJECT_PLAN.md",
    "docs/REMAINING_TASKS.md",
    "data/final/f1_top10_model_dataset.csv",
    "outputs/metrics.json",
    "outputs/model_comparison.csv",
    "outputs/rolling_backtest.csv",
    "outputs/model_selection_summary.json",
    "outputs/figures/confusion_matrix.png",
    "outputs/figures/feature_importance.png",
    "outputs/figures/model_comparison.png",
    "outputs/figures/rolling_backtest.png",
    "outputs/predictions/top10_predictions_2025_24.csv",
    "notebooks/ML_Project_Code.ipynb",
    "report/Report.md",
    "report/Report.docx",
    "submission/IML_Assignment_GroupX.zip",
]

REQUIRED_SCRIPTS = [
    "scripts/run_pipeline.py",
    "scripts/generate_raw_data.py",
    "scripts/generate_fastf1_features.py",
    "scripts/generate_final_dataset.py",
    "scripts/train_model.py",
    "scripts/evaluate_models.py",
    "scripts/make_charts.py",
    "scripts/predict_top10.py",
    "scripts/build_report_docx.py",
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

    if len(df.columns) < 150:
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
    check_metrics(failures)
    check_submission_zip(failures)

    if failures:
        print(f"\nValidation failed with {len(failures)} issue(s).")
        sys.exit(1)

    print("\nProject validation passed.")


if __name__ == "__main__":
    main()
