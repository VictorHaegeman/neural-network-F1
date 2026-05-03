from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_PATH = PROJECT_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete F1 top-10 prediction pipeline.")
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-prediction", action="store_true")
    parser.add_argument("--skip-pit-stops", action="store_true")
    parser.add_argument("--with-historical-weather", action="store_true")
    parser.add_argument("--with-race-control", action="store_true")
    parser.add_argument("--with-fastf1", action="store_true")
    parser.add_argument("--fastf1-start-year", type=int, default=2018)
    parser.add_argument("--fastf1-end-year", type=int, default=date.today().year)
    parser.add_argument("--fastf1-max-races", type=int, default=None)
    parser.add_argument(
        "--fastf1-force-refresh",
        action="store_true",
        help="Refetch FastF1 rows from scratch instead of extending existing local files.",
    )
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print("\n> " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    python = sys.executable

    raw_results = PROJECT_ROOT / "data" / "raw" / "race_results.csv"
    should_fetch = args.force_fetch or not raw_results.exists() or raw_results.stat().st_size == 0

    if should_fetch:
        fetch_command = [
            python,
            str(SCRIPTS_PATH / "generate_raw_data.py"),
            "--start-year",
            str(args.start_year),
            "--end-year",
            str(args.end_year),
        ]
        if args.skip_pit_stops:
            fetch_command.append("--skip-pit-stops")
        run_step(fetch_command)
    else:
        print(f"Using existing raw data: {raw_results}", flush=True)

    if args.with_fastf1:
        fastf1_command = [
            python,
            str(SCRIPTS_PATH / "generate_fastf1_features.py"),
            "--start-year",
            str(args.fastf1_start_year),
            "--end-year",
            str(args.fastf1_end_year),
        ]
        if args.fastf1_max_races is not None:
            fastf1_command.extend(["--max-races", str(args.fastf1_max_races)])
        if args.fastf1_force_refresh:
            fastf1_command.append("--force")
        else:
            fastf1_command.append("--incremental")
        run_step(fastf1_command)

    if args.with_historical_weather:
        run_step(
            [
                python,
                str(SCRIPTS_PATH / "generate_historical_weather.py"),
                "--start-year",
                str(args.start_year),
                "--end-year",
                str(args.end_year),
            ]
        )

    if args.with_race_control:
        race_control_command = [
            python,
            str(SCRIPTS_PATH / "generate_fastf1_race_control.py"),
            "--start-year",
            str(args.fastf1_start_year),
            "--end-year",
            str(args.fastf1_end_year),
            "--incremental",
        ]
        run_step(race_control_command)

    run_step([python, str(SCRIPTS_PATH / "generate_final_dataset.py")])
    run_step([python, str(SCRIPTS_PATH / "train_model.py"), "--model", args.model])

    if not args.skip_evaluation:
        run_step([python, str(SCRIPTS_PATH / "evaluate_models.py")])

    run_step([python, str(SCRIPTS_PATH / "make_charts.py")])

    if not args.skip_prediction:
        run_step([python, str(SCRIPTS_PATH / "predict_top10.py")])

    print("\nPipeline completed.", flush=True)


if __name__ == "__main__":
    main()
